# PRSS2 v2.0 开发日志

> PRSS (Predictive Relation-State Sheaf) — 递归时序模型接口的预测性谱压缩
> v2.0:标准工程架构 + 可插拔压缩算法 + TGB 协议接入 + 实验框架
> 本文记叙系统架构、相对 old/ 归档版的修改点、以及双环境使用方法

---

## 1. 项目定位

PRSS 寄生在宿主时序模型的递归聚合接口上:候选状态 h(d 维,默认 256)经商矩阵
R(k×d,k=宿主宽度)投影为 z=R·h,父节点只接收 z。R 由"未来读取矩阵 B(C)"的预测
Gram G=E[BᵀB] 的 top-k 特征分解周期更新(谱商)。训练损失 =
L_task + λ_resp·L_resp + λ_spec·L_spec;reader/outside/Gram/SVD 仅训练期存在。

v2.0 目标(甲方要求):把方法做成**换数据集、换压缩器、换种子都能一键批量跑**的
标准工程,用 TGB 公开协议(tgbl-wiki 链路预测,mrr)取代自建协议,为"多数据集泛化性"
论文主张搭建实验基建。

## 2. 系统架构

```
scripts/train.py ── scripts/inference.py        # 入口 CLI
        │
        ▼
prss/experiment/  (YAML 矩阵 → runner → summarize)
        │
        ▼
prss/training/    (event_loop / checkpoint / isolation)
        │
        ▼
prss/hosts/       (PyG-TGN 宿主适配 + TGB 链路桥 + vendored 基线模块)
        │
        ▼
prss/data/        (TGBLinkDataset 封装)
        │
        ▼
prss/ 核心包(config/state/losses/spectral/reader/outside/candidate/auxiliary/core/monitoring)
        │
        └── prss/compressors/  (vanilla/random/pca/direct/spectral 可插拔)
```

依赖方向自上而下;**核心包零宿主依赖**(无 "tgn_layer_"、无 TGB 字符串,测试断言),
宿主耦合全部隔离在 `hosts/`;TGB/PyG 为可选依赖(核心包无 torch_geometric 也能导入)。

### 逻辑分层

| 层 | 职责 | 关键文件 |
|:---|:---|:---|
| 入口脚本 | CLI、组装、落盘 | `scripts/train.py` / `scripts/inference.py` |
| 实验框架 | 矩阵展开、幂等调度、汇总 | `prss/experiment/runner.py` / `summarize.py` |
| 训练组件 | 训练循环、隔离审计、断点续跑 | `prss/training/*` |
| 宿主层 | TGN 契约、计算树 trace、链路桥 | `prss/hosts/tgn_pyg.py` / `tgn_pyg_bridge.py` |
| 数据层 | TGB 下载/切分/负采样 | `prss/data/tgb_link.py` |
| 核心层 | 谱商、读取器、outside、监控 | `prss/*.py` |
| 压缩器层 | R 的来源(5 变体) | `prss/compressors/*.py` |

## 3. 目录结构

```
PRSS2/
├── pyproject.toml / requirements.txt     # pip install -e . ; tgb 可选 extra
├── src/prss/
│   ├── config.py state.py losses.py spectral.py reader.py outside.py
│   ├── candidate.py auxiliary.py core.py monitoring.py
│   ├── compressors/{base,vanilla,random,pca,direct,spectral}.py
│   ├── hosts/{base,tgn_pyg,tgn_pyg_bridge}.py
│   ├── hosts/pyg_models/                 # vendored TGB 基线(标注上游 py-tgb 2.3.0)
│   ├── data/tgb_link.py
│   ├── training/{event_loop,checkpoint,isolation}.py
│   └── experiment/{runner,summarize}.py
├── scripts/{train,inference}.py
├── test/                                 # 全部 unittest.TestCase,unittest/pytest 双兼容
├── configs/experiments/{tgbl_wiki_smoke,tgbl_wiki_matrix}.yaml
├── datasets/  outputs/                   # .gitignore 已排除
└── old/                                  # 归档旧实现,只读参考
```

## 4. 相对 old/ 的修改点(五阶段变更)

### 4.1 工程架构(阶段 1)
- 核心包按 τ(str) 索引,hosts/ 隔离全部宿主耦合;`PRSSCore(config, variant)` 注入压缩器。
- 移植:新版谱算法(SpectralQuotient:精确 eigh 目标 + Grassmann 信任域回溯 + L_spec
  部署门控 + 快照全集)、新版 reader(恢复 response_dim 可配)、新版 MonitorWriter
  (整数精确有限性计数)、旧版 config/state/losses/outside_context(τ 化)。
- 重写:core.py(τ 键 + Compressor 注入)、candidate.py(GenericResidualCandidateBuilder,
  打包逻辑上移宿主;raw==candidate 接口直通)。
- 修复旧实现隐患:测试统一 unittest.TestCase(pytest/unittest 双发现一致)。

### 4.2 可插拔压缩器(阶段 2)
5 个变体,纯数据契约(statistic / use_response_loss / use_spectral_loss /
update_projection),注册表 + `build_compressor` 工厂:

| 变体 | statistic | resp | spec | update | R |
|:---|:---|:---|:---|:---|:---|
| vanilla | none | F | F | F | buffer [I,0] 永不变 |
| random | none | T | F | F | buffer 随机半正交,冻结 |
| pca | pca(中心化协方差) | T | F | T | buffer;精确主子空间 |
| direct | none | T | F | F(梯度) | **Parameter** |
| spectral | reader | T | T | T | buffer;完整谱商 |

红线:`set_projection_trainable(True)` 在 spectral/random/pca/vanilla 上抛错;direct
快照标记 `projection_expected_orthogonal: False`(监控跳过正交不变量)。
新增变体 = 继承 `Compressor` + `@register_variant`,入口 `--variant` 即插即用。

### 4.3 TGB 协议接入(阶段 3)
- **宿主换为 PyG 版 TGN**(TGNMemory + GraphAttentionEmbedding 单层子图卷积,官方
  TGB 基线结构):PRSS 接口钉在 "conv 输出 → LinkPredictor 输入" 一处,
  τ=`tgp:node_conv`,host_dim=emb_dim(100),候选 256。
- outside 计算树**深度 = 1**(子图卷积无递归层;occurrence 子节点 = 采样邻域节点),
  occurrence 池按 node 去重,多根共享。
- 链路预测 4 场景桥(照搬旧版 TGNLinkOutsideBridge 语义):(src树,正dst,0,1)、
  (dst树,src,1,1)、(src树,负dst,0,0)、(负树,src,1,0);root_metadata =
  [对端商 detach;归一化 log 时间;角色],label 只进 loss。
- 基线模块 vendor 进 `hosts/pyg_models/`(上游 repo 根导入布局用 `modules` 别名包垫片
  兼容;`msg_agg.py` 与 `memory_module.py:176` 各一处**注明文档的本地补丁**:
  torch_scatter → torch 原生 scatter_reduce,语义等价)。
- 训练/评估时序照官方 TGB tgn.py(每 epoch 重置 memory/loader;先算分后
  update_state+insert+detach);验证/测试三零隔离审计(计数/R/trace);
  测试协议:零 memory 重放 train+val 后测 test(比官方"沿用末轮 val 态"更干净)。
- rolling checkpoint 显式序列化 TGNMemory 的 msg_s_store/msg_d_store(不在
  state_dict 内)+ 三源 RNG 恢复。

### 4.4 实验框架(阶段 4)
- YAML 矩阵(defaults + matrix 笛卡尔积)→ runner 幂等调度(_SUCCESS 跳过、失败归档
  `*__failed__*` 不中断)→ summarize 输出 mean±std MRR 表 + spectral 机制指标列。
- 两个配置:`tgbl_wiki_smoke.yaml`(1 epoch×1 seed×5 变体,工程冒烟)、
  `tgbl_wiki_matrix.yaml`(30 epoch×3 seed×5 变体,正式)。

### 4.5 丢弃的设计
旧版二分 alpha 试步谱更新(被新版 Grassmann 回溯取代)、TGN 注意力池化候选、
TensorBoard 监控体系、reader response_dim=1 硬编码、核心内 TGN 形状推导。

## 5. 红线清单与审计位置(不可破坏)

| 红线 | 实现/审计位置 |
|:---|:---|
| 从第 0 步压缩,R=[I,0] 初始化平价 | `SpectralQuotient` init;`test_tgb_smoke` 平价测试 |
| spectral 是唯一主方法谱求解 | `compressors/`;`test_no_alternate_reduction`(无 sklearn 分解/PCA 主路径) |
| R 是 buffer(除 direct 变体) | `SpectralQuotient.register_buffer`;`test_spectral` R 非参数 |
| reader/outside 仅训练期 | `event_loop` evaluate/replay 前 clear_trace + 门控 |
| 验证/测试不更新 Gram/SVD/R | `prss/training/isolation.py` assert_clean 三零审计 |
| label 不进 context | `TGBLinkOutsideBridge`(label 只进 loss) |

## 6. 双宿主语义差异(当前实现 vs 归档 twitter 宿主)

| | PyG-TGN(当前宿主) | twitter-TGN(old/ 参考) |
|:---|:---|:---|
| 聚合结构 | 单层 TransformerConv 子图卷积 | L 层递归时间注意力树 |
| τ 集合 | {`tgp:node_conv`} | {`tgw:layer_0..L`} |
| outside 树深度 | 1(采样邻域) | L(递归计算树) |
| 任务 | 链路预测(mrr) | 节点分类(AUC/AP) |
| 训练协议 | TGB 官方(联合训练全部组件) | 官方 TGN(自监督预训练+冻结) |

twitter 宿主在新架构中的接入点由 `hosts/base.py` 抽象预留,适配器迁移列为下一步
(见 §9)。PRSS 核心不感知这些差异——这就是 τ 抽象的目的。

## 7. 双环境安装与运行

### 7.1 安装

```bash
# 核心(Windows CPU / AutoDL GPU 同)
pip install -e .
pip install pyyaml pytest

# torch:按环境选 wheel(先装!)
#   Windows CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
#   AutoDL GPU : pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# TGB 协议(阶段 3 起;先 torch 再 PyG 再 tgb)
pip install torch-geometric
pip install tgb==2.3.0
```

数据下载由 `TGBLinkDataset` 首次使用自动完成;目录用环境变量 `TGB_ROOT` 控制
(默认 `datasets/`,AutoDL 可指向挂载盘)。

### 7.2 单任务训练

```bash
python -m scripts.train --variant spectral --dataset tgbl-wiki --seed 0 \
    --output outputs/tgbl-wiki/tgbl-wiki__spectral__seed000 --gpu 0
# 冒烟截断:追加 --max-train 2000 --max-val 500 --max-test 500
```

### 7.3 矩阵实验

```bash
python -m prss.experiment.runner --config configs/experiments/tgbl_wiki_smoke.yaml --gpu 0
python -m prss.experiment.runner --config configs/experiments/tgbl_wiki_matrix.yaml --gpu 0
# 重跑自动跳过已完成 job;失败归档不中断;--only <job_id> 单跑
python -m prss.experiment.summarize outputs/tgbl_wiki
```

### 7.4 推理

```bash
python -m scripts.inference \
    --checkpoint outputs/tgbl-wiki/tgbl-wiki__spectral__seed000/best.pt \
    --split test --output outputs/.../inference_test.json
```

### 7.5 测试

```bash
python -m pytest test -q      # unittest 也兼容:python -m unittest discover -s test
```

## 8. 变更日志

- **2026-08-18 阶段 1(骨架+核心移植)**:pyproject/requirements、src/prss 核心包、
  compressors base/vanilla/spectral、test 三件套;**61 项测试全绿**(谱 SVD 等价、
  τ 键契约、[I,0] 平价、辅助损失契约、核心源码零宿主硬编码断言)。
- **2026-08-18 阶段 2(可插拔压缩器)**:5 变体 + 注册表 + 红线;合成树已知子空间
  门禁通过(spectral 恢复主角度 ≤0.20 rad;direct 基线击败 random)。
- **2026-08-18 阶段 3(TGB 接入)**:pyg_models vendor(2 处注明补丁)、TGBLinkDataset、
  PyGTGNAdapter + TGBLinkOutsideBridge、event_loop/checkpoint/isolation、
  scripts/train.py + inference.py;PyG 宿主 [I,0] 平价、trace/bridge、推理隔离三测试
  通过(本机 torch 2.10 CPU + PyG 2.8 实测)。
- **2026-08-18 阶段 4(实验框架)**:runner/summarize + 2 个 YAML;矩阵展开/幂等格式/
  汇总表单元测试通过。**待办:AutoDL 上跑 smoke 与正式矩阵(需 tgb 包与 GPU)**。

## 9. 已知限制与下一步

1. **tgbl-wiki 单数据集先行**:数据集维度已参数化(name 参数),uci/enron 接入代码零
   增量;enron 3054 万边的算力与内存需单独评估。
2. **λ_spec 敏感性未跑**:规格要求 {0, 0.01, 0.1, 0.5, 1.0},矩阵可加一维。
3. **twitter-TGN 宿主适配器未迁移**:hosts/base.py 抽象已预留;迁移时从
   old/PRSS-method/prss/tgn_adapter.py 移植并 τ 化。
4. **官方锚点数值待复现**:tgbl-wiki 上 vanilla(无 PRSS)MRR 应与 TGB 官方基线对齐,
   作为全部变体的 anchor(首次在 AutoDL 跑 smoke 时记录)。
5. **PyG 宿主 outside 树深度=1**:子图卷积无递归层;深度大于 1 的上下文需要修改
   neighbor loader 或换宿主,论文中需如实声明。
6. **TGBMemory 消息队列 checkpoint**:已由 CheckpointManager 显式序列化;DyRepMemory
   同法处理(当前未用)。
7. **多种子统计口径**:正式矩阵 3 seed 为最小集;论文主张建议 5 seed 对齐 TGB 基线惯例。
