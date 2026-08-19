# PRSS2 v2.0 开发日志

> PRSS (Predictive Relation-State Sheaf) — 递归时序模型接口的预测性谱压缩
> v2.0:标准工程架构 + 可插拔压缩算法 + TGB 协议接入 + 实验框架
> 本文记叙系统架构、相对 old/ 归档版的修改点、部署战报、以及双环境使用方法
> 最后附**代码阅读顺序指导**(供逐行审阅)

---

## 1. 项目定位

PRSS 寄生在宿主时序模型的递归聚合接口上:候选状态 h(d 维,默认 256)经商矩阵
R(k×d,k=宿主宽度)投影为 z=R·h,父节点只接收 z。R 由"未来读取矩阵 B(C)"的预测
Gram G=E[BᵀB] 的 top-k 特征分解周期更新(谱商)。训练损失 =
L_task + λ_resp·L_resp + λ_spec·L_spec;reader/outside/Gram/SVD 仅训练期存在。

v2.0 目标(甲方要求):把方法做成**换数据集、换压缩器、换种子都能一键批量跑**的
标准工程,用 TGB 公开协议(tgbl-wiki 链接预测,mrr)取代自建协议,为"多数据集泛化性"
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
├── pyproject.toml / requirements.txt     # pip install -e . ; tgb/tensorboard 可选
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
├── notes/                                # 规格书、研究索引、调研文档、交接备忘
└── (old/ 已删除,git 未提交过,无法恢复)
```

## 4. 相对 old/ 的修改点

### 4.1 工程架构
- 核心包按 τ(str) 索引,hosts/ 隔离全部宿主耦合;`PRSSCore(config, variant)` 注入压缩器。
- 移植:新版谱算法(SpectralQuotient:精确 eigh 目标 + Grassmann 信任域回溯 + L_spec
  部署门控 + 快照全集)、新版 reader(恢复 response_dim 可配)、新版 MonitorWriter
  (整数精确有限性计数)、旧版 config/state/losses/outside_context(τ 化)。
- 重写:core.py(τ 键 + Compressor 注入)、candidate.py(GenericResidualCandidateBuilder,
  打包逻辑上移宿主;raw==candidate 接口直通)。
- 测试统一 unittest.TestCase(修复旧版 unittest/pytest 双机制混用导致静默跳过)。

### 4.2 可插拔压缩器(5 变体)

| 变体 | statistic | resp | spec | update | R |
|:---|:---|:---|:---|:---|:---|
| vanilla | none | F | F | F | 不挂 PRSS(纯宿主基线) |
| random | none | T | F | F | buffer 随机半正交,冻结 |
| pca | pca(中心化协方差) | T | F | T | buffer;精确主子空间(step=1.0) |
| direct | none | T | F | F(梯度) | **Parameter** |
| spectral | reader | T | T | T | buffer;完整谱商 |

红线:`set_projection_trainable(True)` 在 spectral/random/pca/vanilla 上抛错;direct
快照标记 `projection_expected_orthogonal: False`。新增变体 = 继承 `Compressor` +
`@register_variant`,入口 `--variant` 即插即用。

### 4.3 TGB 协议接入
- **宿主 = PyG 版 TGN**(TGNMemory + GraphAttentionEmbedding 单层子图卷积,官方
  TGB 基线结构):PRSS 接口钉在 "conv 输出 → LinkPredictor 输入" 一处,
  τ=`tgp:node_conv`,host_dim=emb_dim(100),候选 256。
- outside 计算树**深度 = 1**(子图卷积无递归层;occurrence 子节点 = 采样邻域节点),
  occurrence 池按 node 去重,多根共享。
- 链路预测 4 场景桥(照搬旧版 TGNLinkOutsideBridge 语义):(src树,正dst,0,1)、
  (dst树,src,1,1)、(src树,负dst,0,0)、(负树,src,1,0);root_metadata =
  [对端商 detach;归一化 log 时间;角色],label 只进 loss。
- 训练/评估时序照官方 TGB tgn.py(每 epoch 重置 memory/loader;先算分后
  update_state+insert+detach);**验证/测试期切换 eval 模式**(关 dropout、memory 只读,
  官方评估语义);三零隔离审计(计数/R/trace)。
- 测试协议:零 memory 重放 train+val 后测 test;rolling checkpoint 显式序列化
  TGNMemory 的 msg_s_store/msg_d_store + 三源 RNG 恢复。
- **冻结宿主开关**(`--freeze-host`,默认关):memory+gnn `requires_grad=False` +
  保持 eval 模式,link_pred 与 PRSS 模块照常训练——旧架构语义的可开关版。

### 4.4 实验框架与观测
- YAML 矩阵(defaults + matrix 笛卡尔积)→ runner 幂等调度(_SUCCESS 跳过、失败归档
  不中断;布尔值 True 才传旗标)→ summarize 输出 mean±std MRR 表 + 机制指标列。
- **评估指标**:主指标 mrr(官方 one-vs-many);补充 one-vs-many `auc_ovm`/`ap_ovm`
  (官方负样本上的受限口径,键名带 `_ovm` 防与旧节点分类 AUC 混淆)。
- **TensorBoard**:`outputs/<job>/tb/` 事件文件——每 `monitor_every` 批记
  `train/task_loss`,每 epoch 记 `epoch/train_task_loss` + `epoch/val_*`,
  结束记 `final/test_*`;tensorboard 缺失时自动降级为警告。

### 4.5 丢弃的设计
旧版二分 alpha 试步谱更新(被新版 Grassmann 回溯取代)、TGN 注意力池化候选、
TensorBoard 之外的旧监控体系、reader response_dim=1 硬编码、核心内 TGN 形状推导。

## 5. 红线清单与审计位置(不可破坏)

| 红线 | 实现/审计位置 |
|:---|:---|
| 从第 0 步压缩,R=[I,0] 初始化平价 | `SpectralQuotient` init;`test_tgb_smoke` 平价测试 |
| spectral 是唯一主方法谱求解 | `compressors/`;测试断言无 sklearn 分解/PCA 主路径 |
| R 是 buffer(除 direct 变体) | `SpectralQuotient.register_buffer`;`test_spectral` R 非参数 |
| reader/outside 仅训练期 | `event_loop` evaluate/replay 前 clear_trace + 门控 |
| 验证/测试不更新 Gram/SVD/R | `prss/training/isolation.py` assert_clean 三零审计 |
| label 不进 context | `TGBLinkOutsideBridge`(label 只进 loss) |
| 禁止未压缩预热 | 无 vanilla 预热阶段;谱统计预热 = spectral_warmup(合规) |

## 6. 双宿主语义差异(当前实现 vs 归档 twitter 宿主)

| | PyG-TGN(当前宿主) | twitter-TGN(old/ 参考) |
|:---|:---|:---|
| 聚合结构 | 单层 TransformerConv 子图卷积 | L 层递归时间注意力树 |
| τ 集合 | {`tgp:node_conv`} | {`tgw:layer_0..L`} |
| outside 树深度 | 1(采样邻域;无向图,防环靠深度=1+visited) | L(递归树,天然无环) |
| 任务 | 链路预测(mrr/auc_ovm/ap_ovm) | 节点分类(AUC/AP) |
| 训练协议 | TGB 官方(联合训练;可选冻结宿主) | 官方 TGN(自监督预训练+冻结) |

twitter 宿主迁移由 `hosts/base.py` 抽象预留(DEVELOPMENT 下一步清单)。

## 7. 双环境安装与运行

### 7.1 安装

```bash
pip install -e .
pip install pyyaml pytest
# torch:按环境选 wheel(先装!)
#   Windows CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
#   GPU 实例  : pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
pip install py-tgb==2.3.0 --no-deps      # 注意包名是 py-tgb;--no-deps 后需补:
pip install clint tqdm requests matplotlib pytz python-dateutil urllib3 filelock
pip install tensorboard                  # 可选:训练曲线
```

### 7.2 数据放置(重要)

TGB 用 `PROJ_DIR + root` **字符串拼接**,`TGBLinkDataset` 已把 PROJ_DIR 指向仓库根、
只传目录名。**数据文件必须放在 `<仓库>/datasets/tgbl_wiki/` 子目录**(下划线名):
`tgbl-wiki_edgelist_v2.csv`(存在即跳过下载)+ `tgbl-wiki_{val,test}_ns_v2.pkl`
+ 可选缓存 `ml_tgbl-wiki.pkl`。环境变量 `TGB_ROOT` 可覆盖位置。

### 7.3 单任务 / 矩阵 / 推理

```bash
python -m scripts.train --variant spectral --dataset tgbl-wiki --seed 0 \
    --output outputs/tgbl_wiki/tgbl-wiki__spectral__seed000 --gpu 0 [--freeze-host]
python -m prss.experiment.runner --config configs/experiments/tgbl_wiki_matrix.yaml --gpu 0
python -m prss.experiment.summarize outputs/tgbl_wiki
python -m scripts.inference --checkpoint <job>/best.pt --split test --output <json>
```

### 7.4 多进程并行(云端实测教训)

- 用 `nohup bash -c 'cd <repo> && python -m scripts.train ...' > log 2>&1 &` 启动;
  **不要把 `cd` 与第一个 nohup 写在同一 `&&` 链再跟第二个 `&`**——shell 优先级会把
  cd 只留给第一个后台任务;
- 208 核机器必须锁线程(入口已自动处理:OMP/MKL/OPENBLAS=1 + oneDNN 关);
- 全局梯度裁剪**默认关闭**(该宿主梯度合法地达 1e9,全局裁剪会抹零所有参数)。

## 8. 部署战报(2026-08-18,SeetaCloud 克隆实例)

实例:`ssh -p 41165 root@connect.westc.seetacloud.com`(密钥免密;RTX 5090 D 32G、
208 核、数据盘 `/root/autodl-tmp/PRSS2`)。当日修掉 10 个问题(全部本地+云端同步,
详见 `notes/CONTEXT_HANDOFF.md` §9):

| # | 问题 | 修复 |
|:---|:---|:---|
| 1 | 208 核 OpenMP 颠簸(测试 30 分钟/挂点漂移) | 三入口锁定 OMP/MKL/OPENBLAS=1 + 关 oneDNN |
| 2 | PyG 无向子图 trace 成环 → BFS 无限循环 → 内存 97% 拖死实例 | 深度=1 展开 + visited 集合 + 回归测试 |
| 3 | 线程补丁插在 import torch 之前 | 顺序修正 |
| 4 | runner cwd off-by-one(子进程找不到 scripts) | 用 runner 自身位置推仓库根 |
| 5 | 包名 py-tgb;--no-deps 漏依赖 clint 等 | 补装 |
| 6 | TGB 根路径字符串拼接坑(绝对路径被拼坏) | PROJ_DIR 指向仓库根 + 数据放 tgbl_wiki/ 子目录 |
| 7 | MRR 评估 neg 多包一层 list → 排名全乱(0.007) | 对齐官方输入格式 + 回归测试 |
| 8 | root_times 长度 3× 不匹配 → NoneType | repeat_interleave(3) |
| 9 | **全局 grad_clip 抹零**(时间编码梯度 1e9,裁剪系数≈0,损失冻结 1.44) | 默认关裁剪(对齐官方)+ TimeEncoder 输入 ×1e-6 |
| 10 | 验证/测试期缺 eval 模式切换(移植遗漏 tgn.eval()) | evaluate_split 入口 eval/出口恢复;冻结宿主时宿主恒 eval |

**当前结果**:测试套件 69+ 项全绿;smoke(1 epoch×1 seed,修复版)vanilla val mrr
0.0908 / test 0.0602(对齐官方逐字基线 0.074 量级),spectral 0.0180 微弱领先
direct 0.0119 / pca 0.0118 / random 0.0146(1 epoch 非证据);**正式矩阵
30 epoch × pca/vanilla seed 0 正在并行运行**。

## 9. 已知限制与下一步

1. **tgbl-wiki 单数据集先行**:数据集维度已参数化;enron 3054 万边需算力评估。
2. **λ_spec 敏感性未跑**:{0, 0.01, 0.1, 0.5, 1.0},矩阵可加一维。
3. **twitter-TGN 宿主适配器未迁移**:hosts/base.py 抽象已预留;迁移素材 = 官方源码
   (commit d55bbe67 公开)+ 本对话重建的旧适配器逻辑。
4. **PyG 宿主 outside 树深度=1**:论文中需如实声明;更深上下文需改 neighbor loader
   或换宿主。
5. **梯度 1e9 量级**(增长中的 memory 状态所致)未根治:当前靠"关裁剪+时间缩放"
   与官方基线同条件;如需裁剪需先治梯度尺度。
6. **多种子统计口径**:正式矩阵 3 seed 为最小集;论文建议 5 seed 对齐 TGB 惯例。
7. **旧结果不可比**:old/ 节点分类 AUC(0.9001 等)与新协议 mrr/auc_ovm 是不同任务
   不同指标,禁止同表混用。

## 10. 代码阅读顺序指导(逐行审阅用)

> 原则:**从"数据怎么来"到"方法怎么算"到"宿主怎么接"到"实验怎么跑"**。
> 每步给"读什么 + 审阅时问自己的问题"。

### 第 0 步:背景文档(30 分钟)

1. `notes/PRSS_method_spec_v1_2.md` —— 方法规格(重点 §2 等价关系、§4 SVD 最优性、
   §6 outside、§9 谱尾、§12 伪代码、§16 泄漏边界、§33 禁止事项);
2. 本文档 §1/§2/§5 —— 架构与红线。
   审阅问题:**之后的每一行代码,是否能对应到规格的某一节?有没有规格外的主路径?**

### 第 1 步:数据层(15 分钟)

`src/prss/data/tgb_link.py` —— 数据集封装。
审阅问题:TGB 的 root 拼接坑是否被正确规避?mask 切片/负采样/时间统计的口径?
`datasets/tgbl_wiki/` 三个文件的作用?

### 第 2 步:核心数据契约(30 分钟)

`src/prss/config.py`(InterfaceSpec/PRSSConfig)→ `src/prss/state.py`
(QuotientState/RecursiveOccurrence/RecursiveTrace)→ `src/prss/losses.py`。
审阅问题:host_dim 是否只来自宿主?candidate>=host 的校验在哪?
trace 的 root_rows 与 roots 的对齐由谁保证?

### 第 3 步:谱商(方法心脏,1 小时)

`src/prss/spectral.py` 全部。
审阅问题:`accumulate` 的 Gram 是否等于 1/N·ΣBᵀB?`update` 的每一步(eigh→有效秩→
补零空间→Procrustes→Grassmann 步/回退)各对应规格哪一条?`spectral_loss` 的
闸门与分母 detach 为什么存在?`snapshot` 的每个字段谁在消费?

### 第 4 步:可插拔压缩器(40 分钟)

按 `compressors/base.py`(契约)→ `vanilla.py` → `random.py` → `pca.py` →
`direct.py` → `spectral.py` → `__init__.py`(注册表)顺序读。
审阅问题:5 个变体的契约字段(name/statistic/use_response_loss/use_spectral_loss/
update_projection)各是什么?红线(`set_projection_trainable` 抛错)在哪些变体上
成立?pca 的"中心化协方差 + 精确主子空间"与规格的 pca 基线语义是否一致?

### 第 5 步:训练期辅助(1 小时)

`src/prss/reader.py`(B(C) 的线性语义)→ `src/prss/outside.py`(上下文编码、
排除自身、兄弟 detach)→ `src/prss/candidate.py`([vanilla; φ] 的平价结构)→
`src/prss/auxiliary.py`(outside pass、层内/跨层平均、B 不可变快照)。
审阅问题:B(C) 对候选是否严格线性(谱语义的前提)?outside 里 h_v 自己是否绝对
进不了 c_v?层平均策略为什么存在?`detach().clone()` 与 `detach()` 的区别在哪?

### 第 6 步:总装与监控(40 分钟)

`src/prss/core.py`(τ 键、压缩器注入、光谱门控)→ `src/prss/monitoring.py`
(硬不变量、整数精确有限性)。
审阅问题:core 里有没有任何宿主字符串?d==k 接口为何走 vanilla?`set_spectral_
updates_allowed` 门控被谁在何时关/开?监控的哪几项是"失败即中止"的硬闸?

### 第 7 步:宿主层(1.5 小时)

`hosts/base.py`(抽象)→ `hosts/pyg_models/`(vendor 模块,注意 README 中两处
注明补丁)→ `hosts/tgn_pyg.py`(适配器)→ `hosts/tgn_pyg_bridge.py`(链路桥)。
审阅问题:vendor 补丁(scatter_reduce、int64 保持)语义等价吗?`_pack_preagg` 的
打包是否精确覆盖卷积的全部输入?trace 为什么深度=1?visited 防环在哪几处?
4 场景桥的正负/角色/counterpart 是否与规格的泄漏边界一致?

### 第 8 步:训练组件(1 小时)

`training/isolation.py`(三零审计)→ `training/checkpoint.py`(RNG + msg store)→
`training/event_loop.py`(训练/评估/重放时序)。
审阅问题:训练批次的时序(先算分→update_state→insert→detach)与官方是否逐位一致?
评估期 eval/train 切换与冻结宿主的交互是否正确?隔离审计的三个"零"各在哪一步
检查?AUC/AP 的 `_ovm` 口径与主指标 mrr 的关系?

### 第 9 步:入口与实验框架(40 分钟)

`scripts/train.py`(CLI、组装、时间缩放包装、冻结宿主、TB)→ `scripts/inference.py`
(重建+重放+审计)→ `experiment/runner.py`(矩阵、幂等、布尔旗标)→
`experiment/summarize.py`(mean±std 表)→ `configs/experiments/*.yaml`。
审阅问题:时间缩放包装为什么在 build_components 里?冻结宿主时参数组/模式如何
变化?runner 的幂等与失败归档语义?布尔配置值的传递约定?

### 第 10 步:测试套件(1 小时,与上面对照)

按 `test_spectral.py` → `test_core_contract.py` → `test_auxiliary.py` →
`test_compressors.py` → `test_synthetic_tree.py` → `test_tgb_smoke.py` →
`test_experiment.py` 顺序读。
审阅问题:每个测试对应上面哪一步的哪个不变量?有没有规格红线**没有**测试覆盖?
(当前已知缺口:history-mixing 诊断、λ_spec 敏感性、parameter-matched 对照)

### 总时长与产出建议

约 8 小时通读。建议边读边做三件事:① 在规格书上标注每节的实现位置;② 记录
"审阅问题"中答不上来的条目(就是潜在缺陷);③ 读完后对照本文档 §9 的已知限制,
补一份你自己的审阅结论。
