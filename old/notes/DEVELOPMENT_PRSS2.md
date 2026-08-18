# PRSS2 开发日志(理论对照调研版)

> PRSS (Predictive Relation-State Sheaf) — 递归深度模型接口的预测性谱压缩方法
> 宿主:TGN × Wikipedia 动态节点分类;本文对照 `PRSS_method_spec_v1_2.md` 逐项核对当前实现
> 调研日期:2026-08-17

---

## 目录

1. [项目定位与核心理念](#1-项目定位与核心理念)
2. [系统架构总览](#2-系统架构总览)
3. [数据流与实验协议](#3-数据流与实验协议)
4. [核心层 (Core) — 方法的"灵魂"](#4-核心层-core--方法的灵魂)
   - 4.1 [SpectralQuotient 谱商](#41-spectralquotient-谱商--quotient-求解器)
   - 4.2 [Candidate 候选构建](#42-candidate-候选构建--候选加宽车间)
   - 4.3 [Outside 上下文编码器](#43-outside-上下文编码器--训练期未来场景编码)
   - 4.4 [Reader 未来读取器](#44-reader-未来读取器--读取矩阵工厂)
   - 4.5 [TGN 递归适配器](#45-tgn-递归适配器--压缩接口安装点)
   - 4.6 [Auxiliary 辅助损失构建](#46-auxiliary-辅助损失构建--算子库装配)
   - 4.7 [Monitoring 监控系统](#47-monitoring-监控系统--硬不变量审计)
5. [实验层 (Experiments)](#5-实验层-experiments)
6. [测试层 (Tests)](#6-测试层-tests)
7. [诊断层 (Diagnostics) — 动机证据](#7-诊断层-diagnostics--动机证据)
8. [参考代码库](#8-参考代码库)
9. [数据协议与核心张量](#9-数据协议与核心张量)
10. [配置系统](#10-配置系统)
11. [目录结构](#11-目录结构)
12. [理论对照调研总录](#12-理论对照调研总录)
13. [开发状态总览](#13-开发状态总览)

---

## 1. 项目定位与核心理念

### 这不是"训练后压缩",是"接口级预测性商状态"

PRSS 的目标不是把一个训练好的模型变小,而是回答一个问题:**递归模型每个接口只能传 k 维状态时,这 k 维到底应该装什么?**

| 传统压缩 / 传统瓶颈 | 本项目 (PRSS) |
|:---|:---|
| 先训练完整模型,再离线压缩 | **从训练第 0 步起父节点只接收商状态 z=R·h** |
| 用最终任务梯度"间接"决定装什么 | **显式学习"未来 continuation 如何读取历史"(B(C))**,再谱选方向 |
| 用 PCA / 方差选方向 | **用预测算子库的 SVD 选方向**(解析最优的 rank-k 预测子空间) |
| 压缩 = 减小宽度 | **宽度 k 完全由宿主模型决定,压缩的是"候选→接口"的选择** |
| 推理期无额外约束 | **未来信息仅训练期监督 reader,推理期被硬审计禁止** |

### 三大设计原则

1. **预测等价性**:`H~H' ⟺ P(Y|C[H])=P(Y|C[H']) ∀C` 定义"正确的商",且对树构造器天然同余(可层层压缩)
2. **谱最优性**:固定 k 下最小化预测算子残差 ⟺ 最大化 `tr(RGRᵀ)`,解析最优解 = Gram 的 top-k 特征空间(SVD 不是启发式)
3. **因果隔离**:reader/outside/Gram/SVD 全部仅训练期存在;验证/测试期由硬审计保证零变化

---

## 2. 系统架构总览

```
┌──────────────────────────────────────────────────────────────┐
│    experiments/train_supervised_prss_switch.py (实验层)       │
│    "官方 TGN 母本 + 单一 vanilla|prss 开关 + 干净 val/test"    │
├──────────────────────────────────────────────────────────────┤
│                      prss/ 核心层                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐       │
│  │ candidate    │ │ outside      │ │ reader           │       │
│  │ (候选加宽h)  │ │(训练期上下文c)│ │(读取矩阵B(C))     │       │
│  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘       │
│         │                │                  │                 │
│  ┌──────┴────────────────┴──────────────────┴─────────┐       │
│  │ auxiliary.py — 辅助损失(L_resp/L_spec/监控)+算子库  │       │
│  └──────────────────────────┬──────────────────────────┘       │
│  ┌──────────────────────────┴──────────────────────────┐       │
│  │ spectral.py — 谱商 R (Gram EMA→eigh→Procrustes→部署) │       │
│  └──────────────────────────┬──────────────────────────┘       │
│  ┌──────────────────────────┴──────────────────────────┐       │
│  │ tgn_adapter.py — 递归适配 (z=R·h,父只接收 z)         │       │
│  └─────────────────────────────────────────────────────┘       │
│                              │                                │
│  ┌───────────────────────────┴────────────────────────────┐   │
│  │ monitoring.py — 硬不变量审计(有限/对称/正交/谱隔离)     │   │
│  └─────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│  official_tgn/ — byte-for-byte 上游锚点 + SHA256 冻结清单      │
└──────────────────────────────────────────────────────────────┘
```

### 逻辑分层

| 层级 | 角色 | 比喻 | 职责边界 |
|:---|:---|:---|:---|
| **实验层** | 官方母本训练器 | 总装车间 | 数据、宿主、解码器、训练循环、评估协议 |
| **prss 核心层** | 方法本体 | 灵魂 | 候选、上下文、读取器、谱商、适配、监控 |
| **official_tgn/** | 上游锚点 | 公证处 | 证明任务与权重可复现,自身不被修改 |
| **tgn/ (顶层)** | 动机诊断 | 勘探队 | 证明"冻结状态丢信息且低秩"两个前置命题 |
| **TGB/PINT/NSD** | 参考库 | 资料室 | 未来评测框架与理论类比对象 |

---

## 3. 数据流与实验协议

### 数据预处理链(已完成,产物在 `processed_tgn_data/`)

```
wikipedia.csv (157,474 交互;前4列 user/item/time/state_label + 172维特征)
    │
    ├── 官方 bipartite reindex:0=padding, 用户1–8227, 页面8228–9227
    ├── 交互编号 1–157474,时间单调性校验
    │
    └── processed_tgn_data/
          ├── ml_wikipedia.csv / ml_wikipedia.npy / ml_wikipedia_node.npy
```

### 实验协议(官方母本 matched 训练器)

```
预训练:官方自监督 future link prediction(诊断阶段产出 checkpoint,单 seed)
    │
    ▼
下游节点分类(冻结宿主,PRSS 在此介入)
    │
    ├── train/val/test = 110,232 / 23,621 / 23,621(时间序 70/15/15)
    ├── 每 epoch 重置 memory,按时间顺序重放 train → val
    ├── 训练循环:完整官方 compute_temporal_embeddings(src,dst,dst)
    │       → 任务 BCE → (PRSS) outside pass → B(C) → Gram → 周期 SVD
    ├── val 早停(selection metric = AUC,patience=3)
    ├── best checkpoint → 零 memory → 重放 train+val → held-out test
    └── 测试期谱隔离审计(前后 Gram/SVD 计数与 R 矩阵必须零变化)
```

关键点:
- `--mode vanilla|prss` 是唯一科学开关,两侧协议完全一致
- 官方锚点(`official_reference`)保留上游协议怪癖(`use_validation=False` → val=test),仅作 provenance 锚点,不参与方法对比
- 当前仅 seed 0 单次;多 seed 是规格 §30 验收的前提,尚未执行

---

## 4. 核心层 (Core) — 方法的"灵魂"

### 4.1 SpectralQuotient 谱商 — quotient 求解器

> 代码路径:`PRSS-method/prss/spectral.py`
> **状态:已实现;含文档化的信任域工程近似**

**关键特性:**

- `R` 注册为 buffer(`k×d`,行正交),**永远不是梯度参数**;初始化 `R=[I,0]`,保证第 0 步 PRSS 前向数值上等于官方 TGN
- `accumulate()`:`G = (1/N)Σ BᵀB` 的 EMA(ρ=0.05),首次直接赋值;对流式算子库,从不显式构造巨型 B 栈
- `update()`:对称化 + ridge(1e-8)→ `torch.linalg.eigh`(double)→ 有效秩判定(rtol=1e-5)→ top-k 特征向量 → 旧商补全 null space → **正交 Procrustes 对齐** → 阻尼信任域部署
- `spectral_loss()`:归一化谱尾 `‖B(I−P_R)‖²/(‖B‖²+ε)`,分母 **detach + 下界 1e-4**;仅当 R 已离开 `[I,0]` 初始化后才激活
- `snapshot()`:完整谱诊断(energy@k/4、k/2、k、full;tail@k;projector 距离;主角度;live Gram 覆盖率;特征值列表)

**理论偏差提示(详见第 12 节):**

1. 规格 §12 伪代码每次谱更新**直接部署** SVD 目标;实现默认 `spectral_step_size=0.25`,是**有回溯的 Grassmann 上升步**,部署后的 R 一般不是当前 Gram 的解析最优解(`spectral_step_size=1` 才精确,仅测试用)。文档化于 METHOD.md §3 / HOTFIX_V4/V5。
2. Gram 初始化:规格伪代码 `G_ema = eps·I`,实现为零初始化 + 首次直接赋值(避免早期谱被 eps·I 污染,数值小偏差)。
3. ridge eps 实现 1e-8,规格 §23 建议 1e-5(微小差异)。
4. Gram 秩 < k 时的 null-space 补全策略(取旧商的补空间分量)规格未定义,实现注释声称"仍是精确优化器"(任意补全都最优,只是选一个确定性的)。

### 4.2 Candidate 候选构建 — 候选加宽车间

> 代码路径:`PRSS-method/prss/candidate.py`
> **状态:已实现**

**关键特性:**

- `h_l = [原版 TGN 聚合输出; φ(精确聚合输入)]`,d=256 > k=172
- φ 是 GELU+LayerNorm 小 MLP(172+… 输入 → 128 → 84 残差),符合规格 §5.1"不要上大网络"
- 输入为**宿主聚合的全部精确输入**:source lower quotient、source time、邻居 lower quotients、边时间、边特征、padding mask——下层状态已是商状态,**无法绕过更早的递归压缩**

**理论偏差提示:** 规格 §5.1 建议 lift 形状 `raw→128→128` 且"residual if dimensions allow";实现把原版输出拼在前面、φ 只补残差,是等价的设计变体,`[I,0]` 初始化依赖此结构。

### 4.3 Outside 上下文编码器 — 训练期未来场景编码

> 代码路径:`PRSS-method/prss/outside.py`
> **状态:已实现**

**关键特性:**

- `Occurrence`/`Trace`:训练期计算树的数据结构(仅被 trace 的根行)
- `OutsideContextEncoder`:
  - `root_context` = 标准化 log 查询时间 + 层嵌入(根 continuation = 共同分类器 + 合法查询元数据)
  - `child_context` = 父上下文 + 父局部元数据(边/时间/mask)+ 关系嵌入 + `log1p(Δt)` + **兄弟候选 detach 平均摘要**
  - **严格排除当前节点自己的 h**(规格 §6 红线)
- 一次 bottom-up(主前向)+ 一次 top-down outside pass,线性于树大小

**理论偏差提示:** 基本忠实于规格 §6。差异:兄弟聚合用 mean(规格未规定具体聚合算子);父局部元数据包含边特征(规格 §16 允许"当前 edge/relation type")。

### 4.4 Reader 未来读取器 — 读取矩阵工厂

> 代码路径:`PRSS-method/prss/reader.py`
> **状态:已实现**

**关键特性:**

- `ConditionalMatrixReader`:context → trunk(GELU+LN)→ `matrix_head` 输出 `B∈R^{1×d}` + `bias_head` 输出 b;二分类任务 p=1
- `logits = b + B·h`(**对候选严格线性**——这是 SVD 语义成立的前提)
- `UnrestrictedReader`:MLP(h,C) 对照,**仅监控**:输入 detached、独立 optimizer、不进 Gram、不影响主表示

**理论偏差提示:** 规格 §7 推荐"共享 context encoder + 类型条件矩阵头";实现按层各配一套 reader(参数更多,但 τ=layer 时仍符合 `B=M_{η,τ}(c_v)` 形式)。unrestricted reader 的 detach 是监控卫生决策,规格 §21.1 只要求对照。

### 4.5 TGN 递归适配器 — 压缩接口安装点

> 代码路径:`PRSS-method/prss/tgn_adapter.py`
> **状态:已实现**

**关键特性:**

- 包装官方 `GraphEmbedding`,**不改宿主聚合逻辑**;每个递归返回前插入 `make_candidate → project`,父节点只接收 `z=R·h`
- 递归语义精确镜像官方(邻居递归沿用原始 query timestamp;padding 节点按官方语义)
- `set_trace_source_rows` / `clear_trace`:trace 仅训练期、仅选中行;**trace 永不改变主前向**(测试覆盖)
- 验证/测试:`clear_trace()` 后无任何谱状态触碰(测试覆盖 + 运行时审计)

**理论偏差提示:** 忠实于规格 §14。注意:τ 的实现定义是 **layer**(每层一个 R),而规格 §15 建议 τ=(relation type, message/update role)——Wikipedia 单关系下按层共享是 V1 合理简化,论文需明确写出 τ=layer 的定义。

### 4.6 Auxiliary 辅助损失构建 — 算子库装配

> 代码路径:`PRSS-method/prss/auxiliary.py`
> **状态:已实现;存在未文档化的采样偏差(见下)**

**关键特性:**

- `build_auxiliary`:对每个被 trace 的根做 BFS outside pass;每 occurrence 输出 B、b → BCE(logits, **根标签**)+ 谱尾 + 监控对照
- 损失平均策略:**层内先平均、再跨层平均**——防止低层 occurrence 数量(≈10×)主导辅助目标(文档化于 METHOD.md §6)
- Gram 快照用 `B.detach().clone()` 不可变复制(HOTFIX_V5 修复存储复用问题)

**理论偏差提示(重要,需向持有者确认):**

1. **根行子采样**:每批仅 `trace_roots=8` 行进入 outside pass(训练器 `select_trace_rows`);规格 §12 伪代码默认全批。Gram 因此是"被 trace occurrence"上的估计量。
2. **正样本优先采样**:`select_trace_rows` 先取 batch 内全部正根(最多 8),不足再确定性补负根。给定极稀疏标签(≈0.14 正/batch),实际效果是 reader 训练分布**几乎全为负根、且正根被过采样**——读者监督分布与流分布不同,规格未定义此策略,无文档。
3. **每 occurrence 的标签 = 根标签**:中间层 occurrence 的 B(C) 被同一个根 Y 监督(规格 §5.2 的语义),论文需明确"整棵树共享根任务目标"。

### 4.7 Monitoring 监控系统 — 硬不变量审计

> 代码路径:`PRSS-method/prss/monitoring.py`
> **状态:已实现(alert 文件 resume 口径有小瑕疵)**

**关键特性:**

- `MonitorWriter`:step/epoch metrics(jsonl)、alerts、monitor_summary、逐 epoch 投影快照(R/G/snapshot)
- 硬闸门(默认 fail_on_error):loss 非有限、候选/reader 矩阵非有限、Gram 失对称(1e-6)、R 失行正交(5e-4)、验证/测试改变谱状态 → 直接 RuntimeError
- 有限性判定用**整数计数**而非 CUDA float 均值(HOTFIX_V7 修复假阳性)
- 梯度 L2(各模块分开)、B 范数统计、候选坐标方差、structured-vs-unrestricted NLL gap(阈值 0.25 告警)

**已知瑕疵:** resume 时 `reset_files=False`,历史 alert 文件不清理——最终 `prss_matched` 的 monitor_summary 显示 `error_alerts=2`,实为 v7 修复前的遗留假阳性记录,报告口径需清理。

---

## 5. 实验层 (Experiments)

> 代码路径:`PRSS-method/experiments/`

| 文件 | 职责 | 状态 |
|:---|:---|:---|
| `train_supervised_prss_switch.py` | 官方母本训练器,`--mode vanilla\|prss` 唯一科学开关;完整官方 `compute_temporal_embeddings(src,dst,dst)` 调用;干净 70/15/15 val/test;AP/NLL 监控;rolling checkpoint 断点续跑(含 RNG/memory 状态恢复);谱隔离审计 | 已实现 |
| `train_node_classification_faithful.py` | 上者 1 行入口别名 | 已实现 |
| `compare_nodecls.py` | 三 summary 汇总 + PRSS−vanilla 差值 | 已实现 |
| `summarize_official_reference.py` | 官方脚本工作目录 → summary.json | 已实现 |

编排脚本(`PRSS-method/` 根):`run_tonight_2gpu.sh`(GPU0 vanilla→官方锚点,GPU1 PRSS)、`run_prss_only_gpu1.sh`(单跑 PRSS,支持 `--resume-from`)、`run_autodl.sh`(前者的转发)。均为 AutoDL Linux bash,Windows 需 Git Bash/WSL 或手工命令(见 PROJECT_INDEX.md §10)。

**当前结果**(seed 0, L=2, 冻结宿主):official_reference 0.8933 / vanilla_matched 0.8855 / **prss_matched 0.9001**(test AUC)。prss best_epoch=0(val 峰在 epoch 0),需多 seed 判断是否规律。

---

## 6. 测试层 (Tests)

> 代码路径:`PRSS-method/tests/`

| 文件 | 覆盖内容 | 状态 |
|:---|:---|:---|
| `test_prss_runtime.py` | 算子库 SVD 等价性(显式堆叠右奇异子空间 = 流式 Gram 特征空间);候选消费全部官方聚合输入;`[I,0]` 初始化与官方递归前向逐位一致;full TGN 调用 + memory 语义一致;trace 只记录选中行;无 trace 推理不碰谱状态;母本 vanilla 一步与上游逐位一致;无 PCA 等替代还原路径守卫;层 0 非 reader 接口;谱损失闸门与近零 B 梯度有限性;阻尼部署能量单调、被拒解不激活谱损失 | 已实现,覆盖良好 |
| `test_official_parity.py` | 上游 SHA256 清单(commit d55bbe67)逐文件核对 | 已实现 |
| `test_monitor_finiteness.py` | 监控不变量 | 已实现 |
| `test_resume_rng.py` | 断点续跑 RNG 一致性 | 已实现 |

**缺失:** 规格 §29 Phase 2 的合成树识别测试(`D,E→A→Y` 已知子空间恢复)只在**旧实现**(`PRSS-method_old_021746/experiments/synthetic_tree.py`,主角度 0.00256 rad)中,未迁移到当前包。

---

## 7. 诊断层 (Diagnostics) — 动机证据

> 代码路径:`tgn/diagnostics/`(顶层 `tgn/`,非 PRSS-method)

验证两个前置命题:(1) 冻结 TGN 的 (h,C) 遗漏仍影响未来的历史信息;(2) 该信息低秩。

| 文件 | 职责 | 状态 |
|:---|:---|:---|
| `common.py` | 冻结重建、memory 清零重放、浅 probe、paired bootstrap | 已实现 |
| `history.py` | X 描述子(结构 12 统计 + 边特征 3×172 + 层 0 状态统计),精确镜像官方递归语义 | 已实现 |
| `extract_cuts.py` | 冻结权重按时间重放导出 h/C/X/Y(事件写入 memory 前采样) | 已实现 |
| `conditional_residual.py` | M0 / MK / MK-shuffle ΔNLL + 95% CI | 已实现 |
| `collision_analysis.py` | (h,C) 近碰撞与未来分歧曲线 | 已实现 |
| `predictive_rank.py` | cross-fitted predictive SVD + held-out RRR | 已实现 |
| `summarize_baselines.py` | L×memory×seed 基线汇总 | 已实现 |
| `verify_real_data.py` | 数据 fail-fast + observer 只读性(逐位一致) | 已实现 |

注意:本层的 collision 诊断是**冻结 TGN 的动机证据**,不是规格 §21.6 要求的 PRSS 机制版 history-mixing 诊断(后者未实现,见第 12 节 C 类)。

---

## 8. 参考代码库

| 目录 | 角色 | 状态 |
|:---|:---|:---|
| `TGB-main/` | Temporal Graph Benchmark(22 数据集/负采样/mrr·ndcg 评估器) | 未引用;未来标准化评测候选 |
| `PINT-main/` | PINT(NeurIPS 2022 可证明表达力时序 GNN) | 未引用;备用基线 |
| `neural-sheaf-diffusion-master/` | NSD(NeurIPS 2022) | 仅结构类比对象;规格 §18 禁止声称 G 是 sheaf Laplacian |
| `PRSS-method_old_021746/` | 旧实现(宿主无关包 + 7 消融 + 合成测试 + λ 敏感性脚本) | 已归档;其消融基于旧训练器,结果不可直接迁移 |

---

## 9. 数据协议与核心张量

### Occurrence / Trace(训练期计算树)

```
Occurrence
├── oid: int               # 全局唯一 occurrence 编号
├── layer: int             # 递归层
├── candidate: Tensor      # h_l ∈ R^d(候选状态,pre-quotient)
├── local: Tensor          # 父局部元数据(边/时间/mask)
├── children / relations / deltas   # 子 occurrence、关系(0=self,1=nbr)、时间差
```

```
Trace
├── occurrences: Dict[oid, Occurrence]   # 被 trace 根行整棵树
├── roots / root_rows                    # 根 occurrence 与 batch 行号
```

### 核心数学对象

| 符号 | 定义 | 实现位置 |
|:---|:---|:---|
| `h_l` | 候选状态 d=256 维 | `candidate.py` |
| `z_l` | 部署状态 `R·h`,k=172 维(宿主宽度) | `spectral.py project()` |
| `c_v` | 训练期 outside 上下文(64 维) | `outside.py` |
| `B(C)` | 未来读取矩阵 1×256 | `reader.py matrix_head` |
| `G` | 预测 Gram `E[BᵀB]`(EMA) | `spectral.py accumulate()` |
| `R` | 商矩阵 172×256,行正交 buffer | `spectral.py` |
| `L_resp` | `BCE(b+B·h, Y)` | `auxiliary.py` |
| `L_spec` | `‖B(I−P_R)‖²/(‖B‖²+ε)`(分母 detach 下界) | `spectral.py spectral_loss()` |

---

## 10. 配置系统

### 当前实验 CLI 参数(seed_0_l2 实际运行)

| 分组 | 参数 | 默认值 |
|:---|:---|:---|
| 宿主 | `--n-layer` / `--n-degree` / `--bs` / `--use-memory` / `--n-epoch` / `--patience` | 2 / 10 / 100 / True / 10 / 3 |
| 候选 | `--candidate-dim` / `--candidate-hidden` | 256 / 128 |
| 上下文/读取 | `--context-dim` / `--reader-hidden` | 64 / 128 |
| 损失 | `--lambda-response` / `--lambda-spectral` | 1.0 / 0.1 |
| 谱 | `--gram-ema` / `--spectral-warmup` / `--spectral-interval` / `--spectral-step-size` | 0.05 / 200 / 200 / 0.25 |
| 采样 | `--trace-roots` | 8 |
| 监控 | `--monitor-every` / `--grad-clip` / `--checkpoint-every` | 50 / 5.0 / 50 |
| 实验 | `--seed` / `--finetune-host` / `--selection-metric` | 0 / False / auc |

与规格 §23 对照:λ_resp/λ_spec/Gram EMA/interval/outside dim 一致;候选宽度由 CLI 决定(规格"auto_or_configured");**λ_spec ∈ {0,0.01,0.1,0.5,1.0} 敏感性未执行**。

---

## 11. 目录结构

```
PRSS2/
├── PRSS_method_spec_v1_2.md          # 方法总规格(理论权威)
├── PROJECT_INDEX.md                  # 研究索引(供在线 AI 协作)
├── notes/
│   ├── DEVELOPMENT_temp.md           # 模板来源(Shiraha_OS 项目)
│   └── DEVELOPMENT_PRSS2.md          # 本文件
├── wikipedia.csv / processed_tgn_data/
├── PRSS-method/                      # ★当前实现
│   ├── prss/
│   │   ├── core.py / candidate.py / outside.py / reader.py
│   │   ├── spectral.py / auxiliary.py / tgn_adapter.py / monitoring.py
│   ├── experiments/
│   │   └── train_supervised_prss_switch.py(主入口)
│   ├── official_tgn/                 # byte-for-byte 锚点 + SHA256
│   ├── tests/
│   ├── outputs/node_classification/… # 实验结果
│   └── METHOD.md / PRSS_METHOD_SPEC.md / OFFICIAL_PARITY.md / MONITORING.md / HOTFIX_V4·V5·V7.md
├── PRSS-method_old_021746/           # 旧实现(消融/合成测试/λ敏感性)
├── tgn/                              # 官方 TGN + 动机诊断插桩
├── TGB-main/ PINT-main/ neural-sheaf-diffusion-master/
└── papers/TGN_2006.10637.pdf         # 已下载的 TGN 论文
```

---

## 12. 理论对照调研总录

> 原仓库持有者判断:"PRSS 的实现还不太严谨,还有些地方是工程近似"。以下按严重度分级,
> 每条给出 **规格出处 → 实现现状 → 影响与建议**。

### A 类:已有文档的工程近似(可接受,论文中需声明)

| # | 规格出处 | 实现现状 | 影响与建议 |
|:---|:---|:---|:---|
| A1 | §12 伪代码:谱更新**直接部署** SVD 目标 | 默认阻尼信任域步 0.25(回溯保证能量上升);`spectral_step_size=1` 才精确 | 运行时 R 通常非当前 Gram 解析最优;论文若声称"解析最优"须限定于 SVD 目标本身与 =1 情形。已文档化(HOTFIX_V4/V5、METHOD.md §3) |
| A2 | §9/§12:L_spec 恒参与 | 仅当 R 已离开 `[I,0]` 后才激活(闸门) | 防止对任意初始化正则化;合理护栏,已文档化(HOTFIX_V5) |
| A3 | §9 公式:`‖B‖²+ε` 分母可导 | 分母 detach + 下界 1e-4 | 仅改变 B→0 退化点的优化条件,不进 Gram/SVD;代码注释已声明 |
| A4 | §12 伪代码:`G_ema = eps·I` 初始化 | 零初始化 + 首次直接赋值 | 避免早期谱被 eps·I 污染;数值小偏差 |
| A5 | §23:`ridge_eps: 1.0e-5` | 实现 1e-8 | 微小差异,建议统一 |
| A6 | §15:τ=(relation type, message/update role) | 实现 τ=layer | Wikipedia 单关系下按层共享属 V1 合理简化;论文需明确定义 |
| A7 | §0.1:d=k 时"只做坐标重组" | 层 0 完全排除(无 reader/Gram/SVD) | 实现决策,已文档化(METHOD.md §1);合规 |
| A8 | §7:共享 context encoder + 类型条件矩阵头 | 每层独立 reader 参数 | 参数更多但仍符合 `B=M_{η,τ}`;属实现变体 |
| A9 | §6:Agg_{siblings} 未规定算子 | 实现为 mean | 合理 |
| A10 | §21.1:unrestricted 对照 | 输入 detach + 独立优化器 | 监控卫生,保证不污染主表示 |
| A11 | 官方协议 use_validation=False(val=test) | 锚点保留怪癖,matched 协议修复 | 已文档化(OFFICIAL_PARITY.md) |

### B 类:未文档化的偏差(需向持有者确认)

| # | 规格出处 | 实现现状 | 影响与建议 |
|:---|:---|:---|:---|
| B1 | §12 伪代码:全批计算树 | 每批仅 trace ≤8 根行;`select_trace_rows` **正样本优先、负根确定性补齐** | Gram 是子样本估计量;reader 监督分布几乎全为负根且正根过采样——与流分布不同、规格未定义、无文档。**需确认这是否算有偏监督**;建议:文档化策略、消融 trace_roots、报告正/负根响应损失分开 |
| B2 | §5.2:Y 为任务标签 | 每 occurrence 的响应标签 = 根标签(整树共享同一监督) | 语义上合规,但论文需写明"根任务 Y 监督整棵树" |
| B3 | 监控契约 | resume 不清理 alerts.jsonl,最终 summary 含 v7 前 2 条遗留假阳性(error_alerts=2 但 complete) | 报告口径需清理(记录"已修复的遗留告警"与"本次运行告警"分开) |
| B4 | §30:多 seed 验收 | 仅 seed 0 单次;prss best_epoch=0 | 无法判断 best_epoch=0 是否规律;须多 seed |

### C 类:规格要求但未实现/未执行(当前最大短板)

| # | 规格出处 | 缺失内容 | 建议 |
|:---|:---|:---|:---|
| C1 | §21.5 / §30.3 | **基线组**:random orthogonal、PCA、直学 Linear/MLP 投影、ridge/linear reader+SVD、Jacobian-SVD(旧实现有前四种消融,但基于旧训练器,不可直接引用) | 在官方母本训练器中加 `--ablation` 开关 |
| C2 | §30.2 | energy@k 与 random/PCA 的 predictive energy coverage 对比 | 加入基线组后一并计算 |
| C3 | §30.5 | parameter-matched baseline(PRSS 主路径含 φ 参数,vanilla 无) | 加同参数对照(如 vanilla+等宽 MLP) |
| C4 | §21.6 | **history-mixing 机制诊断**(PRSS 版)未实现(tgn/ 的 collision 是冻结 TGN 动机版) | 实现:局部预测相近、不同 sibling/context 下未来不同的 pairs,验证 PRSS 减少混叠 |
| C5 | §22 | A–H 八项消融在当前训练器无开关 | 迁移旧实现 `ablations.py` 到官方母本 |
| C6 | §23 | λ_spec ∈ {0,0.01,0.1,0.5,1.0} 敏感性未执行(旧实现有脚本) | 直接跑 |
| C7 | §20 | Jacobian-SVD fallback 未实现(规格允许诊断失败后再上) | 先完成 §21.1 诊断评估,再决定 |
| C8 | §29 Phase 4 | 多数据集(UCI/Enron)未做 | 数据就绪后跑 |
| C9 | §27 | 四项理论空白:有限样本一致性、lift identifiability、support gap、rank-k→risk 上界 | 理论阶段推进,非编码 blocker |
| C10 | §29 Phase 2 | 合成树识别测试未迁移到当前包(旧实现已通过) | 迁移 `synthetic_tree.py` |

### D 类:规格 §33 禁止事项合规核查(已逐项核对,无违规)

- ✅ 无"先训练 vanilla 再离线压缩";✅ 无 root-to-leaf 后处理;✅ 无 PCA 主路径(`test_no_alternate_reduction_runtime_import` 守卫);✅ 无 IB/gate;✅ 推理期无未来 context/label;✅ 无每条样本动态 basis;✅ 未替换宿主聚合器;✅ SVD 非一次性初始化(R 持续更新);✅ test 不更新 G/R(审计强制);✅ R 是 buffer 非参数。

---

## 13. 开发状态总览

### 已实现(可用)

| 模块 | 路径 | 说明 |
|:---|:---|:---|
| SpectralQuotient 谱商 | `PRSS-method/prss/spectral.py` | Gram EMA/eigh/Procrustes/信任域/null 补全/完整快照;R 非参数 |
| 候选构建 | `PRSS-method/prss/candidate.py` | 精确消费官方聚合输入;`[I,0]` 初始化逐位等于官方前向 |
| Outside 编码器 | `PRSS-method/prss/outside.py` | 根/子上下文递归,严格排除自身 h,兄弟 detach |
| 读取器 | `PRSS-method/prss/reader.py` | B(C) 线性读取 + 监控用 unrestricted 对照 |
| 递归适配器 | `PRSS-method/prss/tgn_adapter.py` | 压缩接口安装,父只见 z;trace 不改变前向 |
| 辅助损失 | `PRSS-method/prss/auxiliary.py` | outside pass + 层内/跨层平均 + 不可变 Gram 快照 |
| 监控审计 | `PRSS-method/prss/monitoring.py` | 硬不变量 + 谱隔离审计 + 梯度/分布诊断 |
| 母本训练器 | `PRSS-method/experiments/train_supervised_prss_switch.py` | vanilla\|prss 开关、干净 val/test、rolling checkpoint、RNG 恢复 |
| 官方锚点 | `PRSS-method/official_tgn/` | byte-for-byte + SHA256 冻结 |
| 测试套件 | `PRSS-method/tests/` | SVD 等价性/初始化一致性/隔离性/监控/续跑 |
| 动机诊断管线 | `tgn/diagnostics/` | 四个主诊断端到端可用 |
| 编排脚本 | `PRSS-method/run_*.sh` | AutoDL 双卡编排 + 断点续跑 |
| 旧实现资产 | `PRSS-method_old_021746/` | 合成树测试、7 消融、λ 敏感性脚本(基于旧训练器) |

### 已有代码但逻辑不完善(需处理)

| 模块 | 路径 | 问题 |
|:---|:---|:---|
| 谱部署 | `prss/spectral.py update()` | 默认阻尼步 0.25,非规格伪代码的直接部署(A1);部署后 R 非当前 Gram 解析最优 |
| 谱损失 | `prss/spectral.py spectral_loss()` | 分母 detach+下界、激活闸门与规格公式不完全一致(A2/A3) |
| 训练器采样 | `experiments/train_supervised_prss_switch.py select_trace_rows()` | 每批 ≤8 根、正样本优先,读者监督分布与流分布不同,无文档(B1) |
| 监控报告 | `prss/monitoring.py` + resume 流程 | alerts.jsonl 不随 resume 清理,遗留告警混入最终 summary(B3) |
| 官方锚点协议 | `official_tgn/train_supervised.py` | test-peeking 怪癖保留(有意为之,仅锚点用) |
| 单次运行 | `outputs/.../seed_0_l2` | 仅 seed 0;prss best_epoch=0 待多 seed 确认(B4) |

### 占位/未实现(规格要求)

| 模块 | 规格出处 | 设计意图 |
|:---|:---|:---|
| 基线组(random/PCA/direct/ridge/Jacobian) | §21.5/§30.3 | 证明 SVD 机制有额外贡献的对照 |
| PRSS 版 history-mixing 诊断 | §21.6 | 验证"减少错误混叠"的机制核心证据 |
| 官方母本消融开关(A–H) | §22 | 证明提升来自谱机制而非 lift 容量 |
| λ_spec 敏感性 | §23 | 旧实现有脚本,未在新训练器执行 |
| Jacobian-SVD fallback | §20 | 诊断失败后的备用路径 |
| 合成树测试迁移 | §29 Phase 2 | 旧实现通过,未迁移到当前包 |
| 多数据集 Phase 4 | §29 Phase 4 | UCI/Enron 泛化 |
| 多 seed + 统计口径 | §30 | 验收标准的统计基础 |
| 理论四项证明 | §27 | 有限样本一致性/identifiability/support gap/风险上界 |

---

## 调研结论(一句话)

当前实现**忠实覆盖了规格的核心数学对象与因果红线**(§33 禁止事项零违规),不严谨处集中在三类:**谱部署的阻尼近似(A1)、读者监督的采样偏差(B1)、规格要求的对照实验缺失(C 类)**——其中 C 类是论文主张成立与否的命门,应优先补齐基线组与 history-mixing 诊断。
