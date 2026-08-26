# PRSS → RPBE 重构计划（分支 `develop_RPBE`）

## Context

项目原方法 **PRSS**（谱商 SVD 压缩：候选加宽 d=256 → 预测 Gram `G=E[BᵀB]` 的 top-k 特征分解得投影矩阵 R → 父节点只收 `z=R·h`）已被**完全废弃**。新方法 **RPBE** 的架构来自 `new_method_v2.0.md` + `method_delta.md`，其中 delta 的两条纠正仍然有效：①无教师（Y 来自数据真实 continuation）②宿主与 Γ_θ 联合训练。**损失定义以用户 2026-08-26 最新算法稿（Ky Fan 谱分数）为唯一权威**——此前文档中的 **U–B 增量 balancing、path-gain 路径增益、tree_energy 树 DP 与所谓 UB 链条全部作废删除**。

方法核心：

```
z_v = Γ_θ(o_v, {z_vi, ξ_vi})          ← 递归压缩，宿主决定硬预算 r_τ
p_v = ψ(c_v, y_v)                      ← 固定联合特征 φ_C(c)⊗φ_Y(y)，白化到 E[ppᵀ]=I_d
Ĵ_τ = tr[(Σ_ZZ+ε_Z I)⁻¹ Σ_ZP (Σ_PP+ε_P I)⁻¹ Σ_PZ]   ← per-τ Ky Fan 分数（Cholesky，无 SVD）
L_KF = −Σ_τ α_τ·Ĵ_τ                    ← 组件内唯一 loss：最大化谱分数
```

理论链（严格版）：ISK 条件等价 `Y⊥H|(Z,C)` → 联合 characteristic 测试 `Φ(c,y)` 把"历史等价"变成"条件未来嵌入 μ(h) 相同" → 解析消去线性读出 A 得 profiled-MSE 恒等式 → **Ky Fan 定理**：在全部 r 维白化历史坐标上 `sup_J = Σ_{j≤r} λ_j(M)`。压缩预算来自接口维数 r_τ（宿主决定），Ky Fan 分数负责决定这 r 维保留哪些预测方向。有限特征/有限样本/on-policy 覆盖下，这是可量化的谱近似，不是逐点 ISK 等价——报告必须如实按此口径写。

预期产出：在官方 TGN / JODIE wikipedia 上，用**不开 RPBE 的纯宿主**作基线对比 RPBE 的任务指标（现有基线：JODIE wikipedia 纯宿主 test AUC **0.8871**）。

## 已拍板的决策（不要偏离）

| # | 决策 | 内容 |
|---|---|---|
| 1 | 分支 | 从 `main`(2027bb0) 新建 `develop_RPBE`；main 保持不动 |
| 2 | 宿主线 | **每个宿主单独一条 loop**。本分支**只做官方 twitter-TGN/JODIE**（L 层递归树，`jodie_loop.py` 是其专属 loop）；**PyG-TGN/TGB 整条线在 develop_RPBE 上删除**，将来作为独立的第二条 loop 再建（见 §1.5） |
| 3 | "唯一 loss" 的含义 | 指 **RPBE 组件内部**只有一个 loss（Ky Fan 谱分数）；**宿主 task loss 照常存在**。总 loss = `L_task − λ_kf·Ĵ`（等价正形式 `L_task + λ_kf·Σ α_τ(d_τ − Ĵ_τ)`），一次前向一次反向，单 optimizer 同时含宿主 ω 与压缩器 θ |
| 4 | 宿主初始化 | 阶段一自监督预训练 → 阶段二节点分类联合微调。阶段一由新增的 `--stage1-rpbe` 开关控制：**开** = 宿主与 Γ_θ 从随机初始化起就联合预训练（RPBE 从第 0 步参与）；**关** = 只预训练 TGN 宿主（Γ_θ 在阶段二才初始化）。阶段二一律联合微调。历史产物 `outputs/jodie_wikipedia/vanilla_seed0/best.pt` 的 `model.tgn` 内含一份阶段一权重，可作过渡期对照，但正式流程走自己新增的阶段一 loop |
| 5 | 损失形式 | **per-τ Ky Fan 谱分数最大化**（决策 #3 的组件内唯一 loss）。`Ĵ_τ = tr[(Σ_ZZ+ε_ZI)⁻¹Σ_ZP(Σ_PP+ε_PI)⁻¹Σ_PZ]`，Cholesky/线性方程求解，**训练路径无 SVD**。**path-gain / JVP / 树 DP / U-B 增量缺失 全部废除，不实现** |

## 一、删除清单

### 1.1 整文件删除

| 路径 | 行数 | 删除依据 |
|---|---|---|
| `src/prss/spectral.py` | 404 | SVD 商体系整体废弃。new_method §九禁止清单含"后处理 SVD"，§十"奇异值谱只作为诊断指标，不进入主 loss" |
| `src/prss/compressors/`（整包 7 文件） | 335 | `Compressor`/`InterfaceData` 是 R 矩阵抽象，被 Γ_θ 神经网络取代（new_method §四、delta §三）。五变体 vanilla/random/pca/direct/spectral 全消失 |
| `src/prss/reader.py` | 57 | 可训练 reader B(C) 被**固定** χ_h/φ_h/TensorSketch 取代。delta §7 明令"禁止使用可训练 encoder 把 C,Y 编成 p" |
| `src/prss/outside.py` | 99 | OutsideContextEncoder 被固定 context map 取代；C 由数据定义（delta §2 判定标准） |
| `src/prss/candidate.py` | 44 | 候选加宽 d>k 消失；Γ_θ 自带 A_τ/E_τ 适配（new_method §四） |
| `src/prss/auxiliary.py` | 147 | `AuxiliaryBatch`/`build_auxiliary`（top-down outside 传播）被 `CutRecord` + 固定测试取代（delta §9） |
| `src/prss/losses.py` | 78 | `response_loss` 只服务 reader 支路；三个 `spectral_tail_*` 是全树死代码。宿主 task loss 直接用 `F.binary_cross_entropy`（`jodie_loop.py:180` 现状） |
| `src/prss/core.py` | 146 | `PRSSCore` 整体由顶层包 `src/rpbe/` 取代 |
| `test/test_spectral.py`、`test_compressors.py`、`test_auxiliary.py`、`test_synthetic_tree.py` | — | 谱商/reader/合成树识别门禁全部失效 |

### 1.2 幸存模块的硬 import（随 §1.3 包名迁移同步手术）

TGB 线文件在 §1.5 整包删除，其中 4 个旧引用点（`tgn_pyg_bridge.py`、`tgn_pyg.py`、`event_loop.py`、`scripts/train.py`）随之消失。幸存模块搬家到 `src/rpbe/` 时必须同步重写的 import 点：

`hosts/base.py:15`（AuxiliaryBatch→CutRecord）、`hosts/jodie_bridge.py:14`、`training/jodie_loop.py:23`（InterfaceData 删除）、`scripts/train_jodie.py:43`，以及 `scripts/`、`test/` 全部 `from prss.…` → `from rpbe.…`

### 1.3 包名整体退役：`prss` → `rpbe`

PRSS 作为方法名与包名**全部废弃**。新架构统一为**顶层包 `src/rpbe/`**，`src/prss/` 整个目录在 develop_RPBE 上删除。幸存模块按下列映射搬家（删除清单 §1.1/§1.5 之外的"改"类文件都走这张表）：

| 旧路径（`src/prss/`） | 新路径（`src/rpbe/`） | 处置 |
|---|---|---|
| `state.py` | `state.py` | **改名改字段**。`QuotientState`→`OccurrenceState`，字段只留 `z`（删 raw/candidate——"Quotient/candidate"是谱商时代命名）。`RecursiveOccurrence` 的 `children`/`child_relations`/`child_delta_t`/`local_features` 不动，**metadata 新增契约键** `node`(全局节点 id) / `time`(as-of 时间) / `own_raw`(瓶颈前原始局部输入 o_v)。`RecursiveTrace.postorder()` 原样复用做 bottom-up |
| `monitoring.py` | `monitoring.py` | 保留 `MonitorWriter` 与通用件；**删** `validate_spectral`(:168-180) 与 `save_projection_snapshot`(:188-197)（当前树内无调用方，只读谱快照键）；**加** `validate_kf`（Ĵ_τ 有限性 + 界 `0 ≤ Ĵ_τ ≤ d_τ`）与 `save_kf_snapshot`（固定映射指纹） |
| `training/isolation.py` | `training/isolation.py` | **重写**（见 §五）。`counts_of_spectral`/`r_copies`/`r_max_change` 的谱三零语义消亡 |
| `config.py` | `config.py` | **重写**为 `RPBConfig`。`InterfaceSpec` 缩为 `{name, host_dim}`（r_τ 由宿主决定）；删全部谱参数（`lambda_spec`/`gram_ema_rho`/`spectral_*`/`procrustes_align`/`variant`） |
| `training/checkpoint.py` | `training/checkpoint.py` | 与谱商零耦合，搬家后保留（删 PyG msg-store 段，见 §1.5）。调用方把 `prss_core` 键换成 `rpbe` |
| `hosts/official_tgn/`（整包 vendored） | `hosts/official_tgn/` | 内容零改动，但**内部 import 前缀全部重写**：`prss.hosts.official_tgn.xxx` → `rpbe.hosts.official_tgn.xxx`（vendored README 的 provenance 说明同步更新） |
| `data/jodie.py`、`hosts/base.py`、`hosts/jodie_tgn.py`、`hosts/jodie_bridge.py`、`training/jodie_loop.py` | 同名搬家 | 随 §三/§四 的接入改造一并改 |

配套：

- `src/prss/__init__.py` 删除（连同整个 `src/prss/` 目录）；新 `src/rpbe/__init__.py` 导出 `RPBConfig`/`RecursiveCompressor`/`FixedMaps`/`CutRecord`/`kf_loss`
- `scripts/` 与 `test/` 的全部 `from prss.…` → `from rpbe.…`；`python -c "import prss"` 验证改为 `import rpbe`
- `pyproject.toml`：`packages.find where=["src"]` 自动发现 `rpbe`；发行名可保持 `prss2` 不改（pip 名与 import 名无关）；重装后删除陈旧的 `src/prss2.egg-info/`
- 宿主无关红线测试（`test_core_contract.py` 改写版）指向新包：`rpbe` 核心模块不含 `tgn_layer_`/`tjo:`/`tgp:` 等宿主字符串

### 1.4 测试处置

- **删**：`test_spectral.py`、`test_compressors.py`、`test_auxiliary.py`、`test_synthetic_tree.py`
- **新增**：`test_rpbe_records.py`（步 3）、`test_rpbe_maps.py`/`test_rpbe_loss.py`（步 4）、`test_rpbe_loop.py`（步 5 起，端到端 smoke）
- **改写**：`test_core_contract.py`（宿主无关红线指向新文件）、`test_experiment.py`（job_id 的 variant→rpbe 开关）、`test_jodie_adapter.py`（保留恒等平价②，trace 结构断言换新，谱审计换新）、`test_jodie_loop.py`（`select_trace_rows`/`metric_bundle` 保留，谱 smoke 换 RPBE 诊断）
- **原样保留**：`conftest.py`、`test_checkpoint.py`、`test_jodie_data.py`、`test_jodie_vendor.py`

### 1.5 整条 PyG-TGN / TGB 线在 develop_RPBE 上删除

每个宿主单独一条 loop；本分支只做官方 TGN。TGB 是将来独立的第二条 loop（main 上有完整历史可恢复），当前分支不保留：

| 删除对象 | 说明 |
|---|---|
| `src/prss/hosts/pyg_models/`（整包）、`src/prss/hosts/tgn_pyg.py`、`src/prss/hosts/tgn_pyg_bridge.py` | PyG 宿主 + vendored TGB 基线（含 `early_stopping.py` 死代码） |
| `src/prss/data/tgb_link.py`、`src/prss/training/event_loop.py` | TGB 数据封装 + 链路预测 loop（TGB 的专属 loop，将来重建） |
| `scripts/train.py`、`scripts/inference.py`、`scripts/protocol_ab.py` | TGB 线的三个入口 |
| `test/test_tgb_smoke.py` | 随线删除 |
| `configs/experiments/tgbl_wiki_*.yaml`、`experiment/runner.py`/`summarize.py` | TGB 实验矩阵与汇总（将来按新 loop 重建） |
| `src/prss/training/checkpoint.py` 中的 PyG `TGNMemory` msg-store 段（`:50-66` 及相关 helper） | 官方 TGN 的 memory 有独立的 `backup_memory/restore_memory`（`official_tgn/modules/memory.py:48,55`），不依赖该段 |
| `pyproject.toml` / `requirements.txt` 的 `tgb`、`torch-geometric` 依赖 | 本分支不再需要（`test_checkpoint.py` 里依赖 PyG 的用例同步删） |

删除后 `src/rpbe/hosts/` 只剩：`base.py` + `jodie_tgn.py` + `jodie_bridge.py` + `official_tgn/`（vendored 上游，内容零改动，import 前缀重写）。

---

## 二、新包 `src/rpbe/`（顶层包，全部 PRSS 定义已废弃）

数据流：宿主前向（adapter 逐节点算 z 并建 trace）→ `CutBuilder`（trace + FutureIndex → CutRecord 列表，每 cut 一条或多条 (c,y) 行）→ `FixedMaps`（no_grad 出 P）→ 按 τ 分组求 `(Z_τ, P_τ)` → 白化协方差 + Cholesky 解 `Ĵ_τ` → `kf_loss = −Σ α_τ·Ĵ_τ`。**无 witness、无交叉拟合、无路径增益、无树 DP。**

### `records.py` —— CutRecord 与未来 continuation 索引

```python
@dataclass
class CutRecord:
    tree_id: int; cut_id: int; occurrence_id: int; tau: str
    node: int; time: float
    z: Tensor                    # [r_tau]，带梯度（网络输出）
    context: Dict[str, Tensor]   # 原始 C：Δt、候选对端 j、role、query_type、关系、静态特征
    outcome: Tensor              # 原始 Y（该 (c,y) 行真实观察到的结果）
    valid: bool                  # 是否观察到（越界/censored 的 cut 不进入 loss）

class FutureIndex:
    """按 node 分组、按时间排序的事件表；np.searchsorted 找第一个 time > t 的事件。"""
    def query(self, node, t, k=1) -> List[dict]
```

**Y 从哪来**（delta §1 纠正：必须是数据真实观察的结果，不是教师输出）——每 cut 构造若干 **(c,y) 行**，一行一个联合测试：

- **阶段一（自监督预训练）**：1 个正行 `c=(Δt, 真实下一事件对端 j, role="source", query_type="link")`, `y=1` + `--neg-per-cut`(默认 4) 个负行（随机候选 j′，`y=0`）；z 在这些行上重复（同一 cut 的同一 z 配不同未来测试）
- **阶段二（节点分类）**：1 行 `c=(Δt, 下一真实事件对端 j 的 id 哈希, role="source", query_type="node_class", 静态特征)`, `y=下一真实事件的 state_label`（label 天然含 0/1 两类）

**valid 语义**（delta §4 红线的简化保留）：`valid=False`（数据窗口不够长 / 未来事件落在 val/test）≠ `y=0`（观察到但结果为否）。invalid 的 cut 整体不进入 loss 行集合。

**泄漏边界**：FutureIndex 建在完整时间流上，事件携带 split 标记（train/val/test）；cut 的未来若落在 val/test 区间一律 `valid=False`，**绝不读取其内容**。评估期根本不构建 CutRecord。

**`CutBuilder`**：`rpbe/hosts/base.py` 的 `OutsideBridge` 改名为 `CutBuilder`，`build(...) -> List[CutRecord]` 取代 `-> AuxiliaryBatch`。本分支唯一实现是 `rpbe/hosts/jodie_bridge.py`（阶段一 link 场景 / 阶段二 node_class 场景各一个构造）。

### `maps.py` —— 固定测量映射（全部 `requires_grad=False`，随机矩阵按 `--rpbe-seed` 预生成为 buffer）

```python
class FixedMaps(nn.Module):
    def context_vector(self, C) -> Tensor         # [d_c]  类别 one-hot/hashing + 连续量标准化后 RFF
    def future_vector(self, Y) -> Tensor          # [d_f]  类别 one-hot/hashing；回归量标准化后 RFF
    def psi(self, C, Y) -> Tensor                 # [m]  ψ(c,y) = sketch([1; φ_C(c)] ⊗ φ_Y(y))
    def isolation_fingerprint(self) -> Dict       # 随机矩阵种子/版本 + buffer 内容指纹
```

- **必须是张量积，禁止简单拼接**（算法稿 §2：拼接只能保留边缘信息，不能描述 context–outcome 关系）
- **TensorSketch 实现**：`(d_c+1)(d_f+1) ≈ 33×33 = 1089` 很小，用**预生成稀疏 CountSketch 矩阵**（1089×m，每列 ~10 非零）一次稀疏 GEMM，不需要 FFT 卷积
- `psi` 全程在 `torch.no_grad()` 下算；**P 侧永不产生梯度**（C、Y 来自数据，映射固定）
- **无 witness、无 JL 见证**：压缩前参考侧随 U-B 链条一起废除

### `compressor.py` —— 递归压缩器 Γ_θ

```python
class RecursiveCompressor(nn.Module):
    def __init__(self, cfg, host_aggregator):
        # A_τ: ModuleDict[Linear(d_o_τ → D)]     输入适配 (new_method §四.1)
        # Q_τ: ModuleDict[Linear(D → r_τ)]       输出头   (new_method §四.5)
        # G:  共享非线性核心 MLP(2D → D) + 残差   (new_method §四.4)
        # Agg: 注入的宿主聚合器（同一模块对象，不复制参数）
    def compress(self, *, tau, own_input, child_states, child_edge,
                 child_roles, child_delta_t, mask) -> Tensor   # [N, r_τ]
```

**TGN 退化**（new_method §五末段明确允许）：同质邻居 + 单一递归运算 → 删掉接口 embedding / child role embedding / operator embedding，`E_τ` 退化为直接拼接 child z + 宿主自带的 edge_time/edge_features/mask。**沿用宿主原本的聚合器**是方法要求，不是妥协。

**无 lift / 无 J_τ / 无 r_B**：Ky Fan 损失直接按 τ 分组用 `z_v` 本身（白化处理尺度），旧版"统一 loss 坐标"概念随 U-B 链条废除。

### `loss.py` —— Ky Fan 谱分数数值实现

```python
def kf_score(Z, P, eps_z, eps_p) -> Tensor:
    # μ_Z, Σ_ZZ, Σ_ZP, Σ_PP（按 τ 组内加权估计，行 = 有效 cuts 的 z 与 p）
    # Ĵ_τ = tr[(Σ_ZZ+ε_Z·I)⁻¹ Σ_ZP (Σ_PP+ε_P·I)⁻¹ Σ_PZ]
    # 实现：两次 Cholesky 分解 + 三角求解（solve 后 Frobenius 平方），无 SVD、无显式逆
def kf_loss(scores: Dict[tau, Tensor], alphas) -> Tensor:
    # L_KF = −Σ_τ α_τ·Ĵ_τ（α_τ 默认 1；等价正形式 Σ α_τ(d_τ − Ĵ_τ) 与 θ 无关项之差）
```

**梯度路径**（唯一要背熟的正确性约束）：

```
P 侧：FixedMaps 固定 + no_grad → p_v 无梯度，Σ_PP 对 θ 是常数
Z 侧：z_v 带梯度（经 Γ_θ + 宿主 ω）；Ĵ 对 z 可微，梯度自动流经 Σ_ZP 与 Σ_ZZ 两条路径
数值稳定：Σ_ZZ 求逆/求解在 z 塌缩时奇异 → ridge ε_Z 兜底；可选对 Σ_ZZ^{-1/2} 加 sg 以稳梯度
```

- **无交叉拟合**：损失就是当前 batch 的 profiled-MSE（闭式消去 A）。batch 内 profile 的乐观偏差记录在 §八 风险中，第一版不做树级交叉拟合
- **白化按 batch 估计**：p 与 z 的白化统计（μ、Σ）用当前 batch 的该 τ 行估计（+ridge）；跨 batch 无 running stats（隔离审计更简单）
- **无 SVD**（算法稿明确）：训练路径只有 Cholesky/线性方程；谱分解仅作诊断（可选）
- 每 τ 行数 < `min_cuts_per_type`(默认 32) 时跳过该 τ 的项并计 skip（warning，不是 error）

### RPBE 作为单一整体组件（无变体、无注册表）

**RPBE 整体作为一个组件接入，不拆分可插拔轴。** 编译期只保留**开启/关闭**：开启时递归压缩、固定测量、Ky Fan 谱分数一起工作；关闭时就是 TGN 原本的固有流程（不引入任何命名或可选变体）。消融不是当前目标——旧五变体（`vanilla/random/pca/direct/spectral`）随 `compressors/` 整包删除，新方法不复制任何"R 从哪来"式的变体轴。

**旧 PRSS 作对照**：决策 #1 是"在 `develop_RPBE` 分支删除"，因此 main 上旧方法原样可运行，"谱商 vs RPBE"对照无需额外设计（`git checkout main` 即可）。仅在将来合回 main 时才需重新决定是否保留一份。

- 旧 `Compressor` 契约直接废弃，不迁移：它的 7 个成员里 6 个直接是 R 的属性（`projection()`:39-41、`statistic`:29、`update_projection`:32、`spectral_loss`:54-56、`set_projection_trainable`:58-60、`InterfaceData`:17-21），RPBE 里没有 R；`project(candidate)`:43-44 的单候选签名与 Γ_θ 的 `(o_v, {z_children}, ξ, ℓ, mask)` 输入根本不匹配。
- 若将来要做消融，再加（届时才需要注册表）；本阶段不为它提前设计。

### `config.py` → `RPBConfig` 关键字段

`interfaces`(τ→r_τ) / `width_D`(128) / `m`(256) / `d_c`,`d_f`(32) / `lambda_kf`(1.0) / `alpha_τ`(默认全 1) / `ridge_eps_z`,`ridge_eps_p` / `neg_per_cut`(4，阶段一) / `min_cuts_per_type`(32) / `rpbe_seed`

---

## 三、宿主线接入（官方 TGN，本分支唯一宿主；`rpbe/hosts/`）

### `local_features` 清零问题（随 witness 废除而消失）

`rpbe/hosts/jodie_tgn.py:77-79, 211-212` 的 C1 清零（把 `flat_preagg` 前 `host_dim + n*host_dim` 列，即孩子状态块，清零）曾是"outside 不读子树状态"的 PRSS 约束。新算法**没有 witness / 压缩前参考侧**——Ky Fan 损失只消费 `z_v`（trace 的 `state.z`）与来自数据的 `(c,y)`，`local_features` 无消费者。

**处理：维持现状即可**。不清零也不改语义：字段保留（供未来诊断），C1 清零留在原地无害；恒等平价测试不受影响（清零只写进 trace，`z` 算完之后）。

### JODIE 线（`rpbe/hosts/jodie_tgn.py`）

- `make_candidate` + `project` 两步（`:161-162`、`:207-208`）→ 一步 `rpbe.compressor.compress(...)`（仅当 `--rpbe` 开启；关闭时 adapter 完全消失，宿主裸跑）
- `own_input = raw_source`（`:154-156`，memory + node features）；`child_states = cat([source_lower, neighbor_lower])`（`:173-187`）；`Agg = host.aggregate(...)`（`:192` 原样保留）
- 删 `traced_candidates`（`:97-106`，原 PCA 统计源）
- metadata 补 `node`/`time`/`own_raw`：**所有 occurrence 的 as-of 时间统一取该树的查询时间戳 `timestamps[row]`**（官方递归对邻居复用查询时间戳：`jodie_tgn.py:181-186` 的 `np.repeat(timestamps, n)`，邻居状态是"截至查询时刻"的）；node 分别取 `source_nodes[row]` / `flat_neighbors[idx]`。未来索引必须从 `time = 查询时刻` 往后查，若误用采样边时间会把 (edge_time, t] 内已发生的过去事件当成"未来"
- **layer 0 是真实切口**：它没有孩子，但 `z_0` 是父节点唯一能看到的东西，`z_0` 与 layer 0 的 `(c,y)` 进入 `tjo:layer0` 组的 Ky Fan 分数。全部 τ（含 layer0）默认计入，不做开关

---

## 四、两阶段训练接线（官方 TGN，本分支唯一宿主）

### 4.1 两个阶段的独立 loop

每个宿主模型单独一条 loop；同一宿主内两个训练阶段各一条。本分支只有官方 TGN：

- `rpbe/training/jodie_loop.py` —— **阶段二（节点分类）loop**，由现有 loop 改写而来
- `rpbe/training/pretrain_loop.py` —— **新增：阶段一（自监督链路预测）loop**，克隆官方 `old/tgn/train_self_supervised.py` 的协议（BCE 正负样本、`backprop_every` 累积反传、每 epoch 重置 memory、反传后 `detach_memory`、val AP 早停、memory 备份/恢复）
- `scripts/train_jodie.py` —— 阶段二入口（现有脚本改写）
- `scripts/train_pretrain.py` —— **新增**：阶段一入口，产物 = 阶段二 `--pretrained-checkpoint` 的宿主权重（+ 可选 Γ_θ 权重）

### 4.2 两个开关正交：阶段一 `--stage1-rpbe` × 阶段二 `--rpbe`（决策 #4）

**两个阶段都可以独立关闭 RPBE。全部关闭 = 官方 TGN 原汁原味的两阶段基线**（阶段一自监督预训练 → 阶段二冻结宿主 + 只训 decoder），这就是对比锚点。

| 阶段一开关 | 阶段二开关 | 含义 |
|---|---|---|
| `--stage1-rpbe` 关（默认） | `--rpbe` 关（默认） | **官方基线**：只预训练 TGN 宿主（官方协议原样，无 adapter、无 RPBE）；阶段二冻结宿主训 decoder（`--finetune-host` 默认 False，官方语义） |
| 关 | 开 | Γ_θ 在阶段二随机初始化，与宿主**联合微调**（`--finetune-host` 自动 True）——旧 PRSS 的接入时机 |
| 开 | 开 | **完整 RPBE**：阶段一起宿主 + Γ_θ 联合预训练（`L = L_link − λ_kf·Ĵ`，link 场景 C/Y），阶段二加载两者继续联合微调（node_class 场景 C/Y） |
| 开 | 关 | Γ_θ 预训练后阶段二弃用（无意义组合，矩阵不跑） |

**`--finetune-host` 默认值跟随 `--rpbe`**：关→False（官方冻结协议），开→True（联合微调）；显式传参可覆盖（保留为实验逃生口）。

**"关"的含义必须是拔掉而不是冻结**：开关关闭时 RPBE 组件完全不构造（`rpbe = adapter = cut_builder = None`，`tgn.embedding_module` 保持官方 `GraphAttentionEmbedding`，不建 trace、不建 CutRecord、无 RPBE loss 项），前向逐位等于官方 TGN。**禁止**做成"模块存在但 requires_grad=False"——那样前向输出是 Γ_θ 算的而非官方聚合输出，基线就不再是官方 TGN。实现沿用现有开关模式：`train_jodie.py:198-242` 的 `None` 分支 + `jodie_loop.py:168` 的 `if self.adapter is not None`。阶段一同理（`--stage1-rpbe` 关 = 无 adapter 无 trace，逐位对齐官方自监督脚本）。

### 4.3 总 loss 与 optimizer（两个阶段同构）

```python
kf = rpbe.kf_loss(cuts)                                    # −Σ α_τ·Ĵ_τ（组件内唯一 loss）
main_loss = task_loss + args.lambda_kf * kf               # 决策 #3；开关关时无 kf 项
main_loss.backward()                                      # 一次反向
optimizer.step()
tgn.memory.detach_memory()                                # 官方截断不变式，位置不变
```

optimizer 单个，按 delta §二纠正：宿主参数 + `rpbe.compressor.params` + task head（阶段一无 decoder，阶段二有）。`FixedMaps` 全是 buffer 无参数。**删除第二个 `unrestricted_optimizer`**（`train_jodie.py:261-262`、`jodie_loop.py:164-165,215-217`）。

- `--finetune-host` 默认值跟随 `--rpbe`（§4.2），显式传参可覆盖
- 阶段一的 `backprop_every` 累积语义照官方保留；Ky Fan 项按同样粒度累计进 `main_loss`，随 `backprop_every` 一起反传

### `rpbe/training/jodie_loop.py` 具体改动（阶段二）

- 删 `from prss.compressors import InterfaceData`(:23)、`unrestricted_optimizer` 全链、`aux_use_resp/aux_use_spec`(:112-115)、谱统计块(:228-251)
- `train_epoch`(:134-273)：主前向与 `task_loss`(:174-180) 照旧；`bridge.build` → `cut_builder.build` 出 `List[CutRecord]`；`rpbe.kf_loss(cuts)`；总 loss 合成（`--rpbe` 关时无 kf 项，且 `--finetune-host` 默认 False）
- `evaluate_split`/`replay_split`(:276-326)：`set_spectral_updates_allowed(False)` → `rpbe.eval()`（关 dropout、不构建 CutRecord；`clear_trace()` 已有）
- `_full_official_embedding_call` 的 `grad_enabled` = `--rpbe` 开关值（开则联合训练，关则官方 no_grad 冻结语义）

### `rpbe/training/pretrain_loop.py`（新增，阶段一）要点

- 协议克隆官方 `old/tgn/train_self_supervised.py`：`BCE(pos,1)+BCE(neg,0)`、随机负采样、`backprop_every` 累积、每 epoch 重置 memory、val 双场景的 memory `backup/restore`（`:290-314` 的官方语义）
- `--stage1-rpbe` 关：无 adapter、无 trace，逐位对齐官方
- `--stage1-rpbe` 开：挂 adapter + CutBuilder（link 场景），`main_loss = L_link − λ_kf·Ĵ` 随 `backprop_every` 一起累计反传；评估期同样走 §五 三零审计
- 产物由 `scripts/train_pretrain.py` 落盘：`outputs/pretrained/{dataset}/best.pt`（host 权重 + 可选 Γ_θ 权重）

### CLI 变更

- `--variant {vanilla,random,pca,direct,spectral}` → `--rpbe`（`action="store_true"`，单一开/关开关；不传就是 TGN 固有流程）
- **删（旧压缩器超参数，全部）**：`--candidate-dim`/`--candidate-hidden`/`--context-dim`/`--reader-hidden`/`--lambda-resp`/`--lambda-spec`/`--gram-ema`/`--spectral-warmup`/`--spectral-interval`/`--spectral-step-size`/`--trace-mode`（`--trace-roots` 保留）
- **加（RPBE 自身的超参数）**：`--kf-lambda`/`--rpbe-width`/`--sketch-dim`/`--neg-per-cut`/`--min-cuts-per-type`/`--ridge-eps`/`--rpbe-seed`
- **阶段一入口 `scripts/train_pretrain.py`（新增）**：`--stage1-rpbe`（决策 #4 的控制开关，默认关）+ `--backprop-every` + 上述 RPBE 超参数（开时生效）；产物 `outputs/pretrained/{dataset}/best.pt`（含 host 权重，`--stage1-rpbe` 开时另存 Γ_θ 权重）
- 阶段二入口 `scripts/train_jodie.py` 的 `--pretrained-checkpoint` 改为可选：不传则从随机初始化（配合从头联合训练实验）

### 必修的 checkpoint 加载 bug

`scripts/train_jodie.py:128-133` 的 `unwrap_state` 只认顶层 `model_state_dict`/`tgn` 两个键；新阶段一 loop 的产物建议直接存 `{host: {...}}` 单层格式规避该问题。若过渡期加载历史 `outputs/jodie_wikipedia/vanilla_seed0/best.pt`（`{"model": {"decoder":…, "tgn":…}}` 两层嵌套），需给 `unwrap_state` 补一条 `model.tgn` 路径。`:176-180` 的 `MEMORY_STATE_KEYS` pop 逻辑保留（memory 是运行时状态，不跨 run 携带）。

---

## 五、审计 / 监控 / 汇总的替代 schema

### `training/isolation.py` 重写 —— 新三零审计

旧的"Gram 计数 / R 矩阵 / trace"三零在新方法下无对象。新审计对象是"eval 期不可变的三件事"：

```python
def rpbe_state(rpbe) -> Dict          # {fingerprint: maps.isolation_fingerprint(), training: rpbe.training}
def assert_clean(before, rpbe, trace_created, label):
    # 1) trace_created 必须 False（评估期不建计算树）
    # 2) fingerprint 不变（固定映射的标准化统计/种子未被 eval 数据触碰 → 防泄漏）
    # 3) 训练态未被翻回
```

调用点同步改：`jodie_loop.py:329-341`（阶段二）、新增的 `pretrain_loop.py`（阶段一）、`scripts/train_pretrain.py`。

### 实验矩阵与汇总

TGB 的 `experiment/runner.py`/`summarize.py` 随线删除。本分支阶段内先以 CLI + shell 串联两个阶段；正式矩阵待阶段一/二跑通后按新 loop 重建（矩阵维度预置：`stage1_rpbe × rpbe × seed × λ_kf`）。

`summary["spectral"]` → `summary["kf"]`：

```json
{"J": {"tjo:layer0":…, "tjo:layer1":…}, "J_frac": {"tjo:layer1":…},
 "kf_loss":…, "skipped_types":…}
```

机制列输出：每个 τ 的 `Ĵ_τ` 与能量占比 `Ĵ_τ/d_τ`（d_τ = P 白化维数）。`Ĵ_τ` 是"压缩状态能读出多少未来嵌入能量"的直接证据（算法稿 §3：`min_A E‖p−Az‖² = d − J`）。谱分解（M 的特征值、rank-r 截断误差）**仅作诊断可选输出，不进入训练**。

---

## 六、分步实施顺序

环境约束：**本机 Windows torch CPU 无 CUDA；未经同意不跑全量测试与训练**。本机可跑的只有 `compileall`、`import rpbe`、纯 torch/numpy 的 rpbe 单测。JODIE 数值测试与一切训练都在云端 GPU。

| 步 | 内容 | 里程碑（跑测试前先征得同意） |
|---|---|---|
| 0 | 建 `develop_RPBE` 分支；本计划落盘为根目录 `RPBE_PLAN.md` | `git branch` 有该分支，main 无新提交 |
| 1 | 删 §1.1 全部文件 + **§1.5 整条 TGB 线** + §1.2 五处 import 手术 + **§1.3 包名迁移（`src/prss/` → `src/rpbe/` 搬家 + vendored import 前缀重写）** + 删旧测试；`build_components` 先只保纯宿主直通（不开 RPBE） | 本机：`python -m compileall src scripts` 全过；`python -c "import rpbe"` 过 |
| 2 | `state.py` 改名改字段 + adapter 的 metadata 扩展（`node`/`time`/`own_raw`）+ 改写 `test_jodie_adapter.py` trace 结构类（C1 清零不动） | 云端：`test_jodie_adapter.py` 绿，**恒等平价仍绿**（证明主前向未被动过） |
| 3 | `rpbe/records.py`（CutRecord + FutureIndex + 阶段一/阶段二两种场景的 CutBuilder） | 本机：合成小事件流单测绿——valid 与 y=0 严格区分、searchsorted 正确、禁止字段不出现、两种 query_type 各建对、阶段一正负行配比正确 |
| 4 | `rpbe/maps.py` + `rpbe/compressor.py` + `rpbe/loss.py`（Ky Fan 分数） | 本机纯张量单测绿：固定映射同 seed 同输出；ψ 为张量积结构；**profiled-MSE 恒等式 `min_A E‖p−Az‖² = d − J`**；`0 ≤ Ĵ ≤ d` 界；白化后单位协方差；Cholesky 路径无 SVD |
| 6 | `monitoring.py`/`isolation.py` 收尾 | 本机：`test_checkpoint.py` 绿；云端：三零审计在阶段一/阶段二的 eval/replay/test 全链路通过 |
| 7 | **冒烟**：阶段一 `train_pretrain.py --max-train 3000 --n-epoch 1`（`--stage1-rpbe` 关/开各一次）→ 阶段二 `train_jodie.py --rpbe --max-train 3000 --max-val 1000 --max-test 1000 --n-epoch 1` | 云端：loss 有限、`validate_kf` 无 error、各 τ 的 Ĵ 有正值且随训练上升、val AUC/AP 落盘 |
| 8 | 正式矩阵：`stage1_rpbe × rpbe × seed × λ_kf` + 汇总出表 | 云端：每 job 产出 summary.json + 机制列 |

**数据**：`old/processed_tgn_data/ml_wikipedia.*`（本机与云端各一份）。

---

## 七、验证方式

**本机（CPU，随时可跑）**
```bash
python -m compileall src scripts          # 步 1 后必过
python -c "import rpbe; from rpbe.state import RecursiveTrace"
python -m pytest test/test_rpbe_loss.py test/test_rpbe_records.py -q    # 步 3-4 新增
python -m pytest test/test_checkpoint.py test/test_jodie_data.py test/test_experiment.py -q
```

**云端 GPU（每次执行前先确认）**
```bash
python -m pytest test/ -q                 # 全量回归
# 阶段一（自监督预训练）：默认只预训练宿主
python -m scripts.train_pretrain -d wikipedia --data-dir old/processed_tgn_data \
    --output outputs/pretrained/wikipedia --max-train 3000 --n-epoch 1
# 阶段一（RPBE 从第 0 步参与联合预训练）
python -m scripts.train_pretrain -d wikipedia --data-dir old/processed_tgn_data \
    --output outputs/pretrained/wikipedia_rpbe --stage1-rpbe --max-train 3000 --n-epoch 1
# 阶段二（节点分类联合微调）
python -m scripts.train_jodie --rpbe -d wikipedia \
    --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/pretrained/wikipedia/best.pt \
    --output outputs/rpbe/smoke --max-train 3000 --max-val 1000 --max-test 1000 --n-epoch 1
```

**关键数学性质单测（步 4，本机纯 torch）**
1. profiled-MSE 恒等式：合成 (z,p) 数据，数值验证 `min_A E‖p−Az‖² = d − J`（J 由 Cholesky 路径算，A* 由闭式算出后与数值拟合对照）
2. Ky Fan 上界：线性高斯构造下（m(H) 线性、M 已知），取 top-r 特征向量坐标 `z_j = v_jᵀm/√λ_j` 验证 `J = Σ_{j≤r} λ_j(M)`，且随机 z 的 J 不超此值
3. 白化不变量：白化后 `E[zzᵀ] = I_r`、`E[ppᵀ] = I_d`
4. 归一化界：`0 ≤ Ĵ ≤ d`（ridge 下略小于 d）
5. 梯度隔离：`p_v`（及 P 侧一切）`.grad_fn is None`；Ĵ 对 `z_v` 的梯度非零且流经 Γ_θ 参数

**方法有效性判据**：同宿主同数据下开 `--rpbe` 的 test AUC 对比纯宿主（JODIE wikipedia 基线 0.8871），以及机制列的 `Ĵ_τ` 随训练上升（说明 r_τ 维预算内读出的未来嵌入能量在增加）。

---

## 八、风险与必须盯住的点

| 风险 | 对策 |
|---|---|
| **协方差近奇异**：`Σ_ZZ`(r_τ×r_τ) 与 `Σ_PP`(m×m=256²) 在小 batch 下秩亏；z 塌缩时 `Σ_ZZ` 退化 | ridge 相对化 `ε·tr(Σ)/dim`；Cholesky 失败则该 τ 项跳过并计数；可选对 `Σ_ZZ^{-1/2}` 加 sg 稳梯度 |
| **batch 内 profile 乐观偏差** | 损失即当前 batch 的 profiled-MSE（闭式消去 A），batch 内估计有乐观性——算法稿未含交叉拟合，第一版接受并记录；若实验中 Ĵ 虚高再考虑树级交叉拟合升级 |
| **固定映射开销**：每 cut 若干行 × sketch | TensorSketch 用稀疏 GEMM 批量算；预计每 batch 成本约为纯宿主的 2~3 倍，必须先冒烟实测 |
| **on-policy 覆盖限制**（算法稿结尾明示） | 数据只有实际发生的 continuation，`C|H` 随历史变化 → 学到的只是 **on-policy predictive quotient**。要声称"所有合法 C"的等价需固定 continuation 采样/重要性权重/干预覆盖——第一版不做，报告如实按 on-policy 口径写 |
| **固定映射指纹泄漏** | 白化统计 per-batch 无 running stats；`isolation_fingerprint`（buffer/种子）在 eval/replay/test 前后比对；P 侧永不进梯度 |
| **基线公平性** | RPBE 模式参数量多于裸宿主，对比时必须报参数量差；另外官方阶段一用 `get_data`（含 10% 节点的 inductive 掩蔽），阶段二用 `get_data_node_classification`（无掩蔽），协议描述里要写清这个既有不一致 |
| **memory detach** | 无冲突：Ĵ 只消费 batch 内 cuts，跨 batch 截断位置不变（`jodie_loop.py:222-224`） |

## 关键文件清单

- `src/rpbe/hosts/jodie_tgn.py` —— metadata 扩展、`:161-162`/`:207-208` 换 `compress`（C1 清零不动）
- `src/rpbe/training/jodie_loop.py` —— 阶段二联合训练接线、删谱统计块
- `src/rpbe/training/pretrain_loop.py`（**新增**）—— 阶段一 loop，含 `--stage1-rpbe` 开关
- `scripts/train_jodie.py` / `scripts/train_pretrain.py`（新增）—— 两阶段入口、新 CLI、单 optimizer、修 `unwrap_state`
- `src/rpbe/state.py` —— trace 与 rpbe 包之间的数据契约
- 新增：`src/rpbe/{records,maps,compressor,loss,config}.py`
- 新增：`test/test_rpbe_*.py`
