# Metaⁿ × RPBE 工程任务书 v3

状态：**待逐条验收**。本文件只规定**契约、类、张量形状、缓存结构、训练生命周期、验收判据**。
实现者不得改变方法。

- v1、v2 已作废，以本文件为准。
- 基线代码：`meta-n-main`，blob `73e0d5097878d759cac0cdc32c67abe756d2b264`。
- proposal-space 来源：`PRSS2` 仓库 `fix/avg-lora-clock`，冻结提交 `c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36`。
- v3 相对 v2 的唯一改动来源：**census 结论**——一次 Metaⁿ run 只提供 1 个独立 root
  （`candidate_id="gen0_seed"` @ `evolutionary_orchestrator.py:933`，`beam_width=1` @ `:119`，
  `beam_candidates=1` @ `:120`）。故 `N_TREE_MIN` 由 128 降为 **32**。

---

## 0. 边界（不可更改）

1. 只修改 Metaⁿ 每次 Ω 调用前的 **context reduction**。注入点有**两处，必须同时覆盖**：

   | # | 位置 | 调用 | 路径 |
   |---|---|---|---|
   | 1 | `meta_n/core/omega.py:119-120` | `OmegaEngine.generate` | 主生成 |
   | 2 | `meta_n/core/omega.py:238-242` | `OmegaEngine.refine` | 自修复 |

   ```python
   # generate (119-120)
   sampled = self.context_manager.sample_traces(traces)
   truncated_stack = self.context_manager.truncate_context_stack(context_stack)

   # refine (238-242)
   sampled = self.context_manager.sample_traces(child_traces)
   refine_stack = list(context_stack) + [prev_injection]
   truncated_stack = self.context_manager.truncate_context_stack(refine_stack)
   ```

   **只改 119-120 会让自修复路径不受 predictive reduction 控制**，三种模式在 refine 路径上会静默退化一致。

2. 保留三种模式：`full`、`official`、`predictive`。
3. 不改 `archive`、`OmegaEngine.generate`/`refine` 本体、`_build_prompt`、`MetaLayer`、evaluator、现有 feasibility solver 内核。
4. frozen encoder、frozen LLM，只训练 Γ。
5. 当前 4080 SUPER 只跑 Metaⁿ + Ours；Official 与 Full Context 在另一张卡单独跑。
6. finite-test 推理只执行一次 context reduction，**不运行两步 rollout**。
7. **离线训练**：正式评估阶段不得更新 Γ（见 §5）。

---

## 1. 冻结常数（v3 已按 A′ 收缩）

| 名称 | v3 冻结值 | v2 值 | 说明 |
|---|---|---|---|
| `N_RUNS_TOTAL` | **64** | — | Phase A 独立 Official-reduction run 总数 |
| `N_TREES_TASK` | **32** | — | task 窗口 tree 数 |
| `N_TREES_PROTECT` | **32** | — | protect 窗口 tree 数 |
| `N_TREE_MIN` | **32** | 128 | 每窗口门禁单位 = unique tree |
| `M_SKETCH` | **8** | 32 | 每条 LPSE branch 的 P 维 |
| `B_V_DIM` | **32** | 128 | `= 4m`，`b_v` 维度 |
| `N_BRANCHES` | 4 | 4 | 仍然独立，禁止拼成单一 covariance |
| `D_E` | 256 | 256 | fusion 维度。**禁止用 LLM hidden 做 fusion dim** |
| `H_LORA` | 64 | 64 | `W_K`/`W_V` 低秩瓶颈 |
| `N_SLOTS` | 4 | 4 | 四槽 |
| `COND_RANK` | **32** | 全秩 | `Cond` 低秩 |
| `KAPPA` | 0.02 | 0.02 | Stage8 冻结几何标定，**不得按结果调** |
| `TAU` | 3e-4 | 3e-4 | certificate 容差，只影响 gate，不改变可行集 |
| `GAMMA_BUDGET` | 1_000_000 | | Γ 参数硬上限，启动 assert |
| `TEMPERATURE` | 1.0 | 1.0 | **固定** |
| `P_B_SEED` | 0 | 0 | `P_b`、`R_S^(r)`、`P_rc` 的固定投影种子 |

**降级路径（仅 pilot/smoke，不得作为正式 Metaⁿ 主结果）**：
若实际只能收集 32 个 root → `16 + 16`、`M_SKETCH=4`、`B_V_DIM=16`。
此配置必须在产物中显式标为 `pilot`，禁止与正式结果并排。

### 1.1 冻结文本编码器 `E_0`（v3 定版）

**单一 CodeBERT，全字段冻结。** `trace` / `InjectedCode` / `query` / `future observation`
**共用同一个 `E_0`**。

```yaml
model_id:   microsoft/codebert-base
revision:   3b0952feddeffad0063f274080e3c23d75e7eb39
tokenizer:  AutoTokenizer.from_pretrained(model_id, revision=<同上>)
model:      AutoModel / RobertaModel, revision=<同上>
hidden_dim: 768
output_dim: 256
pooling:    masked_mean
training:   false
dropout:    通过 model.eval() 关闭
gradient:   torch.no_grad() + detach()
```

**长文本分块（冻结）**：

1. 每块最多 **256 tokens，含 special tokens**；
2. **非重叠**分块；
3. 每块做 masked mean；
4. 按**有效 token 数加权**聚合各块；
5. 用 seed 0 的**固定 Rademacher 投影**降到 256：

   ```
   P_rc ∈ R^{768×256},  P_rc[i,j] ∈ { -1/sqrt(256), +1/sqrt(256) }
   ```

**不使用 text/code 双编码器。** 双编码器会产生两个未校准的表示空间，还需额外对齐模块，
而当前样本量更不足以支撑对齐。`all-MiniLM-L6-v2` 偏句子语义，对当前"trace + 报错文本 + 代码"
混合输入不如 CodeBERT 合适。

### 1.2 Γ 参数预算（`d_e=256, h=64, cond_rank=32`）

| 参数 | 形状 | 数量 |
|---|---|---|
| `A_c` | `[32, 256]` | 8 192 |
| `B_c` | `[1024, 32]` | 32 768 |
| — `Cond` 小计 | | **40 960** |
| `A_K` | `[256, 64]` | 16 384 |
| `B_K` | `[64, 256]` | 16 384 |
| `A_V` | `[256, 64]` | 16 384 |
| `B_V` | `[64, 256]` | 16 384 |
| `w_g` | `[256]` | 256 |
| `Q_0` | `[4, 256]` | 1 024 |
| `LN` weight/bias | `[256] ×2` | 512 |
| **合计** | | **108 288** |

`Cond(q) = B_c A_c q`。`W_K/W_V` 的 `rank=64` **保持不变**，只收缩 `Cond`。

---

## 2. 模块与类清单

```
meta_n/rpbe/__init__.py
meta_n/rpbe/config.py             # 冻结常数 + assert_gamma_budget()
meta_n/rpbe/encoder.py            # FrozenItemEncoder:  U_v -> X_v, q_emb
meta_n/rpbe/fusion.py             # SlottedFusion (Γ_θ)
meta_n/rpbe/selector.py           # PredictiveSelector
meta_n/rpbe/records.py            # CutMeta / CutRecord
meta_n/rpbe/lineage.py            # 两步真实 lineage 抽取
meta_n/rpbe/future.py             # phi_r(S_v) 固定 sketch
meta_n/rpbe/window.py             # StatWindow: B/P/R + OAS CCA
meta_n/rpbe/qp.py                 # proposal-space QP（移植）
meta_n/rpbe/context_reduction.py  # full | official | predictive 三模适配器
meta_n/rpbe/census.py             # archive census（§4.3）
meta_n/rpbe/trainer.py            # 离线 Γ 训练（Phase B）
```

### 2.1 `encoder.py`

```python
class FrozenItemEncoder:
    def __init__(self, spec: EncoderSpec): ...   # §1.1 全部字段
    @torch.no_grad()
    def encode(self, items: list[TextItem]) -> torch.Tensor:
        """-> X_v [n_v, 256]，detach。含固定的 type/depth encoding。"""
    @torch.no_grad()
    def encode_query(self, q: str) -> torch.Tensor:
        """-> q_emb [256]，detach。"""
```

**只输出 `X_v` 和 `q_emb`；不输出 `Q_v`。** `Q_v` 每个 boundary 在当前 Γ 下重算。

### 2.2 `fusion.py`

```python
class SlottedFusion(nn.Module):
    def __init__(self, d_e=256, h=64, n_slots=4, cond_rank=32): ...
    def forward(self, X_v, q_emb, mask) -> tuple[Tensor, Tensor]:
        """-> (Z_v [4,256], A_v [4,n_v])"""
```

冻结公式，**不得增删项**：

```
Q_v = Q_0 + B_c A_c q_emb          # [4, 256]
K = X_v W_K,  V = X_v W_V,  g = sigmoid(X_v w_g)
A_v = softmax_j( Q_v K^T / sqrt(d_e) + 1_4 log(g + eps)^T + M_v )
Z_v = LN( Q_v + A_v V )
```

- `W_K = A_K B_K`，`W_V = A_V B_V`，`h = 64`。**禁止 full-rank `d_e × d_e`。**
- `M_v`：**只屏蔽 padding**——所有 item 对所有 slot 合法，合法组合 bias 固定为 0。
  **不预定义四槽语义，也不写 semantic mask 表。**

### 2.3 `selector.py`

**必须返回原始对象，不能返回 `str`。** `_build_prompt` 签名：

```python
def _build_prompt(self, traces: list[Trace], context_stack: list[InjectedCode],
                  tasks, depth, inspiration_traces=None, previous_scores=None,
                  archive_best_scores=None, solver_language="python",
                  no_code_library=False, current_scores=None, helper_usage="",
                  focus_task=None, prompt_variant=None):
```

正确契约：

```python
class PredictiveSelector:
    def reduce(
        self,
        traces: list[Trace],
        context_stack: list[InjectedCode],
        attention: torch.Tensor,          # A_v [4, n_v]
        budget: ContextBudget,
    ) -> tuple[list[Trace], list[InjectedCode], dict]:
        """-> (selected_traces, selected_stack, diagnostics)"""
```

硬性要求：

1. 返回的 `Trace` / `InjectedCode` 必须是**原始对象**，**不得自行生成新文字**；
2. `selected_stack` 按 `source_depth` **恢复原顺序**；
3. `_build_prompt()` **一行不动**；
4. **不得追加 `C_v`**——`tasks` / `depth` / `scores` / `focus` / `solver_language` 等
   已由 `_build_prompt` 的其他参数渲染；
5. 加权统计量（weighted mean score、failure mass、failure-class distribution、
   source-depth distribution）**只能写进 `diagnostics`**；
6. 多 slot 选中同一 item 时，后面的 slot 取下一个最高权重的未使用 item；
7. **被保留的 code block 必须完整；允许整块丢弃，禁止中间截断。**

### 2.4 `records.py`

```python
@dataclass
class CutRecord:
    cut_id: tuple                # (run_id, root_candidate_id, occurrence_seq)
    tree_id: tuple               # (run_id, root_candidate_id)   —— v3 关键
    occurrence_seq: int
    X_v: torch.Tensor            # [n_v, 256]  frozen，缓存
    q_emb: torch.Tensor          # [256]       frozen，缓存
    mask: torch.Tensor           # [4, n_v]    固定
    p: torch.Tensor              # [8] per branch，detach
    r: float                     # 两步真实收益（标量）
    weight: float                # parent 归一化权重
```

**`tree_id` 必须是 `(run_id, root_candidate_id)` 元组，不是字符串 `"gen0_seed"`。**
`gen0_seed` 是硬编码的候选 id，跨 run 完全相同；若只用它做 tree 身份，
**64 个独立 run 会全部碰撞成同一棵树，门禁形同虚设。**

**没有 `z` 字段。没有 graph。没有 `Q_v`。没有 `A_v`。**

### 2.5 `lineage.py`

- 只接受 archive 中**真实存在**的 `c_v → c_{v+1} → c_{v+2}`；
- `Y_{v+1} = Φ(T_{d+1})`、`Y_{v+2} = Φ(T_{d+2})`：future 是 **child/grandchild 的真实 trace observation，不是 InjectedCode**；
- 缺真实 grandchild 的 cut **不进入 LPSE**，**不得**补零 / 复制 child / 造假 EOS；
- 每条真实 parent→child→grandchild 是一个 occurrence；**按 parent 归一化权重**。

### 2.6 `future.py`

冻结为**直接 aligned pair**：

```
f_v^(1) = E_0(serialize(Y_{v+1}))                    # [256]
f_v^(2) = E_0(serialize(Y_{v+2}))                    # [256]
phi_r(S_v) = R_S^(r) [ f_v^(1) ; f_v^(2) ]           # [8]，R_S^(r) ∈ R^{8×512}
```

- `R_S^(r)` 是**固定随机投影，不训练**，每 branch 独立 seed。
- **不得加入 `f^(2) - f^(1)`（线性投影下冗余）或 `f^(1) ⊙ f^(2)`（改变 finite-test family）。**
- 全程 `detach`。child/grandchild 文字不得进入 `Γ(U_v)`。

### 2.7 `window.py`

```python
class StatWindow:
    def __init__(self, m=8, b_v_dim=32, n_branches=4, seed=0, oas=True,
                 min_unique_trees=32): ...
    def add(self, rows: list[CutRecord]) -> None: ...
    @property
    def ready(self) -> bool: ...          # len(tree_seen) >= 32
    def close_lpse(self, theta) -> ...     # 每 cut cotangent，绝不提前相加
    def close_task(self, theta) -> ...     # 聚合 cotangent
```

**OAS 必须显式启用。** 现有入口默认全是 `False`：

- `latent_z_adjoint(..., oas=False)` — `PRSS2:src/rpbe/loss.py:301`
- `KFMomentWindow(..., oas=False)` — `PRSS2:src/rpbe/loss.py:914`
- `kf_score()` / `kf_adjoint()` **完全没有 OAS 入口**

正确路径：`latent_z_adjoint(..., oas=True)`，或 `KFMomentWindow(..., oas=True)`
（`loss.py:1290` 内部转发 `oas=self.oas`）。
**不得直接调 `kf_score + kf_adjoint` 来"复用 OAS"——会静默丢掉 OAS。**

梯度纪律：

- **只有 OAS intensity 可以 detach**；
- `C_PP`、`C_RR` 侧固定；
- `C_ZZ`、scale normalization、`B` **必须保留梯度**；
- **严禁 stop-gradient `C_ZZ`**（会得到径向导数 `2J` 的半梯度，不是任何目标的梯度；验收 #7 专抓）。

```
J_LPSE = (1/4) Σ_r J_CCA(B, P^(r))
J_task = J_CCA(B, R)
```

两者都用 **paired OAS shrinkage + scale-normalized Cholesky**，**禁用 plain ridge inverse**。

---

## 3. 张量形状表（v3）

| 符号 | 形状 | 梯度 | 缓存? |
|---|---|---|---|
| `X_v` | `[n_v, 256]` | 无 | **是** |
| `q_emb` | `[256]` | 无 | **是** |
| `mask` | `[4, n_v]` | 固定 | **是** |
| `Q_v` | `[4, 256]` | 是（经 `Q_0`/`A_c`/`B_c`） | **否，每次重算** |
| `K` / `V` | `[n_v, 256]` | 经 `W_K`/`W_V` | 否 |
| `g` | `[n_v]` | 经 `w_g` | 否 |
| `A_v` | `[4, n_v]` | 是 | **否**（部署时现算） |
| `Z_v` | `[4, 256]` | 是 | 否 |
| `vec(Z_v)` | `[1024]` | 是 | 否 |
| `b_v` | **`[32]`** | 是 | 否（每 boundary 重算） |
| `φ_r(S_v)` | **`[8]`** | 无 | **是** |
| `B` | **`[M, 32]`** | 是 | 否 |
| `P^(r)` | **`[M, 8]`** | 无 | **是** |
| `R` | **`[M, 1]`** | 无 | **是** |
| `q_j` | `[len(Γ_params)]` | — | **否，每 boundary 重算** |

`P_b ∈ R^{32×1024}` 是固定投影（seed 0），不训练。

---

## 4. 缓存与窗口单位

### 4.1 三层缓存

**核心约束：Metaⁿ 的 Ω 是 LLM 调用，不可重放。** pass2 只能 **replay Γ**。

| 层 | 内容 | 生命周期 |
|---|---|---|
| L1 cut cache | `X_v / q_emb / mask / φ_r(S_v) / R_v / IDs` | Phase A 收集期 |
| L2 task window | 同上 + `B_task` + `R_task` | 跨 boundary 存活（`W_{t-1}`） |
| L3 protect window | 同上 + `P^(r)` | 当前窗口（`W_t`） |

**禁止缓存** graph-connected `z`、`Q_v`、`A_v`、旧 cotangent。
**禁止**缓存后重新调用 Ω，**禁止**在 pass2 重跑原始文本生成。

### 4.2 窗口门禁 = unique tree

`KFMomentWindow.window_ready()` 判据（`PRSS2:src/rpbe/loss.py:1340-1346`）：

```python
len(self._windows[tau]["tree_seen"]) >= self._threshold(tau)
```

`WeightedWelford.add(..., tree_ids_b)` 在提供 `tree_ids` 时**按 tree 聚类**算 `W_2` 和 `D`
（`loss.py:477, 517`）。故 v3 门禁：

```
N_unique_trees^task    >= 32
N_unique_trees^protect >= 32
首次投影需累计 >= 64 个独立 tree
```

硬性要求：

- **同一 tree 的所有 cut 必须进入同一窗口，整棵 tree 不得跨窗口**；
- `set(task_tree_ids).isdisjoint(protect_tree_ids)`；
- **64 个 root 用固定 hash 切分成 32 task / 32 protect；两组永不交换、永不重叠。**
  切分只做一次，固化进配置，不得按结果重切。

**⚠️ 直接后果（必须记录，避免误判）**：`N_RUNS_TOTAL = 64` 在门禁下**只够一次 boundary 更新**。
即正式配置是"**单次 Γ 更新**"的训练。若要多次更新，需增加 run 数让两个窗口滑动；
但滑动的两个窗口之间必须重新满足 tree 互斥，不得复用已进过 task 窗口的 tree 当 protect。

### 4.3 archive census（开工前必跑）

**在写任何训练代码之前**先跑 `census.py`，输出：

1. eligible lineage 数；
2. unique parent 数；
3. **unique tree 数（按 `(run_id, root_candidate_id)` 计）**；
4. 按 `N_TREE_MIN=32` 能组成的完整窗口数。

**若 unique tree < 64，禁止假装门禁已满足、禁止重复计数同一 tree 充数**，
只能按 §1 的降级路径标为 `pilot`。

---

## 5. 训练生命周期（必须离线）

```
Phase A：用 Official reduction 收集 64 次独立 run，冻结完整 archive trajectories（只读，不训 Γ）
Phase B：不调用 Ω，只从冻结 archive 构造两步 lineage，离线训练 Γ
Phase C：冻结 Γ，在新的正式评估 run 中执行 Predictive reduction
```

**Phase A 必须使用 Official reduction**（不是 predictive，也不是 full）来收集轨迹，
否则训练输入与评测输入不同源。

**Phase C 硬性约束：**

- 不读取未来 child/grandchild；
- 不构造 `R_v`；
- **不更新 Γ**；
- 只做当前 cut 的一次 reduction。

### 5.1 Phase B 边界伪代码

```
# ---- B1 门禁 ----
assert N_unique_trees(W_{t-1}) >= 32      # task 窗口
assert N_unique_trees(W_t)     >= 32      # protect 窗口
assert set(W_{t-1}.tree_ids).isdisjoint(W_t.tree_ids)

# ---- B2 在当前 Γ 下重建整窗（不碰 Ω）----
with no_grad:  B_all = [ P_b · vec( SlottedFusion(X_v, q_emb) ) for v in W ]
    # Q_v = Q_0 + B_c A_c q_emb 在此处用当前 θ 计算

# ---- B3 task 方向（W_{t-1}）----
J_task = J_CCA(B_task, R_task, oas=True)
a_v^task = ∂J_task / ∂b_v
L̃_task = -Σ_v ⟨sg(a_v^task), b_v(θ)⟩
g_task  = ∇_θ L̃_task = -∇_θ J_task

# ---- B4 protect 约束（W_t）----
J_LPSE = (1/4) Σ_r J_CCA(B_prot, P^(r), oas=True)
per-cut a_v^LPSE = ∂J_LPSE / ∂b_v            # 分别保存，禁止提前相加
for each cut v:
    L̃_v = ⟨sg(a_v^LPSE), b_v(θ)⟩ - ⟨sg(a_v^LPSE), sg(b_v(θ))⟩   # 数值恒 0
    q_v = ∇_θ L̃_v                                                # 梯度正确

# ---- B5 proposal-space QP（见 §6）----
```

**全程不调用 Ω。**

---

## 6. proposal-space QP（精确写法）

来源：`fix/avg-lora-clock@c05e4fb`，`src/rpbe_embodied/boundary.py::_proposal_space_update`。
语义与下式**逐条一致**，**移植而非重写**。

```python
theta_old = [p.detach().clone() for p in gamma_params]

clip_grad_once()                          # 同一个 clipped grad 用于 preview 与真实 moment

delta_adam = counterfactual_adamw_on_clones()   # clone + 深拷贝 optimizer state
                                                # 真实 m/v/step 不动

d_star = project(delta_adam)              # min 1/2||d - Delta_adam||^2
                                          # s.t. H_j^T d >= -kappa*||Delta_adam||  ∀j

if not certified:                         # v_max > tau
    return                                # 实参、真实 moments、scheduler 都未修改，无需回滚

optimizer.step()                          # 用同一个 clipped task gradient，真实 moments 前进一步

if n_active > 0:
    for p, o, dv in zip(gamma_params, theta_old, split(d_star)):
        p.data.copy_(o + dv.reshape(p.shape))   # 覆盖 optimizer 刚写的参数
else:
    keep_real_adamw_write_byte_for_byte()       # proj_n_active == 0：不重新舍入

scheduler.step()                          # 恰好一次
```

冻结值（`results/STAGE8_RESULTS.md`）：`kappa=0.02`、`tau=3e-4`。
Stage8 实测 `v_max ~1e-8` 对 `tau=3e-4`，四个数量级余量；`proj_shift_ratio` 均值 0.023、最大 0.143。

**写法禁忌**：

- 不得写 `θ ← θ + d*，全量覆盖，不是 +=` —— 自相矛盾。真实操作是 `copy_(theta_old + d_star)`：
  **用 pre-step 快照加投影增量覆盖**，不是对 optimizer 刚写入的值再累加。
- 不得写"提交虚拟 moments"。真实做法是让**真实 optimizer** 用同一个 clipped gradient
  `step()` 一次，然后在有投影时覆盖参数值。
- 有且仅有一次 `optimizer.step()`，且只在 certificate 通过后。

---

## 7. 诊断字段（每个 boundary）

| 字段 | 含义 |
|---|---|
| `N_unique_trees_task` / `N_unique_trees_protect` | 窗口门禁的实际单位 |
| `M_cuts_task` / `M_cuts_protect` | cut 数（参考） |
| `feasible_d0` | `d0 = Δ_adam` 是否已可行 |
| `frac_below_kappa` | `cos(q_j, Δ_adam) < -κ` 的行占比 |
| `cos_q01 / q05 / q10 / q50` | 约束方向 cosine 分位 |
| `n_active` | 激活约束行数 |
| `correction_ratio` | `||d* − Δ_adam|| / ||Δ_adam||` |
| `v_max` | certificate 违反量 |
| `cert_fail` | certificate 是否失败 |
| `gamma_param_count` | Γ 总参数量（期望 108 288） |
| `omega_replay_count` | **严格为 0** |
| `proj_noop_kept_adamw_write` | `n_active == 0` 分支是否命中 |
| `oas_intensity_z` / `oas_intensity_p` | OAS 收缩强度，**v3 新增监控** |

**`feasible_d0` 率是一等监控量。**

**v3 新增风险监控（必须记录，不得静默）**：
`B_V_DIM = 32`，而窗口的**有效独立样本数**上限由 `N_TREES = 32` 决定。
窗口内行数 = cut 数（可多于 tree 数：一条 lineage 的每个深度都是一个 cut），
但**同一 tree 的多个 cut 高度相关**——Welford 按 tree 聚类正是为此，
门禁也按 unique tree 计数而非行数。
当有效样本数 ≈ 维数（32 vs 32）时，样本协方差接近秩亏缺，
OAS 收缩会承担主要工作，`J_CCA` 的估计方差显著偏大。
`oas_intensity_z` 接近 1 表示几乎退化为对角近似，此时 task direction 不可信——
必须如实记录，不得在论文中把该情形当作有效估计。
（这与决定一中否决 36 维 `R_v` 的理由同源：那里是 `R ∈ R^{32×36}` 秩上界 31，
这里是 `B` 的有效样本数 ≈ 维数。）

---

## 8. 验收测试

| # | 测试 | 可执行判据 |
|---|---|---|
| 1 | Γ 参数 < 1M | `sum(p.numel() for p in Γ.parameters()) == 108_288`，启动 assert |
| 2 | `C_v` 不变 | `predictive` 模式下 `C_v` 各字段与 `official` 逐字段一致，且 reducer 未追加 |
| 3 | **保留的** code block 完整 | 所有进入 `selected_stack` 的 InjectedCode 完整；**允许整块丢弃** |
| 4 | lineage 对齐 | `Y_{v+1}`/`Y_{v+2}` 与 archive 真实 child/grandchild 逐字段对齐 |
| 5 | 缺 grandchild 排除 | 构造缺失用例，断言该 cut 不在窗口内 |
| 6 | OAS 启用 | 断言走 `oas=True`；与 `oas=False` 数值不同 |
| 7 | 尺度不变性 | `⟨∇_B J, B⟩ ≈ 0`，防错误 detach `C_ZZ` |
| 8 | tree 不重叠 | `set(task_tree_ids).isdisjoint(protect_tree_ids)`，且无 tree 跨窗口 |
| 9 | virtual AdamW 一致 | 无约束时 `counterfactual_adamw` 结果 == 真实 AdamW step（bitwise） |
| 10 | `feasible_d0` 提交 | `d* = Δ_adam`，提交结果 == 未投影 proposal |
| 11 | cert failure 全冻结 | 参数、moments、step counter、scheduler 全不变 |
| 12 | 只提交一次 | 成功路径 `optimizer.step()` 调用次数 == 1 |
| 13 | Ω 零重放 | Phase B 全程 Ω 调用次数严格为 0 |
| 14 | 三模公平性 | `same task/cohort/model/temperature/round/seed/evaluator`；`official_budget == predictive_budget`；**full 使用未裁剪上下文，仅受模型硬上限约束** |
| 15 | reduce 契约 | `PredictiveSelector.reduce` 返回原始 `Trace`/`InjectedCode` 对象，`_build_prompt` 未改动 |
| 16 | 两处注入点 | `generate` 与 `refine` 两条路径都被同一模开关覆盖 |
| 17 | Phase C 纯净 | 评估 run 中 Γ 无更新、未读 future、未构造 `R_v` |
| 18 | **tree_id 不跨 run 碰撞** | `tree_id = (run_id, root_candidate_id)`；断言 64 个 run 产生 **64 个互异 tree_id**，不是 1 个 |
| 19 | **32/32 切分固定** | 切分由配置里的固定 hash 决定，重复运行结果一致；两组互斥 |
| 20 | **Phase A 用 Official** | 断言轨迹收集阶段走 `official` reduction |

---

## 9. v2 → v3 变更对照

| # | v2 | v3 | 依据 |
|---|---|---|---|
| 1 | `N_TREE_MIN=128`，首次 256 | **32**，首次 64 | census：1 root/run |
| 2 | `M_SKETCH=32` | **8** | 决定一（A′） |
| 3 | `B_V_DIM=128` | **32 = 4m** | 决定一 |
| 4 | 未限定 run 数 | **`N_RUNS_TOTAL=64`**（32 task + 32 protect） | 决定一 |
| 5 | Phase A 未定量 | **64 次独立 Official-reduction run** | 决定一 |
| 6 | `tree_id` 为字符串 | **`(run_id, root_candidate_id)`** | 防跨 run 碰撞 |
| 7 | `Cond` 全秩 `[1024,256]` | **rank-32 低秩** `B_c A_c` | Γ 容量收缩 |
| 8 | 未定 32/32 切分 | 固定 hash 切分，永不交换/重叠 | 决定一 |
| 9 | `E_0` 待签字 + 允许双编码器 | **定版 CodeBERT**，删除双编码器 | 决定二 |
| 10 | `max_length=256` 与"128-token 窗口"并存 | 统一为 **256-token 非重叠分块** | 消除矛盾 |
| 11 | Γ 预算 329 472 | **108 288** | §1.2 |
| 12 | 未提降级路径 | 新增 16+16 / m=4 / 16 的 **pilot** 路径 | 决定一 |
| 13 | 未提单次更新后果 | §4.2 明示 **64 run 只够一次 boundary** | 决定一 |
| 14 | 无 B 秩亏监控 | §7 新增 `oas_intensity_*` + B 秩亏风险条目 | §1 收缩的直接后果 |

---

## 10. 不变量

- `C_v` 永不进入 `U_v`，永不被 Γ 改写，**也永不进入 reducer 的输出**。
- InjectedCode **整块保留或整块裁剪**，禁止中间截断。
- 未来信息**只**进入 `p_v`，全程 `detach`。
- child/grandchild 文字**不得**进入 `Γ(U_v)`。
- 每 cut 的 `q_j` 分别保存，**绝不提前相加**。
- 有且仅有一次 `optimizer.step()`，只在 certificate 通过后。
- `C_ZZ` 保留梯度；只有 OAS intensity 可 detach。
- 四条 branch 独立，禁止拼成单一 covariance。
- 同一 tree 的所有 cut 在同一窗口；task/protect 的 tree 集合互斥。
- **`tree_id` 必须含 `run_id`**；任何形式的全局 `"gen0_seed"` 直用都是 bug。
- Phase A 用 Official reduction；Phase B 全程 Ω 调用为 0；Phase C 全程 Γ 更新为 0。
