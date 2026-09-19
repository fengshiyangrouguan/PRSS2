# Metaⁿ × RPBE 工程任务书 v1（冻结）

状态：方法已冻结，本文件只规定**类、张量形状、缓存结构、训练伪代码、验收判据**。
实现者不得改变方法，只能在本文规定的自由度内选择代码组织方式。

对应冻结规格：`Metaⁿ × RPBE 核心设计 v1`（用户 2026-09-17 冻结）。
基线代码：`meta-n-main`，blob `73e0d5097878d759cac0cdc32c67abe756d2b264`。

---

## 0. 边界（不可更改）

1. 只修改 Metaⁿ 每次 `OmegaEngine.generate` 前的 context reduction。
   注入点在 `meta_n/core/omega.py:107-108`：
   ```python
   sampled = self.context_manager.sample_traces(traces)
   truncated_stack = self.context_manager.truncate_context_stack(context_stack)
   ```
   这两行是**唯一**允许被 `predictive` 模式替换的地方。
2. 保留三种模式：`full`、`official`、`predictive`。
3. 不改 `archive`、`OmegaEngine.generate` 本体、`MetaLayer`、evaluator、现有 feasibility solver 内核。
4. frozen encoder、frozen LLM，只训练 Γ。
5. token budget 与 Official Metaⁿ 严格对齐。
6. 当前 4080 SUPER 只跑 Metaⁿ + Ours；Official 与 Full Context 在另一张卡单独跑。
7. finite-test 推理只执行一次 context reduction，**不运行两步 rollout**；两步 future 只用于离线训练监督。

---

## 1. 冻结常数

| 名称 | 值 | 来源 / 说明 |
|---|---|---|
| `D_E` | 256 | item embedding / fusion 维度。**禁止用 LLM hidden 3584 直接做 fusion dim** |
| `H_LORA` | 64 | `W_K`、`W_V` 的低秩瓶颈 |
| `N_SLOTS` | 4 | 四槽 |
| `B_V_DIM` | 128 | `b_v = P_b · vec(Z_v)` 的目标维度 |
| `M_SKETCH` | 32 | 每条 LPSE branch 的 P 维 |
| `N_BRANCHES` | 4 | 独立 branch 数，**禁止拼成 128 维单一 covariance** |
| `M_MIN` | 128 | `= max(128, 4·m)`，m=32 |
| `KAPPA` | 0.02 | 约束软容忍带（Stage8 冻结的几何标定值，**不得按结果调**） |
| `TAU` | 3e-4 | certificate 容差。只影响 gate，不改变可行集 |
| `GAMMA_BUDGET` | 1_000_000 | Γ 总参数量硬上限，启动时 assert |
| `Z_V_DIM` | 128 | 窗口 z 维（沿用 CCM `frozen_method.json`） |
| `P_B_SEED` | 0 | `P_b`、`P_Z^(r)`、future sketch 的固定投影种子 |

Γ 参数量预算（`d_e=256, h=64`）：

| 参数 | 形状 | 数量 |
|---|---|---|
| `A_K` | `[256, 64]` | 16 384 |
| `B_K` | `[64, 256]` | 16 384 |
| `A_V` | `[256, 64]` | 16 384 |
| `B_V` | `[64, 256]` | 16 384 |
| `w_g` | `[256]` | 256 |
| `Q_0` | `[4, 256]` | 1 024 |
| `Cond` | `[256, 1024]` | 262 144 |
| `LN` weight/bias | `[256] ×2` | 512 |
| **合计** | | **≈ 329 472** |

若 `Cond` 需要降参，允许低秩分解，但**总预算断言不变**。

---

## 2. 模块与类清单

全部新增于 `meta_n/rpbe/`（新包，不改动 meta-n 原有模块）：

```
meta_n/rpbe/__init__.py
meta_n/rpbe/config.py             # 冻结常数 + assert_gamma_budget()
meta_n/rpbe/encoder.py            # FrozenItemEncoder:  U_v -> X_v
meta_n/rpbe/fusion.py             # SlottedFusion (Γ_θ):  X_v, Q_v -> Z_v, A_v
meta_n/rpbe/renderer.py           # DeterministicRenderer: U_v, A_v -> prompt text
meta_n/rpbe/records.py            # CutMeta / CutRecord
meta_n/rpbe/lineage.py            # 两步真实 lineage 抽取
meta_n/rpbe/future.py             # phi_r(S_v) 固定 sketch
meta_n/rpbe/window.py             # StatWindow: B/P/R 累积 + OAS CCA
meta_n/rpbe/qp.py                 # proposal-space QP（移植，见 §8）
meta_n/rpbe/context_reduction.py  # full | official | predictive 三模适配器
meta_n/rpbe/trainer.py            # 边界训练循环
```

### 2.1 `encoder.py`

```python
class FrozenItemEncoder:
    def __init__(self, text_encoder, d_e: int = 256, max_chunk: int = 128): ...
    @torch.no_grad()
    def encode(self, items: list[TextItem]) -> torch.Tensor:
        """-> X_v [n_v, 256]，detach，无梯度。"""
```

- 长 item 先按固定长度切块，再做 **masked mean pooling**。
- 加入固定的 item-type / source-depth encoding（`t_type(j) + r_depth(j)`），二者都是**固定 buffer**，不可训练。
- 输出必须 `detach()`。**训练时不对文本编码器反传。**

### 2.2 `fusion.py`

```python
class SlottedFusion(nn.Module):
    def __init__(self, d_e=256, h=64, n_slots=4, cond_dim=1024): ...
    def forward(self, X_v, Q_v, compat_mask) -> tuple[Tensor, Tensor]:
        """-> Z_v [4, 256], A_v [4, n_v]"""
```

严格按冻结公式，**不得增删项**：

```
K = X_v W_K,  V = X_v W_V,  g = sigmoid(X_v w_g)
A_v = softmax_j( Q_v K^T / sqrt(d_e) + 1_4 log(g + eps)^T + M_v )
Z_v = LN( Q_v + A_v V )
```

- `W_K = A_K B_K`，`W_V = A_V B_V`，`h = 64`。**禁止 full-rank `d_e × d_e`。**
- `Q_v = Q_0 + Cond(E(q_v))`，[4, 256]。
- `M_v` 是**固定的 slot / item-type 兼容 mask**：非法组合 `-inf`，合法组合为固定 bias。它是 buffer，不是参数。
- 四个 slot 语义**不预设**，让语义自行形成。

### 2.3 `renderer.py`

```python
class DeterministicRenderer:
    def render(self, U_v, A_v, C_v) -> str:
        """确定性；无 decoder、无第二 LLM、无新生成文字。"""
```

对每个 slot k：

1. `j_k = argmax_j A_v[k, j]`；
2. 若多个 slot 选中同一 item，后面的 slot 取**下一个最高权重的未使用 item**；
3. 计算该 slot 的加权结构统计：weighted mean score、success/failure weight、failure-class distribution、source-depth distribution；
4. 将选中的原始 trace / code **按 Metaⁿ 原生 context 格式**渲染；
5. **code block 保持完整，禁止从中间截断**；
6. 末尾追加**未经压缩的 `C_v`**。

输出形如：

```
## Predictive Slot 1
Weighted evidence:
- mean score: ...
- failure mass: ...
- dominant failure: dependency_error

Representative trace:
<官方原生 trace 格式>

## Predictive Slot 2
...
```

**训练用连续 `Z_v`；部署用 `A_v` 抽取原文。** 允许逐步降低 softmax temperature 以缩小 soft/hard 差距（temperature 是 `predictive` 模式的可选参数，默认 1.0）。

### 2.4 `records.py`

```python
@dataclass
class CutMeta:
    candidate_id: str          # c_v 在 archive 中的 id
    depth: int                 # d
    n_items: int               # n_v
    parent_id: str | None

@dataclass
class CutRecord:
    cut_id: tuple              # (candidate_id, occurrence_seq)
    tree_id: str               # lineage root id（窗口唯一树 gate 用）
    occurrence_seq: int
    z: torch.Tensor            # [4, 256]，保留 graph
    a: torch.Tensor            # [4, n_v]，detach
    p: torch.Tensor            # [m] per branch，detach
    r: float                   # 两步真实收益
    weight: float              # parent 归一化权重
    X_v: torch.Tensor          # [n_v, 256] 冻结
    Q_v: torch.Tensor          # [4, 256] 冻结
    compat_mask: torch.Tensor  # [4, n_v] 固定
```

### 2.5 `lineage.py`

- 只接受 archive 中**真实存在**的 `c_v → c_{v+1} → c_{v+2}`。
- `Y_{v+1} = Φ(T_{d+1})`，`Y_{v+2} = Φ(T_{d+2})`：future 是 **child/grandchild 的真实 trace observation，不是 InjectedCode**。
- 缺真实 grandchild 的 cut **不进入 LPSE**。
- 每条真实 parent→child→grandchild 路径是一个 occurrence；**按 parent 归一化权重**，避免多 child 的 parent 被重复放大。

### 2.6 `future.py`

```
f_v^(1) = E_0(serialize(Y_{v+1}))
f_v^(2) = E_0(serialize(Y_{v+2}))
phi(S_v) = R_S [ f^(1) ; f^(2) ; f^(2) - f^(1) ; f^(1) ⊙ f^(2) ]
```

- `R_S` 是**固定随机投影，不训练**。
- 每条 branch r 有自己的 `R_S^(r)`，输出 32 维。
- `phi` 全程 `detach`。**child/grandchild 文字不得进入 Γ(U_v)。**

### 2.7 `window.py`

```python
class StatWindow:
    def __init__(self, m: int = 32, z_dim: int = 128, n_branches: int = 4,
                 seed: int = 0, oas: bool = True): ...
    def add(self, rows: list[CutRecord]) -> None: ...
    @property
    def ready(self) -> bool: ...        # eff >= M_MIN
    def close_lpse(self) -> ...          # per-cut cotangents，M 条，绝不提前相加
    def close_task(self) -> ...          # 聚合 cotangent
```

**OAS 必须显式启用。** 现有代码三处入口默认都是 `False`：
- `latent_z_adjoint(..., oas=False)` — `src/rpbe/loss.py:301`
- `KFMomentWindow(..., oas=False)` — `src/rpbe/loss.py:914`
- `kf_score()` / `kf_adjoint()` **完全没有 OAS 入口**

正确路径：

```python
latent_z_adjoint(..., oas=True)
# 或 KFMomentWindow(..., oas=True)（内部在 loss.py:1290 转发 oas=self.oas）
```

**不得直接调用 `kf_score` + `kf_adjoint` 来"复用 OAS"——它们会静默丢掉 OAS。**
若需要等价的新入口，必须先通过 parity test（与 `latent_z_adjoint(oas=True)` 数值一致）。

梯度纪律：

- **只有 OAS intensity 可以 detach**（window 统计量，不是优化变量）。
- `C_PP`、`C_RR` 侧固定。
- `C_ZZ`、scale normalization、`B` **必须保留梯度**。
- **严禁 stop-gradient `C_ZZ`**：尺度不变性 `⟨∇_B J, B⟩ ≈ 0` 依赖 `C_ZZ` 带梯度；stop-grad 会得到"径向导数 2J"的半梯度，不是任何目标的梯度。

两个目标：

```
J_LPSE = (1/4) Σ_r J_CCA(B, P^(r))     # P^(r) [M, 32]
J_task = J_CCA(B, R)                    # R [M, 1]
```

两者都使用 **paired OAS shrinkage + scale-normalized Cholesky**，**不得使用 plain ridge inverse**。

---

## 3. 张量形状表

| 符号 | 形状 | 梯度 | 说明 |
|---|---|---|---|
| `n_v` | scalar | — | cut v 的 item 数 |
| `X_v` | `[n_v, 256]` | 无（frozen） | 冻结编码 + 固定 type/depth encoding |
| `Q_v` | `[4, 256]` | 经 `Q_0`/`Cond` | `Q_0 + Cond(E(q_v))` |
| `K` / `V` | `[n_v, 256]` | 经 `W_K`/`W_V` | 低秩 |
| `g` | `[n_v]` | 经 `w_g` | sigmoid |
| `M_v` | `[4, n_v]` | 固定 buffer | 兼容 mask |
| `A_v` | `[4, n_v]` | 是 | softmax |
| `Z_v` | `[4, 256]` | 是 | `LN(Q_v + A_v V)` |
| `vec(Z_v)` | `[1024]` | 是 | flatten |
| `b_v` | `[128]` | 是 | `P_b · vec(Z_v)`，`P_b` 固定 |
| `φ_r(S_v)` | `[32]` | 无（detach） | branch r |
| `B` | `[M, 128]` | 是 | 窗口内所有 cut |
| `P^(r)` | `[M, 32]` | 无 | 四条独立 branch |
| `R` | `[M, 1]` | 无 | `S̄(c_{v+2}) − S̄(c_v)` |
| `a_v^task` | `[128]` | — | `∂J_task/∂b_v` |
| `q_j` | `[len(Γ_params)]` | — | 每 cut 一条，**分别保存** |

---

## 4. 三层缓存

**核心约束：Metaⁿ 的 Ω 是 LLM 调用，不可重放。** CCM 的 pass2 是"重放前向"；Metaⁿ 的 pass2 只能 **replay Γ**。

### 4.1 第一层：cut cache（Metaⁿ run 存活期间）

按 cut 保存：

| 字段 | 形状 | 用途 |
|---|---|---|
| `X_v` | `[n_v, 256]` | Γ 重放输入 |
| `Q_v` | `[4, 256]` | Γ 重放输入 |
| `compat_mask` | `[4, n_v]` | Γ 重放输入 |
| `U_v` | 原始对象 | **仅在生成部署 prompt 时需要**；训练期可不留 |
| `A_v` | `[4, n_v]` | renderer 部署用（detach） |
| `phi_r(S_v)` | `[32] × 4` | detach |
| `R_v` | scalar | — |
| `lineage/occurrence id` | — | 权重归一化与去重 |

**禁止缓存后重新调用 Ω，禁止在 pass2 重跑原始文本生成。**

### 4.2 第二层：task window cache（`W_{t-1}`，跨边界存活）

保留 `M_task_eff ≥ 128` 行的 `X_v / Q_v / mask`，以便**在当前 Γ 参数下重放**。
**禁止复用旧参数下的 stale gradient。**

### 4.3 第三层：protect window cache（`W_t`）

保留 `M_protect_eff ≥ 128` 行的同类字段 + 每 cut 的 LPSE cotangent。

---

## 5. 训练伪代码（每个边界）

```
# ---------- 阶段 A：采样与收集（不更新参数） ----------
for 每个 Metaⁿ run:
    for 每个 depth d:
        c_v = 当前父候选
        U_v = (T_d，I_{2:d})              # InjectedCode 整块保留或整块裁剪
        C_v = (task, depth, scores, focus, environment, budget)
        X_v = FrozenItemEncoder.encode(U_v)          # detach
        Q_v = Q_0 + Cond(E(q_v))
        Z_v, A_v = SlotsedFusion(X_v, Q_v, M_v)      # 保留 graph
        b_v = P_b · vec(Z_v)
        记录 CutRecord(...)

# ---------- 阶段 B：lineage 配对 ----------
pairs = [ (c_v, c_{v+1}, c_{v+2}) ]                  # 只取真实存在两步的
丢弃缺 grandchild 的 cut
weight = normalize_by_parent(pairs)

# ---------- 阶段 C：窗口门禁 ----------
assert M_task_eff  >= M_MIN      # W_{t-1}
assert M_protect_eff >= M_MIN    # W_t
# 首次投影需累计约 256 条；稳态每新增 128 行关闭一次

# ---------- 阶段 D：task 方向（上一完整窗口） ----------
B_task = [b_v for v in W_{t-1}]
R_task = [R_v for v in W_{t-1}]
J_task = J_CCA(B_task, R_task, oas=True)
a_v^task = ∂J_task/∂b_v
L̃_task = -Σ_v <sg(a_v^task), b_v(θ)>
g_task = ∇_θ L̃_task = -∇_θ J_task

# ---------- 阶段 E：protect 约束（当前完整窗口） ----------
B_prot = [b_v for v in W_t]
for r in 0..3: J_LPSE += (1/4) J_CCA(B_prot, P^(r), oas=True)
per-cut cotangent a_v^LPSE = ∂J_LPSE/∂b_v        # 分别保存，禁止提前相加
对每个 cut 逐条 replay Γ:
    L̃_v = <sg(a_v^LPSE), b_v(θ)> - <sg(a_v^LPSE), sg(b_v(θ))>
    q_v = ∇_θ L̃_v                                 # 数值恒 0，梯度正确

# ---------- 阶段 F：proposal-space QP ----------
t = AdamWPropose(θ, g_task, s) - θ                # 包含真实 m/v/step/betas/eps/lr/wd
H_j = q_j / ||q_j||
d* = argmin_d 1/2||d - t||²  s.t.  H_j^T d >= -κ||t||  ∀j
v_max = max_j max(0, -κ||t|| - H_j^T d*) / (||t|| + ε)

if v_max <= τ:
    θ ← θ + d*                     # 全量覆盖，不是 +=
    提交虚拟 AdamW 的 moment / step state
    scheduler.step()                # 只在这里前进一步
    # 不得再调用第二次 optimizer.step()
else:
    参数 / moments / step counter / scheduler 全部不动
    整个 Gamma boundary update 跳过

# 若可行集空（无 j 的约束激活）→ feasible_d0 = True，d* = t
# 此时正常提交原始 AdamW proposal，只是没有投影修正
```

---

## 6. 必须记录的诊断（每个 boundary）

| 字段 | 含义 |
|---|---|
| `M_task_eff` | task 窗口有效 cut 数 |
| `M_protect_eff` | protect 窗口有效 cut 数 |
| `feasible_d0` | `d0 = t` 是否已可行 |
| `frac_below_kappa` | `cos(q_j, t) < -κ` 的行占比 |
| `cos_q01 / q05 / q10 / q50` | 约束方向与 t 的 cosine 分位 |
| `n_active` | 激活约束行数 |
| `correction_ratio` | `||d* - t|| / ||t||` |
| `v_max` | certificate 违反量 |
| `cert_fail` | certificate 是否失败 |
| `gamma_param_count` | Γ 总参数量 |
| `omega_replay_count` | 必须严格为 0 |

**`feasible_d0` 率是一等监控量。** 但不要声称"小 M 数学上必然导致 no-op"——小 M 的确定问题是 covariance / cotangent 不稳定。**只有两个 128 行门禁都满足后才允许投影。**

---

## 7. 验收测试（全部通过后才允许 smoke test）

| # | 测试 | 可执行判据 |
|---|---|---|
| 1 | Γ 参数 < 1M | `sum(p.numel() for p in Γ.parameters()) < 1_000_000`，且启动时 assert |
| 2 | `C_v` 不变 | `C_v` 文本级/bitwise 与 `official` 模式一致 |
| 3 | code block 不截断 | 所有 InjectedCode 在 renderer 输出中完整出现 |
| 4 | lineage 对齐 | `Y_{v+1}`/`Y_{v+2}` 与 archive 真实 child/grandchild 逐字段对齐 |
| 5 | 缺 grandchild 排除 | 构造缺失用例，断言该 cut 不在窗口内 |
| 6 | OAS 启用 | 断言实际走 `oas=True` 路径；与 `oas=False` 数值不同 |
| 7 | 尺度不变性 | `⟨∇_B J, B⟩ ≈ 0`（容忍 1e-6 相对量级），防止错误 detach `C_ZZ` |
| 8 | occurrence ID 不重叠 | `set(task_ids) ∩ set(protect_ids) == ∅` |
| 9 | virtual AdamW 一致 | 无约束时 `AdamWPropose` 结果 == 真实 AdamW step 结果（bitwise） |
| 10 | `feasible_d0` 提交 | `feasible_d0=True` 时提交结果 == 未投影 proposal |
| 11 | cert failure 全冻结 | 参数、moments、step counter、scheduler 全部不变 |
| 12 | 只提交一次 | 成功投影路径上 `optimizer.step()` 调用次数 == 1 |
| 13 | Ω 零重放 | pass2 期间 Ω 调用次数严格为 0 |
| 14 | 三模同输入 | `full` / `official` / `predictive` 使用相同输入、预算与 evaluator |

---

## 8. proposal-space QP 的移植（不要重写）

**proposal-space 实现已存在**，位于 `PRSS2` 仓库 `fix/avg-lora-clock` 分支：

- `src/rpbe_embodied/boundary.py` — `_proposal_space_update()`（第 155 行起）
- `src/rpbe_embodied/loss.py` — `active_set_feasibility_projection()`
- `results/STAGE8_RESULTS.md` — 冻结常数与证据

**注意**：`develop_CCM` 与 `develop_VLA` 上的确实是 raw-gradient QP；proposal-space 只在
`fix/avg-lora-clock`（Stage8 线）上。移植时不要误取 `develop_*`。

现有实现的语义与本文 §5 阶段 F **逐条一致**，可直接移植：

1. `_counterfactual_task_step()` — 在 **clone + 深拷贝 optimizer state** 上做一次真实 AdamW，
   真实参数与 m/v/step 不动，得到精确解析 proposal `Δ_adam`；
2. `active_set_feasibility_projection(..., t_override=Δ_adam, write_grad=False)` — 解同一个 QP；
3. `proj_feasible == False` → **在任何写操作之前 `return`**，因此无需回滚；
4. 通过后才 `optimizer.step()` **一次**（用同一个 clipped gradient 推进 moments）；
5. `Γ ← Γ_old + Δ*` — **全量覆盖，不是 `+=`**；
6. `proj_n_active == 0` 时**逐字节保留 AdamW 自己的写入**（不重新舍入）。

移植后必须复核的冻结值（来自 `STAGE8_RESULTS.md`）：
`kappa = 0.02`（冻结几何标定，从未按结果调）、`tau = 3e-4`（仅 certificate 容差）。
Stage8 实测 `v_max ~1e-8` 对 `tau = 3e-4`，四个数量级余量；`proj_shift_ratio` 均值 0.023、最大 0.143。

---

## 9. 实施顺序

1. `config.py` — 冻结常数 + `assert_gamma_budget()`；测试 #1 立即可跑
2. `encoder.py` + `records.py` — `U_v → X_v`，含 masked pooling 与固定 type/depth encoding
3. `fusion.py` — 四槽公式 + 低秩 `W_K/W_V` + 固定兼容 mask
4. `renderer.py` — 确定性抽取；测试 #2、#3
5. `lineage.py` + `future.py` — 两步配对与固定 sketch；测试 #4、#5
6. `window.py` — `StatWindow` + **确认 `oas=True` 路径**；测试 #6、#7
7. `qp.py` — 从 `fix/avg-lora-clock` 移植；测试 #9、#10、#11、#12
8. `context_reduction.py` — 三模适配器，接到 `omega.py:107-108`；测试 #13、#14
9. `trainer.py` — 边界循环 + 全诊断
10. smoke test → 启动当前卡上的 Metaⁿ + Ours

---

## 10. 不变量（任何时刻违反即为 bug）

- `C_v` 永不进入 `U_v`，永不被 Γ 改写。
- `U_v` 中的 InjectedCode **整块保留或整块裁剪**，禁止中间截断。
- 未来信息**只**进入 `p_v`，且全程 `detach`。
- child/grandchild 文字**不得**进入 `Γ(U_v)`。
- 每 cut 的 `q_j` **分别保存**，绝不提前相加。
- 有且仅有一次 `optimizer.step()`，且只在 certificate 通过后。
- `C_ZZ` 保留梯度；只有 OAS intensity 可 detach。
- 四条 branch 独立，禁止拼成单一 128 维 covariance。
