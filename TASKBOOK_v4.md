# Metaⁿ × RPBE 工程任务书 v4

状态：**方法已冻结，待开工**。本文件只规定**契约、类、张量形状、缓存结构、训练生命周期、验收判据**。
实现者不得改变方法。

- v1/v2/v3 已作废，以本文件为准。
- 基线代码：`meta-n-main`，blob `73e0d5097878d759cac0cdc32c67abe756d2b264`。
- proposal-space 来源：`PRSS2` 仓库 `fix/avg-lora-clock`，冻结提交 `c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36`。
- v4 相对 v3 只有两类改动：**(a)** 修正"64 roots 只能更新一次"的方法级错误；
  **(b)** 修正 OAS 诊断的文字与字段。变更对照见 §9。

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

## 1. 冻结常数

| 名称 | v4 冻结值 | 说明 |
|---|---|---|
| `N_RUNS_TOTAL` | **64** | Phase A 独立 Official-reduction Ω run 总数 |
| `DATA_WINDOW_PAIRS` | **1** | 只建立**一对**互斥的 task/protect 窗口 |
| `TRAIN_STEPS` | **300** | Phase B 在这对固定窗口上的离线优化步数 |
| `N_TREES_TASK` | **32** | task 窗口 tree 数 |
| `N_TREES_PROTECT` | **32** | protect 窗口 tree 数 |
| `N_TREE_MIN` | **32** | 每窗口门禁单位 = unique tree |
| `M_SKETCH` | **8** | 每条 LPSE branch 的 P 维 |
| `B_V_DIM` | **32** | `b_v` 维度。**正式配置；16 只作 ablation，不得替换主配置** |
| `N_BRANCHES` | 4 | 仍然独立，禁止拼成单一 covariance |
| `D_E` | 256 | fusion 维度。**禁止用 LLM hidden 做 fusion dim** |
| `H_LORA` | 64 | `W_K`/`W_V` 低秩瓶颈 |
| `N_SLOTS` | 4 | 四槽 |
| `COND_RANK` | 32 | `Cond` 低秩 |
| `KAPPA` | 0.02 | Stage8 冻结几何标定，**不得按结果调** |
| `TAU` | 3e-4 | certificate 容差，只影响 gate，不改变可行集 |
| `GAMMA_BUDGET` | 1_000_000 | Γ 参数硬上限，启动 assert |
| `TEMPERATURE` | 1.0 | **固定** |
| `P_B_SEED` | 0 | `P_b`、`R_S^(r)`、`P_rc` 的固定投影种子 |

**`B_V_DIM = 32` 的依据**：参考配置 `(N_tree, d_b, m) = (128, 128, 32)` 等比例缩小四倍
→ `(32, 32, 8)`，三者同步，`d_b : m = 4 : 1` 的结构比例不变。
`B_V_DIM = 16` 会把 `d_b : m` 改成 `2 : 1`，属于**新的方法变体**，
只能作为 ablation / pilot，**不应替换主配置**。

（若 Phase A 实际只能收集 32 个 root，则整份配置降级标 `pilot`，不得与正式结果并排。）

### 1.1 冻结文本编码器 `E_0`

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
5. seed 0 的**固定 Rademacher 投影**降到 256：

   ```
   P_rc ∈ R^{768×256},  P_rc[i,j] ∈ { -1/sqrt(256), +1/sqrt(256) }
   ```

**不使用 text/code 双编码器。**

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

`Cond(q) = B_c A_c q`。

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

**只输出 `X_v` 和 `q_emb`；不输出 `Q_v`。** `Q_v` 每一步在当前 Γ 下重算。

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
4. **不得追加 `C_v`**；
5. 加权统计量**只能写进 `diagnostics`**；
6. 多 slot 选中同一 item 时，后面的 slot 取下一个最高权重的未使用 item；
7. **被保留的 code block 必须完整；允许整块丢弃，禁止中间截断。**

### 2.4 `records.py`

```python
@dataclass
class CutRecord:
    cut_id: tuple                # (run_id, root_candidate_id, occurrence_seq)
    tree_id: tuple               # (run_id, root_candidate_id)
    occurrence_seq: int
    X_v: torch.Tensor            # [n_v, 256]  frozen，缓存
    q_emb: torch.Tensor          # [256]       frozen，缓存
    mask: torch.Tensor           # [4, n_v]    固定
    p: torch.Tensor              # [8] per branch，detach
    r: float                     # 两步真实收益（标量）
    weight: float                # parent 归一化权重
```

**`tree_id` 必须是 `(run_id, root_candidate_id)` 元组。**
`gen0_seed` 是硬编码候选 id，跨 run 完全相同；只用它做 tree 身份会让
**64 个独立 run 全部碰撞成同一棵树**。

**没有 `z` 字段。没有 graph。没有 `Q_v`。没有 `A_v`。**

### 2.5 `lineage.py`

- 只接受 archive 中**真实存在**的 `c_v → c_{v+1} → c_{v+2}`；
- `Y_{v+1} = Φ(T_{d+1})`、`Y_{v+2} = Φ(T_{d+2})`：future 是 **child/grandchild 的真实 trace observation，不是 InjectedCode**；
- 缺真实 grandchild 的 cut **不进入 LPSE**，**不得**补零 / 复制 child / 造假 EOS；
- 每条真实 parent→child→grandchild 是一个 occurrence；**按 parent 归一化权重**。

### 2.6 `future.py`

```
f_v^(1) = E_0(serialize(Y_{v+1}))                    # [256]
f_v^(2) = E_0(serialize(Y_{v+2}))                    # [256]
phi_r(S_v) = R_S^(r) [ f_v^(1) ; f_v^(2) ]           # [8]，R_S^(r) ∈ R^{8×512}
```

- `R_S^(r)` 固定随机投影，不训练，每 branch 独立 seed。
- **不得加入 `f^(2) - f^(1)` 或 `f^(1) ⊙ f^(2)`。**
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
- **严禁 stop-gradient `C_ZZ`**（验收 #7 专抓）。

```
J_LPSE = (1/4) Σ_r J_CCA(B, P^(r))
J_task = J_CCA(B, R)
```

两者都用 **paired OAS shrinkage + scale-normalized Cholesky**，**禁用 plain ridge inverse**。

---

## 3. 张量形状表

| 符号 | 形状 | 梯度 | 缓存? |
|---|---|---|---|
| `X_v` | `[n_v, 256]` | 无 | **是** |
| `q_emb` | `[256]` | 无 | **是** |
| `mask` | `[4, n_v]` | 固定 | **是** |
| `Q_v` | `[4, 256]` | 是（经 `Q_0`/`A_c`/`B_c`） | **否，每步重算** |
| `K` / `V` | `[n_v, 256]` | 经 `W_K`/`W_V` | 否 |
| `g` | `[n_v]` | 经 `w_g` | 否 |
| `A_v` | `[4, n_v]` | 是 | **否** |
| `Z_v` | `[4, 256]` | 是 | 否 |
| `vec(Z_v)` | `[1024]` | 是 | 否 |
| `b_v` | `[32]` | 是 | 否（每步重算） |
| `φ_r(S_v)` | `[8]` | 无 | **是** |
| `B` | `[M, 32]` | 是 | 否（每步重算） |
| `P^(r)` | `[M, 8]` | 无 | **是** |
| `R` | `[M, 1]` | 无 | **是** |
| `q_j` | `[len(Γ_params)]` | — | **否，每步重算** |

`P_b ∈ R^{32×1024}` 固定投影（seed 0），不训练。

---

## 4. 缓存与窗口单位

### 4.1 三层缓存

**核心约束：Metaⁿ 的 Ω 是 LLM 调用，不可重放。** pass2 只能 **replay Γ**。

| 层 | 内容 | 生命周期 |
|---|---|---|
| L1 cut cache | `X_v / q_emb / mask / φ_r(S_v) / R_v / IDs` | Phase A 收集期，之后冻结 |
| L2 task window | 同上 + `B_task` + `R_task` | Phase B 全程固定 |
| L3 protect window | 同上 + `P^(r)` | Phase B 全程固定 |

**禁止缓存** graph-connected `z`、`Q_v`、`A_v`、旧 cotangent。
**禁止**缓存后重新调用 Ω，**禁止**在 pass2 重跑原始文本生成。

### 4.2 窗口是数据集，不是消耗品

`KFMomentWindow.window_ready()` 判据（`PRSS2:src/rpbe/loss.py:1340-1346`）：

```python
len(self._windows[tau]["tree_seen"]) >= self._threshold(tau)
```

故 v4 门禁：

```
N_unique_trees^task    >= 32
N_unique_trees^protect >= 32
```

硬性要求：

- **同一 tree 的所有 cut 必须进入同一窗口，整棵 tree 不得跨窗口**；
- `set(task_tree_ids).isdisjoint(protect_tree_ids)`；
- 64 个 root 用**固定 hash** 切分成 32 task / 32 protect，切分固化进配置，不得按结果重切。

```python
task_roots    = fixed_hash_split[:32]
protect_roots = fixed_hash_split[32:64]
assert set(task_roots).isdisjoint(protect_roots)
```

**`DATA_WINDOW_PAIRS = 1` 的含义是：64 个 root 建立一对互斥窗口后，
这对窗口在 Phase B 全程固定不变，被反复 replay 至多 `TRAIN_STEPS = 300` 步。**
门禁回答的是"数据是否够用"，**不是"能用几次"**——Phase B 不调用 Ω，
窗口是冻结数据集，不是一次性消耗品。

**tree 的角色永久固定**：一旦进入 task 窗口，永不改判为 protect，反之亦然。
300 步中可反复 replay，但角色不变。

### 4.3 archive census（开工前必跑）

**在写任何训练代码之前**先跑 `census.py`，输出：

1. eligible lineage 数；
2. unique parent 数；
3. **unique tree 数（按 `(run_id, root_candidate_id)` 计）**；
4. 按 `N_TREE_MIN=32` 能组成的完整窗口对数。

**若 unique tree < 64，禁止假装门禁已满足、禁止重复计数同一 tree 充数**，降级标 `pilot`。

---

## 5. 训练生命周期（必须离线）

```
Phase A：用 Official reduction 做 64 次独立 Ω run，收集并冻结完整 archive trajectories（只读，不训 Γ）
Phase B：Ω 调用严格为 0；在同一对固定互斥窗口上做 300 个离线 Γ 优化步
Phase C：冻结 Γ，在新的正式评估 run 中执行 Predictive reduction
```

**Phase A 必须使用 Official reduction**（不是 predictive，也不是 full）收集轨迹。

**Phase C 硬性约束：** 不读取未来 child/grandchild；不构造 `R_v`；**不更新 Γ**；只做当前 cut 的一次 reduction。

### 5.1 Phase B 伪代码（300 步）

```python
task_roots    = fixed_hash_split[:32]
protect_roots = fixed_hash_split[32:64]
assert set(task_roots).isdisjoint(protect_roots)

for optimizer_step in range(TRAIN_STEPS):          # 300
    # pass 1：在当前 Γ 下重算统计量
    B_task    = replay_gamma_no_grad(task_cache, theta)
    B_protect = replay_gamma_no_grad(protect_cache, theta)

    # 每一步重新计算 OAS、score 和 cotangent（不得复用旧值）
    a_task    = task_oas_adjoint(B_task, R_task)
    a_protect = lpse_oas_adjoint(B_protect, P_protect)

    # pass 2：重新 replay 当前 Γ，构造真实参数梯度
    g_task = replay_task_surrogate(task_cache, a_task, theta)
    q_rows = replay_protect_surrogates(protect_cache, a_protect, theta)

    delta_adam = counterfactual_adamw(theta, g_task)
    d_star     = proposal_space_qp(delta_adam, q_rows)

    certificate_then_commit_once(d_star)
```

**每一步必须重算 `B / OAS / cotangent / g_task / q_rows`。**
只能复用**冻结的编码输入与未来标签**（`X_v / q_emb / mask / P / R / IDs`），
**不能复用旧梯度、旧 cotangent、旧 OAS 强度。**

**Phase B 全程 Ω 调用严格为 0。**

---

## 6. proposal-space QP（精确写法）

来源：`fix/avg-lora-clock@c05e4fb`，`src/rpbe_embodied/boundary.py::_proposal_space_update`。
**移植而非重写。**

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

冻结值：`kappa=0.02`、`tau=3e-4`。Stage8 实测 `v_max ~1e-8` 对 `tau=3e-4`；`proj_shift_ratio` 均值 0.023、最大 0.143。

**写法禁忌**：

- 不得写 `θ ← θ + d*，全量覆盖，不是 +=`。真实操作是 `copy_(theta_old + d_star)`。
- 不得写"提交虚拟 moments"。真实做法是让**真实 optimizer** 用同一个 clipped gradient `step()` 一次，
  然后在有投影时覆盖参数值。
- 有且仅有一次 `optimizer.step()`，且只在 certificate 通过后。

---

## 7. 诊断字段

| 字段 | 含义 |
|---|---|
| `N_unique_trees_task` / `N_unique_trees_protect` | 窗口门禁的实际单位 |
| `M_cuts_task` / `M_cuts_protect` | cut 行数（矩阵秩相关，非独立样本量） |
| `D_task` / `D_protect` | Welford 的 cluster-aware 有效自由度 `n` |
| `feasible_d0` | `d0 = Δ_adam` 是否已可行 |
| `frac_below_kappa` | `cos(q_j, Δ_adam) < -κ` 的行占比 |
| `cos_q01 / q05 / q10 / q50` | 约束方向 cosine 分位 |
| `n_active` | 激活约束行数 |
| `correction_ratio` | `\|\|d* − Δ_adam\|\| / \|\|Δ_adam\|\|` |
| `v_max` / `cert_fail` | certificate 违反量与是否失败 |
| `gamma_param_count` | 期望 108 288 |
| `omega_replay_count` | Phase B **严格为 0** |
| `proj_noop_kept_adamw_write` | `n_active == 0` 分支是否命中 |
| `oas_alpha_z_task` | task 窗口 Z 侧收缩强度 |
| `oas_alpha_p_task` | **R 为标量时恒为 0；非零基本表示实现错误** |
| `oas_alpha_z_protect` | protect 窗口 Z 侧收缩强度 |
| `oas_alpha_p_protect[4]` | 四条 branch 各一 |
| `j_cca_task_on_task` / `j_cca_task_on_protect` | 300 步过拟合监控（见下） |

### 7.1 OAS 的正确表述

`C'_ZZ = (1-a_Z) C_ZZ + a_Z μ_Z I`，收缩目标是**缩放单位阵 `μ_Z I`**
（`μ_Z = tr(C_ZZ)/d`，`loss.py:84-113` 的 docstring 明确写 μ·I 而非 plain I，
因为 plain identity 破坏尺度不变性），**不是"对角近似"**。

同时 `C'_ZP = sqrt((1-a_Z)(1-a_p)) C_ZP` 被强烈压小。

`a_Z → 1` 表示**当前窗口信号被高度收缩**，是诊断量。
**不得仅凭它接近 1 就事后宣布结果无效。**

**标量 `R` 的必然结果**（由 `_oas_alpha` 公式直接得出，可作断言）：

- `p = 1` ⇒ `beta = (1-2)tr(C²)+tr(C)² = -c²+c² = 0`，
  `delta = (n-1)(c²-c²) = 0` ⇒ **`alpha = 0`**；
- 故 `C'_PP = C_PP` 原样不变，`C'_ZP = sqrt(1-a_Z) C_ZP`。

### 7.2 有效样本量 vs 矩阵秩（v3 此处写错过）

`WeightedWelford` 的

```
M2_zz = Σ_i w_i (z_i - z̄)(z_i - z̄)^T
```

是**按全部 cut 行**累加的（`loss.py:505-517`）。`tree_id` 只经
`wsum → W2_b → W2_cut → D` 起作用，`D = W - W2_cut/W` 是传给 `_oas_alpha` 的
cluster-aware 有效自由度 `n`。

因此必须区分两件事：

- **有效独立样本量 ≈ N_TREES = 32**，由 `D` 体现；
- **矩阵代数秩由 cut 行数 `M_cuts` 决定**，不能断言秩上界是 31。

小样本风险确实存在（有效样本 32 对 `B_V_DIM=32`），但这正是 OAS 的用途，
**不构成把 `B_V_DIM` 降到 16 的充分理由**（16 会改 `d_b:m` 比例，属新方法变体）。

### 7.3 300 步过拟合监控

Phase B 在同一对固定窗口上跑 300 步，数据量固定。
task 与 protect 是**互斥**的两组 tree，因此可用一个干净的泛化监控：

- `j_cca_task_on_task`：`J_CCA(B_task, R_task)`，即 task 目标本身；
- `j_cca_task_on_protect`：同一 `J_CCA` 泛函在 protect 窗口上评估。

若前者持续上升而后者停滞或下降，说明在过拟合 32 个 task tree。
**这两个量是诊断，不是 gate**（gate 仍只有 certificate）。

---

## 8. 验收测试

| # | 测试 | 可执行判据 |
|---|---|---|
| 1 | Γ 参数 | `sum(p.numel() for p in Γ.parameters()) == 108_288`，启动 assert |
| 2 | `C_v` 不变 | `predictive` 下 `C_v` 各字段与 `official` 逐字段一致，且 reducer 未追加 |
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
| 15 | reduce 契约 | 返回原始 `Trace`/`InjectedCode` 对象，`_build_prompt` 未改动 |
| 16 | 两处注入点 | `generate` 与 `refine` 两条路径都被同一模开关覆盖 |
| 17 | Phase C 纯净 | 评估 run 中 Γ 无更新、未读 future、未构造 `R_v` |
| 18 | tree_id 不跨 run 碰撞 | `(run_id, root_candidate_id)`；64 个 run 产生 **64 个互异 tree_id** |
| 19 | 32/32 切分固定 | 固定 hash 决定，重复运行一致；两组互斥 |
| 20 | Phase A 用 Official | 轨迹收集阶段走 `official` reduction |
| 21 | **300 步复用同一窗口** | 断言 Phase B 的 `task_cache` / `protect_cache` 在所有 300 步中**逐字节不变**（只有 Γ 变） |
| 22 | **每步重算** | 断言第 t 步的 `a_task`/`a_protect`/`g_task`/`q_rows` 不等于第 t−1 步（除非恰在不动点） |
| 23 | **标量 R 的 OAS** | `oas_alpha_p_task == 0.0`（浮点严格），否则断言失败 |
| 24 | tree 角色不变 | 300 步前后 `task_tree_ids` 与 `protect_tree_ids` 集合完全一致 |

---

## 9. v3 → v4 变更对照

| # | v3 | v4 | 依据 |
|---|---|---|---|
| 1 | "64 run 只够一次 boundary 更新" | **删除**。改为 `DATA_WINDOW_PAIRS=1` + `TRAIN_STEPS=300`：一对固定互斥窗口上做 300 个离线优化步 | 把 CCM 的**在线**协议误套到 Phase B 的**离线**协议；离线阶段窗口是数据集不是消耗品 |
| 2 | "tree 用过一次不能再用于后续 optimizer step" | **删除**。tree 可反复 replay，只有**角色**永久固定 | 同上 |
| 3 | 未定 Phase B 步数 | `TRAIN_STEPS = 300`（§1、§5.1） | 用户冻结 |
| 4 | `B_V_DIM=16` 作为数据不足的降级路径 | 16 改为 **ablation 变体**（改 `d_b:m` 比例）；32 为正式配置 | 用户冻结 |
| 5 | `"32 棵 tree 所以协方差秩最多 31"` | **删除**。改为 §7.2：有效样本量≈32 由 `D` 体现；矩阵秩由 cut 行数决定 | `M2_zz` 按全部 cut 行累加，`tree_id` 只进 `D` |
| 6 | OAS 写作"退化为对角近似" | 改为 §7.1：收缩到**缩放单位阵 `μ_Z I`**，非对角；`a_Z→1` 只是"信号被高度收缩"的诊断，不得凭此宣布结果无效 | `_oas_shrink` docstring |
| 7 | OAS 诊断 1 个字段 | 拆为 `oas_alpha_z_task` / `oas_alpha_p_task` / `oas_alpha_z_protect` / `oas_alpha_p_protect[4]` | 用户冻结 |
| 8 | 未标注 R 为标量时的 OAS 必然值 | §7.1 补：`p=1 ⇒ alpha=0`（可作断言）；`C'_PP` 原样、`C'_ZP=sqrt(1-a_Z)C_ZP` | `_oas_alpha` 公式 |
| 9 | 无泛化监控 | §7.3 新增 `j_cca_task_on_task` / `j_cca_task_on_protect`（诊断，非 gate） | 300 步固定数据的过拟合可见性 |
| 10 | 无 `D_task`/`D_protect` 诊断 | §7 补入 | 有效自由度的直接可见性 |
| 11 | 21 条验收 | 扩到 **24 条**（新增 21/22/23/24） | 覆盖 300 步语义 |

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
- **tree 角色在 300 步中永久固定**，不得 task↔protect 互换。
- **Phase B 只复用冻结输入（`X_v/q_emb/mask/P/R/IDs`），每步重算 `B/OAS/cotangent/梯度`。**
- Phase A 用 Official reduction；Phase B 全程 Ω 调用为 0；Phase C 全程 Γ 更新为 0。
