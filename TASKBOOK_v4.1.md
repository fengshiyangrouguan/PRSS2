# Metaⁿ × RPBE 工程任务书 v4.1

状态：**方法已冻结**。`census.py` 与基础目录/模式开关可立刻开工；
`encoder.py` / `fusion.py` / `trainer.py` 以本文件为准。

- v1–v4 已作废，以本文件为准。
- 基线代码：`meta-n-main`，blob `73e0d5097878d759cac0cdc32c67abe756d2b264`。
- proposal-space 来源：`PRSS2` `fix/avg-lora-clock`，冻结提交 `c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36`。
- v4.1 = v4 + **补齐 6 组实现契约**（§2.8–§2.13、§6.1）+ 3 处验收修正 + 2 处修正。
  变更对照见 §9。

---

## 0. 边界（不可更改）

1. 只修改 Metaⁿ 每次 Ω 调用前的 **context reduction**。注入点有**两处，必须同时覆盖**：

   | # | 位置 | 调用 | 路径 |
   |---|---|---|---|
   | 1 | `meta_n/core/omega.py:119-120` | `OmegaEngine.generate` | 主生成 |
   | 2 | `meta_n/core/omega.py:238-242` | `OmegaEngine.refine` | 自修复 |

   **只改 119-120 会让自修复路径不受 predictive reduction 控制**，三模在 refine 路径上静默退化一致。

2. 保留三种模式：`full`、`official`、`predictive`。
3. 不改 `archive`、`OmegaEngine.generate`/`refine` 本体、`_build_prompt`、`MetaLayer`、evaluator、现有 feasibility solver 内核。
4. frozen encoder、frozen LLM，只训练 Γ。
5. 当前 4080 SUPER 只跑 Metaⁿ + Ours；Official 与 Full Context 在另一张卡单独跑。
6. finite-test 推理只执行一次 context reduction，**不运行两步 rollout**。
7. **离线训练**：正式评估阶段不得更新 Γ（见 §5）。

---

## 1. 冻结常数

| 名称 | 值 | 说明 |
|---|---|---|
| `N_RUNS_TOTAL` | **64** | Phase A 独立 Official-reduction Ω run 总数 |
| `DATA_WINDOW_PAIRS` | **1** | 只建立**一对**互斥的 task/protect 窗口 |
| `TRAIN_STEPS` | **300** | Phase B 离线优化步数 |
| `N_TREES_TASK` | **32** | task 窗口 tree 数 |
| `N_TREES_PROTECT` | **32** | protect 窗口 tree 数 |
| `N_TREE_MIN` | **32** | 每窗口门禁单位 = unique tree |
| `M_SKETCH` | **8** | 每条 LPSE branch 的 P 维 |
| `B_V_DIM` | **32** | `b_v` 维度。**正式配置；16 只作 ablation** |
| `N_BRANCHES` | 4 | 独立，禁止拼成单一 covariance |
| `D_E` | 256 | fusion 维度。**禁止用 LLM hidden 做 fusion dim** |
| `H_LORA` | 64 | `W_K`/`W_V` 低秩瓶颈 |
| `N_SLOTS` | 4 | 四槽 |
| `COND_RANK` | 32 | `Cond` 低秩 |
| `KAPPA` | 0.02 | Stage8 冻结几何标定，**不得按结果调** |
| `TAU` | 3e-4 | certificate 容差 |
| `GAMMA_BUDGET` | 1_000_000 | Γ 参数硬上限 |
| `TEMPERATURE` | 1.0 | **固定** |
| `MAX_DEPTH` | 10 | 取自 `evolutionary_orchestrator.max_depth`，depth buffer 用 |

**`B_V_DIM = 32` 依据**：参考 `(N_tree, d_b, m) = (128,128,32)` 等比例缩小四倍 → `(32,32,8)`，
`d_b:m = 4:1` 不变。16 会改成 `2:1`，属**新方法变体**，只能 ablation。

### 1.1 冻结文本编码器 `E_0`

> **论文表述（必须照此写，避免误读）**
>
> `E_0` is **fixed across all training and evaluation**. The **only learned
> component** introduced by our method is `Γ_θ`, with **108,288 trainable
> parameters**. `E_0` is a frozen feature extractor — **it is not fine-tuned**,
> and no gradient ever reaches it.
>
> 禁止出现"我们训练/微调了一个编码器"这类表述。CodeBERT 在这里的作用等同
> 于一个固定的特征变换，与方法本体无关。

`trace` / `InjectedCode` / `query` / `future observation` **共用同一个 `E_0`**。

```yaml
model_id:   microsoft/codebert-base
revision:   3b0952feddeffad0063f274080e3c23d75e7eb39
tokenizer:  AutoTokenizer.from_pretrained(model_id, revision=<同上>)
model:      AutoModel / RobertaModel, revision=<同上>
hidden_dim: 768
output_dim: 256
pooling:    masked_mean
training:   false
dropout:    model.eval() 关闭
gradient:   torch.no_grad() + detach()
```

**长文本分块（冻结）**：每块 ≤ **256 tokens（含 special tokens）**；**非重叠**；每块 masked mean；
按**有效 token 数加权**聚合；seed 0 固定 Rademacher 投影降到 256：

```
P_rc ∈ R^{768×256},  P_rc[i,j] ∈ { -1/sqrt(256), +1/sqrt(256) }
```

**不使用 text/code 双编码器。**

### 1.2 Γ 参数预算

| 参数 | 形状 | 数量 |
|---|---|---|
| `A_c` | `[32, 256]` | 8 192 |
| `B_c` | `[1024, 32]` | 32 768 |
| `A_K` / `B_K` | `[256,64]` / `[64,256]` | 16 384 ×2 |
| `A_V` / `B_V` | `[256,64]` / `[64,256]` | 16 384 ×2 |
| `w_g` | `[256]` | 256 |
| `Q_0` | `[4, 256]` | 1 024 |
| `LN` weight/bias | `[256] ×2` | 512 |
| **合计** | | **108 288** |

`Cond(q) = B_c A_c q`。

### 1.3 优化器与初始化（v4.1 新增，必须冻结）

```yaml
optimizer:      AdamW
lr:             3e-4
betas:          [0.9, 0.999]
eps:            1e-8
weight_decay:   0
foreach:        false
grad_clip:      1.0
scheduler:      cosine
warmup_steps:   9
total_steps:    300
gamma_init_seed: 0
```

**初始化（全部在 `gamma_init_seed = 0` 下，用同一个 `torch.Generator` 依次抽取）**：

| 参数 | 初始化 |
|---|---|
| `A_c` / `A_K` / `A_V` | `Normal(0, 0.02)` |
| `B_c` / `B_K` / `B_V` | `Normal(0, 0.02)` |
| `Q_0` | `Normal(0, 0.02)` |
| `w_g` | **恰好 0**（⇒ `g = sigmoid(0) = 0.5`，`log(g+eps)` 恒为常数） |
| `LN.weight` | **恰好 1** |
| `LN.bias` | **恰好 0** |

**固定投影（buffer，不训练）**：

| buffer | 形状 | 取值 | seed |
|---|---|---|---|
| `P_rc` | `[768, 256]` | `{±1/√256}` | 0 |
| `P_b` | `[32, 1024]` | `{±1/√32}` | 0 |
| `R_S^(r)` | `[8, 512]` | `{±1/√8}` | **branch seeds `0,1,2,3`** |

**固定 type/depth buffer（v4.1 新增，buffer，不训练）**：

```python
g = torch.Generator().manual_seed(0)
# item 类型：0 = trace, 1 = injected code
t_type = torch.stack([torch.randn(D_E, generator=g) * 0.02 for _ in range(2)])
# 来源深度：0..MAX_DEPTH
r_depth = torch.stack([torch.randn(D_E, generator=g) * 0.02
                       for _ in range(MAX_DEPTH + 1)])
```

二者均为 **buffer（`register_buffer`，无梯度）**，在冻结编码器内部**相加**：

```
X_v[j] = P_rc · e_vj + t_type(type(j)) + r_depth(depth(j))
```

**不得**改为拼接、不得加可训练缩放、不得换 seed。

---

## 2. 模块与契约

```
meta_n/rpbe/__init__.py
meta_n/rpbe/config.py             # 冻结常数 + assert_gamma_budget()
meta_n/rpbe/modes.py              # full | official | predictive 模式枚举
meta_n/rpbe/encoder.py            # FrozenItemEncoder
meta_n/rpbe/fusion.py             # SlottedFusion (Γ_θ)
meta_n/rpbe/selector.py           # PredictiveSelector
meta_n/rpbe/records.py            # CutRecord
meta_n/rpbe/lineage.py            # 两步真实 lineage 抽取
meta_n/rpbe/future.py             # phi_r(S_v)
meta_n/rpbe/window.py             # StatWindow
meta_n/rpbe/kf.py                 # ★ vendored numerical backend
meta_n/rpbe/qp.py                 # proposal-space QP（移植）
meta_n/rpbe/context_reduction.py  # 三模适配器（接 omega.py 两处）
meta_n/rpbe/census.py             # archive census（§4.3）
meta_n/rpbe/trainer.py            # Phase B
```

> **`kf.py` is a vendored numerical backend, not an additional RPBE component.**
> It is a byte-for-byte copy of PRSS2 `src/rpbe/loss.py`
> (`_covs` / `_oas_alpha` / `_oas_shrink` / `_score_from_covs` /
> `latent_z_adjoint`), kept self-contained so that "OAS is on" is a fact
> rather than an assumption — an import from another tree could silently
> diverge. **It introduces no method.** The method is: frozen `E_0` +
> 108,288-parameter `Γ_θ` + the existing QP training constraint.

### 2.1 `encoder.py`

```python
class FrozenItemEncoder:
    def __init__(self, spec: EncoderSpec): ...
    @torch.no_grad()
    def encode(self, items: list[TextItem]) -> torch.Tensor:
        """-> X_v [n_v, 256]，detach，含固定 type/depth encoding。"""
    @torch.no_grad()
    def encode_query(self, q: str) -> torch.Tensor:
        """-> q_emb [256]，detach。"""
```

**只输出 `X_v` 和 `q_emb`；不输出 `Q_v`。**

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
- `M_v`：**只屏蔽 padding**（所有 item 对所有 slot 合法，合法组合 bias 固定 0）。
  **不预定义四槽语义。**

### 2.3 `selector.py`

**必须返回原始对象，不能返回 `str`。** `_build_prompt` 签名：

```python
def _build_prompt(self, traces: list[Trace], context_stack: list[InjectedCode],
                  tasks, depth, inspiration_traces=None, previous_scores=None,
                  archive_best_scores=None, solver_language="python",
                  no_code_library=False, current_scores=None, helper_usage="",
                  focus_task=None, prompt_variant=None):
```

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

硬性要求：返回**原始对象**；`selected_stack` 按 `source_depth` 恢复顺序；`_build_prompt()` 一行不动；
**不得追加 `C_v`**；加权统计量只进 `diagnostics`；
**被保留的 code block 必须完整，允许整块丢弃，禁止中间截断。**

### 2.4 `records.py`

**两个独立的键，互不替代**：`cut_id` 回答「这条 cut 是**哪个候选**产生的」；
`tree_id` 回答「属于**哪条独立历史**」，驱动 unique-tree 门禁。
parent / child / grandchild 关系**不在这里**——由 `lineage.py` 外置维护。

```python
@dataclass(frozen=True)
class CutMeta:                   # 候选自身的 provenance，无 tensor、无 lineage
    candidate_id: str            # **run 限定**的候选身份，见下
    depth: int                   # 递归层 d
    run_id: str
    root_candidate_id: str       # 该 run 内的 lineage root

@dataclass
class CutRecord:
    meta: CutMeta
    cut_id: tuple                # (candidate_id, occurrence_seq)
    tree_id: tuple               # (run_id, root_candidate_id)
    occurrence_seq: int
    X_v: torch.Tensor            # [n_v, 256]  frozen，缓存
    q_emb: torch.Tensor          # [256]       frozen，缓存
    mask: torch.Tensor           # [4, n_v]    固定
    p: torch.Tensor              # [4, 8]      detach（四条 branch 各一行）
    r: float                     # 两步真实收益（标量，见 §2.9）
    weight: float                # parent 归一化权重
```

**`p` 形状是 `[4, 8]`，不是 `[8] per branch`。**
**`tree_id` 必须是 `(run_id, root_candidate_id)` 元组**，否则 64 个 run 碰撞成同一棵树。
**没有 `z` 字段。没有 graph。没有 `Q_v`。没有 `A_v`。**

**`candidate_id` 必须 run 限定。** orchestrator 自己的 id（`gen0_seed`、`gen1_b0_k0`）
**跨 run 不唯一**——每个 run 都有自己的 `gen0_seed`。用
`make_candidate_id(run_id, candidate_id)` 构造（如 `run_0:gen1_b0_k0`）；
**裸 id 一律拒绝**，否则会重现「所有 run 塌成一个候选」——与裸 `tree_id` 同一类 bug。

`cut_id` 与 `tree_id` **相互独立**：一条 cut 所属的候选不必是树的 root。
校验只要求二者各自与 `meta` 一致（`cut_id[0] == meta.candidate_id`、
`tree_id == (meta.run_id, meta.root_candidate_id)`），**不要求 `tree_id == cut_id[:2]`**。

### 2.5 `lineage.py`

- 只接受 archive 中**真实存在**的 `c_v → c_{v+1} → c_{v+2}`；
- `Y_{v+1} = Φ(T_{d+1})`、`Y_{v+2} = Φ(T_{d+2})`：future 是 **child/grandchild 的真实 trace observation，不是 InjectedCode**；
- 缺真实 grandchild 的 cut **不进入 LPSE**，不得补零 / 复制 child / 造假 EOS；
- 每条真实 parent→child→grandchild 是一个 occurrence；**按 parent 归一化权重**。

### 2.6 `future.py`

```
f_v^(1) = E_0(serialize(Y_{v+1}))                    # [256]
f_v^(2) = E_0(serialize(Y_{v+2}))                    # [256]
phi_r(S_v) = R_S^(r) [ f_v^(1) ; f_v^(2) ]           # [8]，R_S^(r) ∈ R^{8×512}
```

**不得加入 `f^(2) - f^(1)` 或 `f^(1) ⊙ f^(2)`。** 全程 `detach`。

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

**OAS 必须显式启用**（现有入口默认全 `False`）：

- `latent_z_adjoint(..., oas=False)` — `PRSS2:src/rpbe/loss.py:301`
- `KFMomentWindow(..., oas=False)` — `PRSS2:src/rpbe/loss.py:914`
- `kf_score()` / `kf_adjoint()` **完全没有 OAS 入口**

正确路径：`latent_z_adjoint(..., oas=True)` 或 `KFMomentWindow(..., oas=True)`（`loss.py:1290` 转发）。
**不得直接调 `kf_score + kf_adjoint` 来"复用 OAS"。**

**梯度纪律**：只有 OAS intensity 可 detach；`C_PP`/`C_RR` 侧固定；
`C_ZZ`、scale normalization、`B` **必须保留梯度**；**严禁 stop-gradient `C_ZZ`**。

```
J_LPSE = (1/4) Σ_r J_CCA(B, P^(r))
J_task = J_CCA(B, R)
```

两者都用 **paired OAS shrinkage + scale-normalized Cholesky**，**禁用 plain ridge inverse**。

### 2.8 `U_v` / `C_v` / `query` / `serialize` 契约（v4.1 新增）

**`U_v`（进 reducer 的被压缩内容）**：

| 调用点 | `U_v` |
|---|---|
| `generate` | `(traces, context_stack)` |
| `refine` | `(child_traces, context_stack + [prev_injection])` |

**`C_v`（不进 reducer、原样传递）**：`_build_prompt` 的其余参数——
`tasks, depth, inspiration_traces, previous_scores, archive_best_scores,
solver_language, no_code_library, current_scores, helper_usage, focus_task, prompt_variant`。
`C_v` **不进入 `U_v`、不被 Γ 改写、也不出现在 reducer 输出里**。

> **更正（2026-09-17，三模 E2E 实测发现）**：上面这份清单把两类东西混在了一起，
> 实现时必须分开看，否则会误判 reducer 有泄漏：
>
> | 类别 | 成员 | 性质 |
> |---|---|---|
> | **任务级上下文**（真 C_v） | `tasks, depth, inspiration_traces, previous_scores, archive_best_scores, solver_language, no_code_library, current_scores, focus_task, prompt_variant` | 与如何裁剪无关，**三模必须逐位相同** |
> | **stack 派生**（非 C_v） | `helper_usage` | 引擎**从裁剪后的 stack 现算**，code item 被丢掉 helper 列表必然缩短 |
>
> `helper_usage` 是 `_build_prompt` 的一个参数，但它的**值**是栈的函数，不是任务级输入。
> 实测反证：把 **official** 的 stack 预算收紧到会丢一层，它的 helper 列表同样从 3 降到 2
> —— 与 predictive 完全一样。所以这不是"predictive 独有的泄漏"。
>
> **验收口径**：`C_v` 等价性只对**任务级上下文**成立；`helper_usage` 单独按"是否随 stack 变化"检查。

**query（只取当前 cut 的局部观测）**：

```python
q_text = OmegaEngine._format_raw_traces(current_traces)
q_emb  = E_0(q_text)
```

`current_traces` 即 `U_v` 的 trace 部分（`generate` 用 `traces`，`refine` 用 `child_traces`）。
**不得包含 child/grandchild，不得包含 `C_v`。**

**serialize（必须复用官方 formatter，禁止另造模板）**：

| item | 编码文本 |
|---|---|
| `Trace` | `OmegaEngine._format_raw_traces([trace])`（`omega.py:798`） |
| `InjectedCode` | `OmegaEngine._format_context_stack([code])`（`omega.py:950`） |

**禁止**为 `E_0` 另写一套文字模板。

### 2.9 `R_v` 契约（v4.1 新增）

```
R_v = mean_score(c_{v+2}) - mean_score(c_v)
```

- `mean_score(c)` = 该候选在 **Official 评估**下的任务平均分；
- 两端必须来自**相同 task cohort**，且**均为有限值**；任一不满足 → **丢弃该 cut**；
- **不裁剪、不归一化、不成对截断**。

### 2.10 serializer（`Y` 的序列化，v4.1 新增）

`Y(c)` 是 child/grandchild 的**真实 trace observation**，序列化后过 `E_0`：

```python
serialize(Y) = OmegaEngine._format_raw_traces(Y.traces)
```

即**复用同一个官方 formatter**，与 `Trace` item 同口径。**不包含 InjectedCode。**

### 2.11 四槽 → 原始对象的还原契约（v4.1 新增，必须写死）

```
1. slot 按 k = 0,1,2,3 顺序处理
2. used = ∅
3. 对 slot k：按注意力 A_v[k, ·] 降序遍历 item j，跳过 j ∈ used
4. 若 item j 所属类型的累计 token 数会超过该类型的 Official budget，
   继续试该 slot 的下一个未使用 item
5. 入选后 used.add(j)，记录 (k, j)
6. 每个 slot 至多产出一个 item；串起来至多 4 个原始对象；
   不足 4 个就少返回，不为了填满 budget 继续加项
7. Trace 按 slot 选择顺序输出；InjectedCode 最后按 source_depth 升序排序
8. 不生成任何新文字
```

### 2.12 梯度符号（v4.1 恢复 v2 中丢失的负号）

```
task：   L̃_task = -Σ_v ⟨sg(a_v^task), b_v(θ)⟩
         g_task = ∇_θ L̃_task = -∇_θ J_task        ← 负号
protect：q_v = ∇_θ ⟨sg(a_v^LPSE), b_v(θ)⟩          ← 正向，保持梯度
```

**task 是负号（最大化 `J_task`）；protect 是正向。写反会把 Γ 训练成降低真实收益。**

### 2.13 `predictive` 模式的 token budget（v4.1 新增）

`predictive` 与 `official` 共用**完全相同的 `ContextBudget`**（trace section 与 stack section 各自的上限）。
`full` 不做 official reduction，使用全部可用上下文，仅受模型硬上限约束。
**token budget 与 Official Metaⁿ 严格对齐。**

---

## 3. 张量形状表

| 符号 | 形状 | 梯度 | 缓存? |
|---|---|---|---|
| `X_v` | `[n_v, 256]` | 无 | **是** |
| `q_emb` | `[256]` | 无 | **是** |
| `mask` | `[4, n_v]` | 固定 | **是** |
| `Q_v` | `[4, 256]` | 是 | **否，每步重算** |
| `K` / `V` | `[n_v, 256]` | 经 `W_K`/`W_V` | 否 |
| `g` | `[n_v]` | 经 `w_g` | 否 |
| `A_v` | `[4, n_v]` | 是 | **否** |
| `Z_v` | `[4, 256]` | 是 | 否 |
| `vec(Z_v)` | `[1024]` | 是 | 否 |
| `b_v` | `[32]` | 是 | **否，每步重算** |
| `φ_r(S_v)` | `[8]` | 无 | **是** |
| `p_v` | `[4, 8]` | 无 | **是** |
| `B` | `[M, 32]` | 是 | **否，每步重算** |
| `P^(r)` | `[M, 8]` | 无 | **是** |
| `R` | `[M, 1]` | 无 | **是** |
| `q_j` | `[len(Γ_params)]` | — | **否，每步重算** |

---

## 4. 缓存与窗口单位

### 4.1 三层缓存

**核心约束：Metaⁿ 的 Ω 是 LLM 调用，不可重放。** pass2 只能 **replay Γ**。

| 层 | 内容 | 生命周期 |
|---|---|---|
| L1 cut cache | `X_v / q_emb / mask / p_v / R_v / IDs` | Phase A 收集期，之后冻结 |
| L2 task window | L1 字段（`R_task` 即标量收益切分） | Phase B 全程固定 |
| L3 protect window | L1 字段 + `P^(r)` | Phase B 全程固定 |

**`B` 明确不得缓存**——每个 optimizer step 在当前 Γ 下重算。
**禁止缓存** graph-connected `z`、`Q_v`、`A_v`、旧 cotangent。
**禁止**缓存后重新调用 Ω，**禁止**在 pass2 重跑原始文本生成。

### 4.2 窗口是数据集，不是消耗品

`KFMomentWindow.window_ready()` 判据（`loss.py:1340-1346`）：

```python
len(self._windows[tau]["tree_seen"]) >= self._threshold(tau)
```

故门禁：`N_unique_trees^task >= 32`、`N_unique_trees^protect >= 32`。

- **同一 tree 的所有 cut 必须进入同一窗口，整棵 tree 不得跨窗口**；
- `set(task_tree_ids).isdisjoint(protect_tree_ids)`；
- 64 个 root 用**固定 hash** 切成 32/32，固化进配置，不得按结果重切。

```python
task_roots    = fixed_hash_split[:32]
protect_roots = fixed_hash_split[32:64]
assert set(task_roots).isdisjoint(protect_roots)
```

**`DATA_WINDOW_PAIRS = 1`**：64 个 root 建立一对互斥窗口后，这对窗口在 Phase B 全程固定不变，
被反复 replay 至多 `TRAIN_STEPS = 300` 步。门禁回答的是"数据是否够用"，**不是"能用几次"**。
**tree 角色永久固定**，300 步中可反复 replay，角色不变。

### 4.3 archive census（立刻可开工）

输出：① eligible lineage 数；② unique parent 数；
③ **unique tree 数（按 `(run_id, root_candidate_id)` 计）**；④ 按 `N_TREE_MIN=32` 能组成的完整窗口对数。

**若 unique tree < 64，禁止假装门禁已满足、禁止重复计数同一 tree 充数**，降级标 `pilot`。

---

## 5. 训练生命周期（必须离线）

```
Phase A：用 Official reduction 做 64 次独立 Ω run，收集并冻结完整 archive trajectories（只读，不训 Γ）
Phase B：Ω 调用严格为 0；在同一对固定互斥窗口上做 300 个离线 Γ 优化步
Phase C：冻结 Γ，在新的正式评估 run 中执行 Predictive reduction
```

**⚠️ 载体状态（2026-09-17，必须区分开发与正式）**

| | 载体 | 说明 |
|---|---|---|
| **开发期** | **AlgoTune** | 评估走本地 `evaluate(program_path)`，**结构上不调用 API**（已实测 inner LLM calls = 0）。用于在 ¥0 下跑通并验证 `records / lineage / future / window / QP / trainer`。 |
| **正式 Phase A** | **v4.1 冻结的 benchmark/协议** | 仍是「64 个独立 Ω run + Official reduction + 冻结完整 archive trajectory」。早期冻结的 archive 载体是 **CO-Bench 36 tasks**，**不是 S2D**。 |

**AlgoTune 不改变 RPBE 算法本身**；但正式 Phase A 若直接拿它替换冻结载体，属于**实验协议变更**，必须显式记录，不得默认继承。

**S2D 彻底冻结**：不再花钱、不作为 Phase-A archive source。其 inner solver 逐条调 `llm()`，叠加 `eval_repeats` 与隐藏的 empty-content escalation，正是 2026-09-17 那次 ¥15.68 事故的来源。**不给 S2D 加「禁止 solver 调 `llm()`」的限制**——那会改变搜索空间。

**Phase C 硬性约束**：不读取未来 child/grandchild；不构造 `R_v`；**不更新 Γ**；只做当前 cut 的一次 reduction。

### 5.1 Phase B 伪代码

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
    g_task = replay_task_surrogate(task_cache, a_task, theta)   # 含负号，见 §2.12
    q_rows = replay_protect_surrogates(protect_cache, a_protect, theta)

    delta_adam = counterfactual_adamw(theta, g_task)
    d_star     = proposal_space_qp(delta_adam, q_rows)

    ok = certificate_then_commit_once(d_star)
    if not ok:
        # 固定窗口下 θ/输入/moments 都没动 ⇒ 下一步会逐位重演同样的失败
        raise PhaseBFailed("certificate failed at step {}: "
                           "formal training FAILED, aborting".format(optimizer_step))
```

**certificate 失败必须终止 Phase B 并标记正式训练失败**，不得空转剩余步数。
理由：cert failure 时参数、moments、scheduler 全未修改，窗口又是固定的，
下一步的 `B / cotangent / Δ_adam / QP` 输入逐位相同 ⇒ **确定性无限重复**。

**每一步必须重算 `B / OAS / cotangent / g_task / q_rows`。**
只能复用冻结输入（`X_v / q_emb / mask / p_v / R_v / IDs`），**不能复用旧梯度、旧 cotangent、旧 OAS 强度**。
**Phase B 全程 Ω 调用严格为 0。**

---

## 6. proposal-space QP

来源：`fix/avg-lora-clock@c05e4fb`，`src/rpbe_embodied/boundary.py::_proposal_space_update`。**移植而非重写。**

### 6.1 QP 约束的正确写法（v4.1 补行范数）

未归一化形式：

```
q_j^T d >= -κ ||q_j||_2 ||Δ_Adam||_2        ∀j
```

若使用归一化行 `H_j`，必须先定义：

```
H_j = q_j / ||q_j||_2
H_j^T d >= -κ ||Δ_Adam||_2
```

**`||q_j||_2 = 0` 的行直接排除**（不参与约束集）。

### 6.2 提交顺序

```python
theta_old = [p.detach().clone() for p in gamma_params]

clip_grad_once()                          # 同一个 clipped grad 用于 preview 与真实 moment

delta_adam = counterfactual_adamw_on_clones()   # clone + 深拷贝 optimizer state
                                                # 真实 m/v/step 不动

d_star = project(delta_adam)              # 见 §6.1

if not certified:                         # v_max > tau
    return False                          # 实参/moments/scheduler 全未修改；上层终止 Phase B

optimizer.step()                          # 用同一个 clipped task gradient，真实 moments 前进一步

if n_active > 0:
    for p, o, dv in zip(gamma_params, theta_old, split(d_star)):
        p.data.copy_(o + dv.reshape(p.shape))   # 覆盖 optimizer 刚写的参数
else:
    keep_real_adamw_write_byte_for_byte()       # proj_n_active == 0：不重新舍入

scheduler.step()                          # 恰好一次
```

冻结值：`kappa=0.02`、`tau=3e-4`。

**Stage8 实测（v4.1 修正——v4 把两行数字串了）**：

| 量 | 值 |
|---|---|
| `vmax_proj` | ~1e-8 对 `tau=3e-4`（四个数量级余量） |
| `proj_shift_ratio`（投影挪动 AdamW proposal 多远） | **mean 4.0%，max 23.7%** |
| `proj_max_viol_before`（未投影的 AdamW step 本会违反多少） | **mean 0.023，max 0.143**，95.5% 的 boundary >1e-4 |
| 约束是否真绑定 | `proj_n_candidates > 0` 占 **98.3%** |
| `abort` | 全程 0 |

**写法禁忌**：不得写 `θ ← θ + d*，全量覆盖，不是 +=`；真实操作是 `copy_(theta_old + d_star)`。
不得写"提交虚拟 moments"。有且仅有一次 `optimizer.step()`，且只在 certificate 通过后。

---

## 7. 诊断字段

| 字段 | 含义 |
|---|---|
| `N_unique_trees_task` / `_protect` | 窗口门禁的实际单位 |
| `M_cuts_task` / `_protect` | cut 行数（矩阵秩相关，非独立样本量） |
| `D_task` / `D_protect` | Welford cluster-aware 有效自由度 `n` |
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
| `oas_alpha_p_task` | **R 为标量时恒为 0；非零即实现错误** |
| `oas_alpha_z_protect` / `oas_alpha_p_protect[4]` | protect 窗口 |
| `j_cca_task_on_task` | task 目标本身 |
| `j_cca_transfer_on_protect` | 见 §7.3（**v4.1 改名**） |
| `theta_version` | 单调递增的参数版本号（验收 #22 用） |

### 7.1 OAS 的正确表述

`C'_ZZ = (1-a_Z) C_ZZ + a_Z μ_Z I`，收缩目标是**缩放单位阵 `μ_Z I`**（`μ_Z = tr(C_ZZ)/d`，
`loss.py:84-113` docstring 明确用 μ·I 因为 plain identity 破坏尺度不变性），**不是"对角近似"**。
同时 `C'_ZP = sqrt((1-a_Z)(1-a_p)) C_ZP` 被强烈压小。
`a_Z → 1` 表示**当前窗口信号被高度收缩**，是诊断量，**不得凭此宣布结果无效**。

**标量 `R` 的必然结果**：`p=1` ⇒ `beta = (1-2)c²+c² = 0`、`delta = (n-1)(c²-c²) = 0` ⇒ **`alpha = 0`**；
故 `C'_PP = C_PP` 原样、`C'_ZP = sqrt(1-a_Z) C_ZP`。

### 7.2 有效样本量 vs 矩阵秩

`M2_zz = Σ_i w_i (z_i - z̄)(z_i - z̄)^T` 按**全部 cut 行**累加（`loss.py:505-517`）。
`tree_id` 只经 `wsum → W2_b → W2_cut → D` 起作用，`D = W - W2_cut/W` 是 `_oas_alpha` 的 `n`。

- **有效独立样本量 ≈ N_TREES = 32**，由 `D` 体现；
- **矩阵代数秩由 cut 行数 `M_cuts` 决定**，不能断言秩上界是 31。

小样本风险确实存在，但这正是 OAS 的用途，**不构成把 `B_V_DIM` 降到 16 的理由**。

### 7.3 迁移诊断（v4.1 改名 + 限制）

- `j_cca_task_on_task`：`J_CCA(B_task, R_task)`；
- `j_cca_transfer_on_protect`：同一泛函在 protect 窗口上评估。

**这两个量不是"干净泛化监控"。** protect 的 `X/P` 每一步都参与 QP、已经影响 Γ，
因此该量只说明"在**未使用 protect-R 标签**方向上的迁移表现"。
**严禁用它选 checkpoint 或调超参。** 二者都是诊断，**gate 仍只有 certificate**。

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
| 9 | virtual AdamW 一致 | 无约束时 `counterfactual_adamw` == 真实 AdamW step（bitwise） |
| 10 | `feasible_d0` 提交 | `d* = Δ_adam`，提交结果 == 未投影 proposal |
| 11 | cert failure 全冻结 | 参数、moments、step counter、scheduler 全不变 |
| 12 | 只提交一次 | 成功路径 `optimizer.step()` 调用次数 == 1 |
| 13 | Ω 零重放 | Phase B 全程 Ω 调用次数严格为 0 |
| 14 | 三模公平性 | `same task/cohort/model/temperature/round/seed/evaluator`；`official_budget == predictive_budget`；**full 使用未裁剪上下文** |
| 15 | reduce 契约 | 返回原始 `Trace`/`InjectedCode` 对象，`_build_prompt` 未改动 |
| 16 | 两处注入点 | `generate` 与 `refine` 都被同一模开关覆盖 |
| 17 | Phase C 纯净 | 评估 run 中 Γ 无更新、未读 future、未构造 `R_v` |
| 18 | tree_id 不跨 run 碰撞 | 64 个 run 产生 **64 个互异 tree_id** |
| 19 | 32/32 切分固定 | 固定 hash 决定，重复运行一致；两组互斥 |
| 20 | Phase A 用 Official | 轨迹收集阶段走 `official` reduction |
| 21 | 300 步复用同一窗口 | `task_cache` / `protect_cache` 在 300 步中**逐字节不变**（只有 Γ 变） |
| 22 | **每步确实重算**（v4.1 重写） | 见 §8.1——**不得**用"相邻步数值必须不同"作判据 |
| 23 | 标量 R 的 OAS | `oas_alpha_p_task == 0.0`（浮点严格），否则断言失败 |
| 24 | tree 角色不变 | 300 步前后 `task_tree_ids` / `protect_tree_ids` 集合完全一致 |
| 25 | **梯度符号** | 人为翻转 `g_task` 符号，断言 `J_task` 下降（而非上升） |
| 26 | **cert failure 终止** | 注入一次 cert failure，断言 Phase B 抛出并停止，且未跑满 300 步 |
| 27 | **R_v 合法性** | 构造 cohort 不一致 / 非有限的用例，断言该 cut 被丢弃且不裁剪不归一化 |
| 28 | **serialize 复用** | 断言 `E_0` 输入逐字节等于 `_format_raw_traces` / `_format_context_stack` 的输出 |

### 8.1 验收 #22 的正确判据（v4.1）

原判据"第 t 步的 `a_task` 等不等于第 t−1 步"**不可执行**：
相邻步可能因 certificate fail、零学习率、或恰在不动点而**完全相等**。

改为同时检查：

1. **每步函数调用计数**：`replay_gamma_no_grad` / `*_oas_adjoint` / `*_surrogate` 在各步各调用 ≥1 次；
2. **`theta_version` 单调递增**（每个成功提交的 step +1）；
3. **新建张量**：第 t 步的 `B` 不是第 t−1 步 `B` 的同一对象（`id()` 不同），且未被缓存；
4. **人为扰动测试**：把 θ 加一个微小扰动后重算，断言 `B` **必须变化**——
   这一条才真正证明 `B` 是当前 Γ 的函数而非缓存。

---

## 9. v4 → v4.1 变更对照

| # | v4 | v4.1 | 依据 |
|---|---|---|---|
| 1 | `R_v` 无公式 | §2.9 冻结 `mean_score(c_{v+2}) - mean_score(c_v)`，同 cohort + 有限值，否则丢弃；不裁剪不归一化 | 用户 |
| 2 | `CutRecord.p` 写作 `[8] per branch` | 改为 **`[4, 8]`** | 用户 |
| 3 | 无 `U/C/query/serialize` 定义 | 新增 §2.8 / §2.10，明确复用 `_format_raw_traces` / `_format_context_stack` | 用户；已核实两函数在 `omega.py:798/950` |
| 4 | 四槽还原过程未写死 | 新增 §2.11 完整 8 步契约 | 用户 |
| 5 | 梯度符号丢失 | §2.12 恢复：task 负号、protect 正向 | v2 有、v3/v4 丢 |
| 6 | QP 约束漏行范数 | §6.1 补 `q_j^T d ≥ -κ\|\|q_j\|\|\|\|Δ_Adam\|\|`，并定义 `H_j`；零范数行排除 | 用户 |
| 7 | 优化器/初始化未冻结 | 新增 §1.3：AdamW 全套 + 初始化 + 全部固定投影/buffer 的公式与 seed | 用户 |
| 8 | `j_cca_task_on_protect` 称"干净泛化监控" | 改名 `j_cca_transfer_on_protect`（§7.3），**严禁用于选 checkpoint/调参** | protect 的 X/P 每步参与 QP，已影响 Γ |
| 9 | 验收 #22 不可执行 | §8.1 重写为调用计数 + `theta_version` + 新建张量 + **扰动测试** | 相邻步可能合法地完全相等 |
| 10 | L2/L3 cache 表含 `B_task` | **删除**；§4.1 明确 **`B` 不得缓存** | 用户 |
| 11 | cert failure 后继续空转 | §5.1 改为**终止 Phase B 并标记正式训练失败**（新增验收 #26） | 固定窗口下确定性无限重复 |
| 12 | Stage8 数字串行 | §6.2 修正：`proj_shift_ratio` mean 4.0%/max 23.7%；`proj_max_viol_before` mean 0.023/max 0.143 | 已核实 `STAGE8_RESULTS.md` |
| 13 | 24 条验收 | 扩到 **28 条** | 覆盖新增契约 |

---

## 10. 不变量

- `C_v` 永不进入 `U_v`，永不被 Γ 改写，**也永不进入 reducer 的输出**。
- InjectedCode **整块保留或整块裁剪**，禁止中间截断。
- 未来信息**只**进入 `p_v`，全程 `detach`。
- child/grandchild 文字**不得**进入 `Γ(U_v)`。
- **`E_0` 的输入只经官方 formatter 产生**，禁止另造文字模板。
- 每 cut 的 `q_j` 分别保存，**绝不提前相加**。
- 有且仅有一次 `optimizer.step()`，只在 certificate 通过后。
- `C_ZZ` 保留梯度；只有 OAS intensity 可 detach。
- 四条 branch 独立，禁止拼成单一 covariance。
- 同一 tree 的所有 cut 在同一窗口；task/protect 的 tree 集合互斥。
- **tree 角色在 300 步中永久固定**。
- **`B` 不得缓存**；每步重算 `B/OAS/cotangent/梯度`；只复用冻结输入。
- **cert failure 终止 Phase B**，不得空转。
- Phase A 用 Official reduction；Phase B 全程 Ω 调用为 0；Phase C 全程 Γ 更新为 0。

---

## 11. 实现状态（2026-09-17 完成）

全部 13 个模块已实现并通过自测，**全程 `paid backend_requests = 0`**。

| 模块 | 关键验证 |
|---|---|
| `config.py` | Γ 参数 **108 288**；`d_b:m = 4:1` |
| `modes.py` | full / official / predictive |
| `census.py` | archive census；跨 run tree_id 不碰撞 |
| `records.py` | deterministic / round-trip / no-leakage / zero-cost；**run-qualified `candidate_id` 硬拒绝** |
| `selector.py` | 对象保真用 `is` 判定；无效槽不渲染；宽度不匹配报错 |
| `lineage.py` | 真实两步 lineage；**无 grandchild → 0 lineage**（不补零）；parent 归一化权重 |
| `future.py` | `R_S^(r)` 固定 ±1/√8；**直接 aligned pair**（无 `f2−f1`、无 `f1⊙f2`） |
| `fusion.py` | 参数量精确 108 288；`w_g≡0`、LN(1,0) |
| `kf.py` | **逐字 vendor** PRSS2 的 Ky-Fan/OAS 核心；`latent_z_adjoint(oas=True)` |
| `window.py` | **尺度不变性 `d/ds J(sB) \|₁ = 0.00e+00`**；门禁按 unique tree；per-cut cotangent 不预加 |
| `qp.py` | **无约束 == 纯 AdamW step（bitwise）**；cert 失败参数+moments 全冻结；恰好一次 step |
| `encoder.py` | 真实 CodeBERT；X_v [n,256]、q_emb [256] 全 detach |
| `context_reduction.py` | **两处注入点都覆盖**；official 模式不包装（字节一致）|
| `trainer.py` | **`J_task` 0.0385→0.0437（最大化，符号正确）**；cert 失败终止 Phase B；role lock |

### 已确认的实证结论（非文档冲突，但重要）

**§7.2 的小样本病态是真的，且已量化。** 当有效自由度 D ≈ 4 而 `B_V_DIM = 32` 时，
OAS 把收缩强度推到 **α = 1**，于是 `C_ZP' = sqrt((1−α_z)(1−α_p))·C_ZP = 0`，
**J 恰好为 0**，梯度全零、Γ 完全不动——不是 bug，是 OAS 的正确行为。
在冻结的 32 tree/窗口下 D ≈ 62，分数非退化。

**含义**：`oas_alpha_z` 必须作为一等诊断监控。它逼近 1 时窗口实际没有信号，
此时任何"训练成功"的结论都不成立。

### 实现中发现、文档未规定、已按最保守方式处理并在此登记

1. **`selector.py` 签名无 mask**（§2.3）：按"`n_v` 精确 = item 数"处理，
   宽度不匹配直接报错；若将来引入 batching/补齐，必须给签名加 `valid` 参数。
2. **§2.11 第 3 步未规定退化注意力行**：全零行 / NaN 行判为无效槽 → **不渲染任何东西**。
   该规则来自验收要求，非文档原文。
3. **§2.7 未给 ridge 值**：取 `RIDGE_EPS = 1e-3`，来源是 PRSS2 `configs/ccm/frozen_method.json`
   的 `rpbe.ridge_eps`，已写入 `config.py` 并注明"只负责数值安全，OAS 承担正则"。
4. **§0.1 只说"修改这两处"，未规定安装方式**：实现为 duck-type 适配器，
   **零编辑 `omega.py`**，靠"sample_traces 返回可原地 finalise 的 list、truncate 时改写它"
   覆盖两处；有顺序守卫与 `refine` 路径测试。
5. **"批级 record 选择"缺口经检验不存在**：`window.py` 的门禁只有
   `unique_trees >= 32`，**没有条数上限、也没有裁剪规则**，因此 §4.2 不需要
   "从超量合格记录里挑"的规则，不需要新建模块。

