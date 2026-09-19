# SPLIT PROTOCOL — V4.2 DRAFT（待批准）

**状态：草案。未改任何代码。`census.fixed_hash_split` 保持原样，`build_records.py` 不受影响。**
本文件只做协议逻辑的比较与推荐，供人工批准后再动代码。

---

## 1. 结论摘要

1. **截断式 split 应当正式废弃**——它把 `32` 从「formal target」偷偷变成了算法结构，
   与已冻结的 V4.2 语义直接冲突（§2）。
2. **推荐改为逐-tree 固定哈希分配**，用四条不变量约束它（§3）。
3. **当前实测结果（1 task / 2–3 protect）照单接受**，不重 hash、不换 salt、不手工搬 tree（§4）。
4. **`phase_b_runnable` 与 `formal_32x32_ready` 是两件事**，必须分开定义（§5）。
5. **现有 3 棵 eligible tree 仍然不跑 Phase B**（§6）。
6. **明确拒绝「由维数反推最小 run 数」这类推理**（§7）。

---

## 2. 现制（截断式）为什么必须废弃

现在的实现是 `meta_n/rpbe/census.py::fixed_hash_split`：

```python
ordered = sorted(uniq, key=sha256)     # 按 tree_id 的 sha256 排序
return ordered[:n_task], ordered[n_task:]   # n_task 默认 32
```

问题在于 `32` 出现在**分配机制内部**，于是 tree 数一旦不足，角色分配就被结构性地扭曲：

| unique trees | 结果 | 说明 |
|---|---|---|
| 1–32 | `n / 0` | **protect 必为空** |
| 33 | `32 / 1` | protect 只有 1 棵 |
| 40 | `32 / 8` | |
| 64 | `32 / 32` | 只有到这里才等于期望 |

这与已冻结的语义**直接冲突**：

> 「32/32 是 formal-data target，**不是**生产训练 gate」（TASKBOOK v4.2 §1、§4.2）

截断式把「目标」实现了成「算法结构」：它不只是"达不到 32/32 时降级"，
而是**在 n<33 时让 protect 窗口变成空的**——而空窗口会被 `PhaseB` 合法拒绝
（`both windows must be non-empty`；protect 行正是 QP 约束的来源）。
结果是：一个本该只是「数据不足、标 pilot」的情形，被放大成了「结构上无法构造」。

**附带不一致**：V4.2 §4.3 写的是「不足 32/32 就用真实得到的 27/25 继续 Phase B」，
但截断式**永远只能给出 `(32, n−32)`，给不出 `(27, 25)`**。
也就是说当前实现连 V4.2 自己写下的降级语义都无法表达。

---

## 3. 推荐方案：逐-tree 固定哈希分配

```python
# SPLIT_SALT 是冻结常量，一经确定永不更改
digest = sha256(SPLIT_SALT + canonical_tree_id).digest()
role   = "task" if (digest[0] & 1) == 0 else "protect"
```

- `canonical_tree_id`：tree 的**不可变规范形式**，即 `(run_id, root_candidate_id)`
  以固定分隔符拼接（与 `census` 现有 `key(t)` 的拼法一致即可，但必须写死）。
- `SPLIT_SALT`：冻结字符串。**一旦冻结，任何情况下不得修改。**

### 3.1 四条不变量（这才是重点，不是具体写法）

| # | 不变量 |
|---|---|
| **I1** | **角色只由 tree 自身的不可变 ID 决定**。与 run 顺序、目录名、批次、并发度无关。 |
| **I2** | **在看到任何结果之前就已决定**——即在看到 score、lineage 数量、cuts 数、`D`、`α_z`、`J` 之前。分派阶段不得读取任何结果字段。 |
| **I3** | **后续新增 tree 时，旧 tree 永不换角色**。角色是 `tree_id` 的纯函数，不是集合的函数。 |
| **I4** | **禁止为了凑 50/50 而重 hash、换 salt、手工搬 tree**。也不得因某任务"不利于产 lineage"而把它踢出。 |

其中 I3 是相对截断式最实质的改进：截断式下 `ordered[:32]` 会随集合变化而整体位移，
**同一个 tree 在不同批次里可能是不同角色**——这在"冻结实验协议"下是不可接受的。

### 3.2 这样做的代价（如实列出）

- 实际比例会**随机偏离 50/50**，小样本下偏差很大。
- **接受它**。§4 说明为什么"不好看"反而是正确性的证据。

---

## 4. 当前实测落比：照单接受

对现有 tree 做了一次性测算（**只算，未写成代码、未冻结任何东西**）：

```
census 的 4 棵树        -> 1 task / 3 protect
有 records 的 3 棵树    -> 1 task / 2 protect
```

这不是 4/0——哈希规则下 protect 窗口非空，这是截断式永远给不出的。

**这个 1/3 不好看，但必须原样接受。** 理由正是 I4：
如果我们把它"调"成 2/2 或 3/1，就等于**按结果调 split**，而 split 的全部意义就是
在结果出现之前固定角色。一个能被调到满意的 split，等于没有 split。

> 如实声明统计效力：n=4 的 1/3 基本是抛硬币的结果，**统计上什么都说明不了**。
> 它的价值不在于比例本身，而在于证明「角色已按预注册规则固定」这件事可被执行。

---

## 5. 两个状态必须分开：`phase_b_runnable` 与 `formal_32x32_ready`

| 状态 | 含义 | 判据 |
|---|---|---|
| **`phase_b_runnable`** | **能不能构造并跑 Phase B** | task / protect **两侧都有真实 records**，且满足训练结构契约（窗口非空、无 cut 跨窗口、无重复 cut）；**不设 tree 数下限** |
| **`formal_32x32_ready`** | **数据量是否够撑正式置信度** | `N_unique_trees_task >= 32` **且** `N_unique_trees_protect >= 32` |

- `phase_b_runnable = true`、`formal_32x32_ready = false` ⇒ **可以跑**，结果标 `pilot` / `underpowered`。
- `phase_b_runnable = false` ⇒ **不可构造**，这才是真正的"跑不了"（例如某侧窗口为空）。
- **`formal_32x32_ready` 是标注，不是准入。** 它永不阻断 Γ 训练。

这样语义彻底干净：把「数据不够好」和「结构上跑不了」这两件完全不同的事分开报告。

---

## 6. 现有 3 棵 eligible tree **仍然不跑** Phase B

这是本轮 builder 给出的真正答案。真实 records 已实测（**不是合成推测**）：

```
6 cuts
3 unique parents
3 unique trees
D        = 2.0
alpha_z  = 1.0
J_task   = 0.0
protect  = 空（冻结规则下）
```

**现有数据下 Γ 没有可训练信号**：`α_z = 1` 意味着 OAS 把协方差完全收缩掉，`J_task` 恰好为 0。

必须说清楚一件事：**逐-tree hash split 解决的是「协议错误」，不是「数据不足」。**
如果把 3 棵 eligible tree 按新规则拆成例如 1/2，**单个窗口的有效 tree 数只会更少**。
所以：

> **不要把 split 草案当成 Phase B 可以开跑的理由。**

---

## 7. 明确拒绝的推理方式

**不得**写下这条链条：

> ~~`B_V_DIM = 32` → 所以需要 32+32 → 所以再跑几十个 run。~~

它是机械的，而且把两件不同的事混为一谈：**知道"不够"不等于知道"要多少才够"**。
`α_z` 能否从 1 脱离，取决于真实 records 的**协方差结构**，不能由维数单独推出。

同样**不得**写「由 `D = n_trees − 1` 反推最小 run 数」。
当前只能冻结这条**经验事实**：

> **3 棵独立 eligible tree 明显不够。**

以及这条可测的定量观察（本批 6 cuts / 3 trees 实测）：

> **`D` 主要由 tree 数驱动，不由 cut 数驱动**——在同一棵树上多产 cut 不会增加有效自由度。
> （这是本批的一条实测观察，不是一条可以外推的闭式最小样本量公式。）

### 7.1 推荐写进规范的表述

> `32/32` is the formal target; **no smaller tree count is declared a priori sufficient**.
> Pilot sufficiency is determined **empirically from the frozen diagnostics** rather than
> inferred solely from dimensionality.

---

## 8. 后续的零成本再评估流程（每次新增一小批 Official run 后）

**不允许**一次性铺开去"凑数"。每增加一小批，先零成本重新 materialize，然后记录：

```
N_eligible_trees_task / N_eligible_trees_protect
D_task / D_protect
alpha_z_task / alpha_z_protect
J_task / J_LPSE
```

**真正的转折点**只有一个：

> **真实数据第一次从 `α_z = 1, J = 0` 脱离退化。**

这个数**只能实测**，现在不得拍脑袋定成 8、16 或 32。

---

## 9. 本草案不包含 / 待人工决定

1. **不改代码**：`census.fixed_hash_split` 保持原样，本文件不落地任何实现。
2. **`SPLIT_SALT` 取什么值**——待批准后一次性冻结，此后永不更改。
3. **`canonical_tree_id` 的确切拼接形式**——实现时写死，并加测试防止漂移。
4. **是否给 `fixed_hash_split` 提供 v4.2 旁路**（保留旧函数以兼容历史结果）
   还是直接替换——待决定。
5. **下一批 Official run 的规模**——按 §7，不得由维数反推；等 §8 的实测数据。

---

## 10. 与 V4.2 的关系

- TASKBOOK v4.2 §4.4（Phase A.5）已把 `build_records.py` 纳入正式协议；
  本草案的 `phase_b_runnable` / `formal_32x32_ready` 之分是 §4.4 的准入依据。
- 本草案若获批准，应作为 **`## 13. Split 协议（v4.2.1）`** 并入 TASKBOOK，
  并在 §12 变更对照中登记。
