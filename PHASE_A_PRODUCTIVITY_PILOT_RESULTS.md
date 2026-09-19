# Phase-A Productivity Pilot — RESULTS

**状态：3/3 run 完成。等待人工决策。未启动第 4 个 run，未扩到正式 Phase A。**
执行日期：2026-09-18。批准依据：用户对 `PHASE_A_PRODUCTIVITY_PILOT.md` 的正式批准。
配置按 manifest 原样，未改任何 task / 模型 / 参数。

---

## 1. 逐 run 统计（配置核验均通过）

每个 run 启动前都核验过：`consolidate=true`、`regression_guard=true`、
`within_task_recursion=true`、`benchmark_config=meta_n/configs/benchmark_features.yaml`
（**非 `none`**）、`--reduction-mode official`（RPBE adapter 未安装）、且**同一时刻只有一个 `meta_n.main`**。

| # | task | archive_size | max_depth | eligible_lineages | unique eligible parents | 真实 depth≥3 | backend requests | prompt | completion | total tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `Flow shop scheduling` | 5 | 3 | **2** | 1 | ✅ | 11 | 20,118 | 27,249 | 47,367 |
| 2 | `Bin packing - one-dimensional` | 5 | 2 | **0** | 0 | ❌ | 11 | 17,005 | 27,478 | 44,483 |
| 3 | `Job shop scheduling` | 5 | 3 | **2** | 1 | ✅ | 12 | 20,380 | 25,213 | 45,593 |

**run status / failure mode**：三条 run 全部正常收尾，**无 plumbing/relay/capacity 异常导致的中断**。
唯一事件：run 3 有一次 `transport_retry`（自恢复，未放宽预算）。
**没有任何 run 触及 40 次请求的硬闸**（实际 11 / 11 / 12）。

---

## 2. 汇总（manifest §1 的口径）

```
p_eligible-tree = 2 / 3 = 0.667
L_per-run       = [2, 0, 2]        mean = 1.333
total backend requests = 34        (11 + 11 + 12)
total tokens (relay units) = 137,443
```

**按 manifest §4 的决策规则：`p_eligible-tree = 0.667 ≥ 2/3` ⇒ Phase A 值得扩。**
但 n=3 仍很小，且下面 §3 给出了一个影响预算推算的结构性因素。

---

## 3. lineage 结构（逐 task）

**run 1 — `Flow shop scheduling`**（seed 记 0.0000）

```
gen0_seed   (d1, 0.0000)
 ├─ gen1_b0_k0 (d2, 0.7303)  ← 超过 seed，成为新的 best
 │   ├─ gen3_b0_k0 (d3, 0.0000)
 │   └─ gen4_b0_k0 (d3, 0.0000)
 └─ gen2_b0_k0 (d2, 0.0000)
eligible_lineages = 2
```

**run 2 — `Bin packing - one-dimensional`**（seed 记 0.9296）

```
gen0_seed   (d1, 0.9296)   ← 始终是最优，从未被超过
 ├─ gen1_b0_k0 (d2, 0.0000)
 ├─ gen2_b0_k0 (d2, 0.0000)
 ├─ gen3_b0_k0 (d2, 0.3219)
 └─ gen4_b0_k0 (d2, 0.9296)   ← 追平但未超过
eligible_lineages = 0        （深度从未到 3，故无 grandchild）
```

**run 3 — `Job shop scheduling`**（seed 记 0.0875）

```
gen0_seed   (d1, 0.0875)
 ├─ gen1_b0_k0 (d2, 0.1141)  ← 超过 seed，成为新的 best
 │   ├─ gen3_b0_k0 (d3, 0.1141)
 │   └─ gen4_b0_k0 (d3, 0.0000)
 └─ gen2_b0_k0 (d2, 0.0877)
eligible_lineages = 2
```

### 结构性观察（对预算推算重要）

**lineage 出现在「某个子代超过当前最优、从而成为新的 parent」的时候；如果 seed 一直是最好，
parent selection 每轮都回到 seed，深度永远停在 2，就产不出 grandchild。**

- run 1 / run 3：seed 弱（0.000 / 0.0875）⇒ 子代轻易超过 ⇒ 形成 d2→d3 链 ⇒ 2 条 lineage。
- run 2：seed 强（0.9296）⇒ 4 个子代无一超过 ⇒ 全部停在 d2 ⇒ 0 条 lineage。

所以 `p_eligible-tree` **与「seed 强弱 / 任务 headroom」高度相关，是 task-dependent 的**。
这意味着：若 Phase A 的 64 个 run 全部落在「模型一上来就接近最优」的任务上，
实际 eligible tree 产出会显著低于按 0.667 外推的值。
**三个 run 用三个不同 task，因此本轮的 0.667 混了 task 方差与 run 方差，不能分离二者。**

### 一个需要注意的 seed 记分细节

run 1 的 seed 三次采样是 `✗ IndexError / ✗ Timeout(10s) / ✓ 0.712`，但**记录结果是
`0/1 passed, mean_score=0.000`**；parity run 是 `✓0.795 / ✗ / ✗` 记 0.795。
两例一致指向：**seed 的记分只取第一次采样，另外两次只用于建立 `regression_guard` base floor**。
即 **seed 分数是单次抽样的、有噪声的**——run 1 因此以一个「实际本可得 0.712」的种子被记为 0.000。
这不改变 lineage 结论（run 1 仍产 2 条），但它解释了 run 1 与 run 2 的 seed 分数差异来源，
读数据时不应把 seed 分当作该任务的真实能力。

---

## 4. 明确未做的事

1. 未启动第 4 个 run。
2. 未扩到正式 Phase A（64 run）。
3. 未因 run 1、3 结果好而追加样本。
4. 未修改任何 task / 模型 / 参数（run 2 产 0 lineage 也未因此停止或调整）。
5. watchdog 保持只报告 capacity，全程未启动任何 benchmark。

**下一步需人工决策。** 参考 `TASKBOOK_v4.2.md` §0.8 与 §10。
