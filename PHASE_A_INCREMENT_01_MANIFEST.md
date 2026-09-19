# Phase-A Increment #01 — 4-run Official manifest

**状态：PREPARED / NOT EXECUTED. `prepare, do not execute.`**
未经人工批准不得运行。本文件不触发任何 API 调用。

**前置：V4.2.1 代码已冻结**，本批次**不需要改任何代码**。

---

## 0. 这是什么

| | |
|---|---|
| **是** | 一个小批次 **checkpoint**：跑 4 条独立 Official run，然后重新 materialize + census |
| **不是** | ~~「4 条就够」的声明~~ |
| **不是** | ~~由 `D = n_trees − 1` 反推出来的数量~~ |

**4 只是一个足够小的批次**（成本约 4 × 4.5 万 ≈ **18 万 token**），跑完就停、再判断。
**禁止**自动追加第 5、6、7……条。

---

## 1. 预注册表（先冻结，后执行）

| # | task | seed | 计划 output-dir | 依据 |
|---|---|---|---|---|
| 1 | `Flow shop scheduling` | **43** | `runs/inc01_flow_shop_scheduling` | **已实测 headroom**：seed 记 0.0000 → d2 0.7303，产出 2 条 lineage |
| 2 | `Job shop scheduling` | **44** | `runs/inc01_job_shop_scheduling` | **已实测 headroom**：seed 0.0875 → d2 0.1141，产出 2 条 lineage |
| 3 | `Hybrid Reentrant Shop Scheduling` | **45** | `runs/inc01_hybrid_reentrant_shop` | 同**调度族**外推（**尚无实测 headroom**） |
| 4 | `Crew scheduling` | **46** | `runs/inc01_crew_scheduling` | 同**调度族**外推（**尚无实测 headroom**） |

**seed 冻结说明（2026-09-18，执行前最后一个合理修改窗口）**：#1/#2 是**已跑过的 task**，
若仍用旧 run 的 `seed=42`，新增 trajectory 可能在本地随机部分与旧 run 重复。
故本批一次性冻结为 **43 / 44 / 45 / 46**。这**不是**按结果调参，也**不涉及 split**。

**不含 `Bin packing - one-dimensional`**：它在强 seed（0.9296）、headroom 极小的情形下产出
0 lineage，作为 Phase-A 数据生产者效率差。**它以后仍保留在 Phase C 评价中**——
排除它是因为数据生产效率，**不是**因为它"对我们不利"。此决定在结果出现前已做出。

**#3 / #4 无实测 headroom，照样冻结执行、照单接受。** 不因结果中途替换。

---

## 2. split role 在预注册时**未被计算**

`tree_id = (run_id, root_candidate_id)`，而 Meta^n 的 **run 目录名 = `<时间戳>_co_bench_<model>`**，
由 Meta^n 在启动时生成。

**准确表述（不夸大）**：

> **split role was not computed at preregistration.** 本批次的行是按「任务族 headroom 迹象」
> 选的，**不是**按 `tree_role_v421()` 选出来的。
> **每个预注册行 exactly-once**；
> **禁止通过重启、改时间戳、等待特定启动时刻等任何方式 reroll split role。**

### 2.1 为什么必须写死 exactly-once

角色依赖 `run_id`，而 `run_id` 依赖启动时刻。因此「换个时刻重启」在**原理上**可以改变角色分布——
**知道命名规则的人是有能力这么做的手法的**，所以这里**不声称"数学上不可能"**。
防线是规则，不是不可能性：

> **每个预注册行只执行一次（exactly once）。** 若某条因 plumbing 失败，
> **如实报告失败，不得以新时间戳静默重启**（那等于重掷它的角色）。
> 重试必须回到人工批准，并显式记录这次重试。

`run_increment.sh` 已把这条做成**硬检查**：该 index 的输出目录已有 run 时 **rc=3 拒跑**。

### 2.2 无条件接受

执行后**无论** lineage 是否为 0、**无论**落到 task 还是 protect，**一律保留**。
禁止替换、重命名、重跑、挑选。

---

## 3. 执行配置（与正式 Phase A 逐项一致）

| 参数 | 值 |
|---|---|
| `--benchmark` | `co_bench` |
| `--bench-tasks` | 上表单个 task |
| `--reduction-mode` | `official` |
| `--use-archive` | 必开 |
| `--benchmark-config` | **不传**（= `auto`；**绝不传 `none`**） |
| `--beam-width` / `--beam-candidates` | `1` / `1` |
| `--max-iterations` | `4`（与 parity/pilot 相同，**不得为提高 lineage 上调**） |
| `--no-early-stop` | 开 |
| `--max-tokens` | `1024` |
| `--empty-retry-max-tokens` | `0` |
| `--max-retries` | `0` |
| `--reasoning-effort` | `low` |
| `--model` | `grok-4.6` |
| `--seed` | **43 / 44 / 45 / 46**（按上表逐行，一次性冻结） |

**硬闸**：每 run `--max-backend-requests 40`（实测约 11–12 次/run）。
**预算**：约 44 次请求、**≈18 万 token**；**无 USD 计价**（relay 无 pricing，只按 requests+tokens 计）。

**执行规则**：严格逐条串行、**禁止并行**、**禁止批量无人值守**；每条启动前核验 resolved config
（`consolidate=true`、官方 config 路径、无 `none`、RPBE adapter 未安装）。

---

## 4. 停止条件（跑完 4 条即停）

4 条**全部完成**后**立即停止**，然后：

1. 运行 Phase A.5 materialization：`python -m meta_n.rpbe.build_records <4 个新 run dir>`
   （可连同已有 3 条正确 run 一起 materialize）。
2. 报告：

```
N_eligible_trees_task / N_eligible_trees_protect
D_task / D_protect
alpha_z_task / alpha_z_protect
J_task / J_LPSE
phase_b_runnable / formal_32x32_ready
```

3. **无论结果如何都不自动继续。** 若仍出现「一侧为空」，或两侧都有但 `α_z = 1, J = 0`，
   **停下来再判断**，等人工决定。

**明确禁止**：自动跑第 5 条；因结果不好而换任务重跑；因结果好而临时加样本。

---

## 5. 本批次**不**主张的事

- 不主张「4 条足够」；不主张「还差多少条」。
- **不从维数反推最小 run 数。** 当前只能持有两条已冻结的结论：
  1. **经验事实**：3 棵独立 eligible tree 明显不够；
  2. **实测观察**：`D` 主要由 **tree 数**驱动，不由 cut 数驱动。
  能否从 `α_z = 1` 脱离，取决于真实 records 的**协方差结构**，**只能实测**。
- 唯一有意义的转折点：**真实数据首次从 `α_z = 1, J = 0` 脱离退化。**

---

## 6. 执行方式（人工批准后）

```bash
# 逐条执行，不要批量无人值守：
INCREMENT_APPROVED=yes bash run_increment.sh 1
INCREMENT_APPROVED=yes bash run_increment.sh 2
INCREMENT_APPROVED=yes bash run_increment.sh 3
INCREMENT_APPROVED=yes bash run_increment.sh 4

# 然后（0 API）：
python -m meta_n.rpbe.build_records runs/inc01_*/*
```

参考：`TASKBOOK_v4.2.md` §13（v4.2.1 split）、§4.4（Phase A.5）、§4.4.1（端到端验收）。
