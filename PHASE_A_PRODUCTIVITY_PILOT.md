# Phase-A Productivity Pilot — MANIFEST

**状态：PREPARED / NOT EXECUTED. `prepare, do not execute.`**
必须经人工批准后才能运行。本文件本身不会触发任何 API 调用。

---

## 0. 这是什么，不是什么

| | |
|---|---|
| **是** | **Phase-A productivity pilot** —— 估计「一次独立 Official run 能稳定产多少真实 lineage、多少 run 会产 0」 |
| **不是** | ~~parity~~（parity 已完成，见 §5） |
| **不是** | ~~正式 64-run Phase A~~（**明确不启动**） |

**为什么需要它**：目前只有 **1 条**配置正确的 parity run。虽然结果很好
（`archive 1→5`、`eligible_lineages=2`、`seed 0.795→d2 0.829`），但 **n=1 不足以据此预算
64-run Phase A**。直接铺开就会重演「先花钱、后才发现生产率不够」的老问题。

---

## 1. 口径（取代「凑 32+32」）

**不再以「凑够 32+32」作为 Phase A 的直接预算依据。** 先估这两个量：

```
p_eligible-tree = (# runs producing at least one eligible lineage) / (# independent runs)
L_per-run       = eligible lineage count per run
```

有了它们才能反推 64-run budget 大致能买到多少 task/protect eligible trees。

**唯一的正确样本只能记成（n=1，不得写成成功率 100%）：**

> **Official productivity observation #1**: 1/1 run produced eligible lineage;
> 2 eligible lineages observed.

---

## 2. 三个 run 的规格

3 个**独立** Official run，覆盖 **3 个不同的 CO-Bench task**。
**每个 run 的 Meta^n 配置与正式 Phase A 完全一致**；
**不为了提高 lineage 人为改动 iteration / gate / beam / acceptance。**

| # | task | 选择理由 |
|---|---|---|
| 1 | `Flow shop scheduling` | 与 observation #1 连续，可直接对比 |
| 2 | `Bin packing - one-dimensional` | 结构不同（装箱 vs 排序），早期 preflight 用过 |
| 3 | `Job shop scheduling` | 同族但更难，检验难度对生产率的影响 |

> task 列表可由人工替换；关键是**三个不同 task**，其余参数不得动。

| 参数 | 值 | 说明 |
|---|---|---|
| `--benchmark` | `co_bench` | |
| `--bench-tasks` | 上表单个 task | 每个 run 一个 task |
| `--reduction-mode` | `official` | 与 Phase A 一致 |
| `--use-archive` | 必开 | 非 archive run 会被拒绝启动 |
| `--benchmark-config` | **不传**（= `auto`） | **绝不传 `none`**（那是 control） |
| `--beam-width` / `--beam-candidates` | `1` / `1` | 与 Phase A 一致 |
| `--max-iterations` | `4` | **与 parity 相同，不得为提高 lineage 上调** |
| `--no-early-stop` | 开 | 跑满 4 轮 |
| `--max-tokens` | `1024` | |
| `--empty-retry-max-tokens` | `0` | 不隐式升预算 |
| `--max-retries` | `0` | |
| `--reasoning-effort` | `low` | 与 parity 相同 |
| `--model` | `grok-4.6` | 唯一验证可用的 backbone（gemini 关不掉推理） |
| `--seed` | `42`（Meta^n 默认） | 三个 run 用不同 task，故天然独立 |
| `--output-dir` | `runs/pilot_<task_slug>` | 每 run 独立目录 |

**预估调用上限（硬闸）**：每 run `--max-backend-requests 40`，总计 **最多 120 次**。
（parity 实测 11 次/run，故 40 是 ~3.6× 余量。）

**预估 token**：parity 实测 40,881 token/run（19,390 prompt + 21,491 completion）
⇒ 3 run 约 **123k token**。

**预算口径说明**：这台 relay **没有 USD 计价**（ledger 的 `Outer LLM cost` 恒为 `$0.0000`，
因为 grok-4.6 不在 `PRICING` 表里）。因此预算只能按 **backend requests + tokens** 计，
不要报美元。真实成本是**实验室 relay 的配额**。

**预估墙钟**：parity 单 run ≈ 11 分钟 ⇒ 3 run ≈ 35 分钟（串行）。

---

## 3. 每个 run 收集的统计字段

| 字段 | 来源 |
|---|---|
| `archive_size` | `archive/index.json` → `len(candidates)` |
| `max_depth` | archive 内 `depth` 最大值 |
| `eligible_lineages` | `rpbe.lineage.extract_lineages(run_dir)` |
| `unique_eligible_parents` | 上述 lineage 的 `candidate_id` distinct 数 |
| `has_real_depth_ge3` | `max_depth >= 3`（真实出现 depth≥3，而非仅规划） |
| `total_calls` | ledger 的 `backend_requests`（+ `logical_calls`） |
| `prompt_tokens` / `completion_tokens` | `summary.json` 的 outer usage |
| `cost_relay_units` | 同上 token 数（无 USD 计价） |

派生量：`p_eligible-tree`、`L_per-run`（§1）。

---

## 4. 三个 run 跑完之后的决策规则

| 观测 | 判断 |
|---|---|
| ≥2/3 run 产出 ≥1 eligible lineage（如 `2/2/1`） | Phase A **值得扩** |
| 少数 run 产出（如 `2/0/0`） | **先研究生产率**，不要硬烧到 64 |

**注意**：三个 run 用**三个不同 task**，所以这第一轮混了「task 难度方差」与「run 方差」，
**不能**据此分离二者。它回答的是「跨 task 的产出范围」，这对预算已经是足够保守的上界估计。

---

## 5. 已完成的前置事实（不重复花钱验证）

**parity（1 run，配置正确的 Official）**：

```
runs/canary_parity/20260918_002348_co_bench_grok-4.6
  archive_size      1 -> 5
  eligible_lineages = 2
  seed 0.7953 -> gen1_b0_k0 (d2) 0.8292 -> gen3/gen4_b0_k0 (d3)
  child.injected_codes[:-1] == parent.injected_codes     (继承)
  depth-3: pre_process layers=[2,3], ran=True            (跨层 context 累积)
  11 backend calls / 40,881 tokens
```

四条疑点（Meta^n recursion / 父层继承 / 跨层 context accumulation / lineage extractor）
已全部闭合。**因此本 pilot 只测「生产率」，不重复测「正确性」。**

---

## 6. 明确禁止

1. **禁止** 自动执行本 manifest 的任何 run —— `prepare, do not execute`。
2. **禁止** 传 `--benchmark-config none`（control 路径，会让 quality gate 复活）。
3. **禁止** 为提高 lineage 改 `--max-iterations` / gate / beam / acceptance。
4. **禁止** 用本轮结果宣称「lineage 成功率」——n=3 仍然很小。
5. **禁止** 把本轮当成正式 Phase A 的一部分；它是**独立命名的 pilot**。

---

## 7. 执行方式（人工批准后）

```bash
# 人工批准后，逐个 run 执行（不要批量无人值守）：
bash run_pilot.sh "Flow shop scheduling"
bash run_pilot.sh "Bin packing - one-dimensional"
bash run_pilot.sh "Job shop scheduling"

# 收统计（0 API）：
python scripts/pilot_stats.py runs/pilot_flow_shop_scheduling/<run_dir> ...
```

参考：`TASKBOOK_v4.2.md` §0.8（Official 必须保留官方 config）、§10（本次事故与 parity 数字）。
