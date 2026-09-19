# Phase-A Increment #02 — 逐条补 run，直到 task 侧可跑（manifest）

**状态：PREPARED，随后执行**（用户 2026-09-18 明确授权「继续逐条补 Official run」）。
V4.2.1 代码保持冻结，本批次**不改代码**。

---

## 0. 目标与停止规则（预注册，先于任何 run）

**目标**：让 `phase_b_runnable` 变为 `true`。
当前 `task 1 / protect 5`，task 侧单棵树 → 协方差奇异 → 闭包失败。

**停止规则（预注册，先定后跑）**：

> 严格按顺序逐条执行；**每条跑完立即 0-API 重新 materialize 并检查 `phase_b_runnable`**。
> **一旦为 `true`，立即停止 Phase A，不再追加任何 run。**
> 4 条跑完若仍为 `false`，停止并汇报，**不自动扩展**。

**必须如实的两个声明**：
1. 这是一条**可选停止规则**（optional stopping）：样本量取决于每条的 hash 结果。
   对「能不能跑起来」这个目的无害，但**不得**据此对 lineage 产出率做无偏推断。
2. 停止条件用的是**实测的 `phase_b_runnable`**，不是「task 侧有 2 棵」——
   2 棵是否足够**只能实测**（`D = n_trees − 1` 在此为 1，是否脱离奇异未知）。
   **不从维数反推任何最小 run 数。**

---

## 1. 预注册表

| # | task | seed | 计划 output-dir | 依据 |
|---|---|---|---|---|
| 1 | `Flow shop scheduling` | 47 | `runs/inc02_flow_shop_scheduling` | inc01 同任务产 3 条 lineage |
| 2 | `Crew scheduling` | 48 | `runs/inc02_crew_scheduling` | inc01 同任务产 3 条 lineage，深度到 d5 |
| 3 | `Hybrid Reentrant Shop Scheduling` | 49 | `runs/inc02_hybrid_reentrant_shop` | inc01 同任务产 1 条 lineage |
| 4 | `Flow shop scheduling` | 50 | `runs/inc02_flow_shop_scheduling_b` | 同 #1（不同 seed ⇒ 独立 run） |

**不含 `Job shop scheduling`**：inc01（seed 44）产 0 lineage，作 Phase-A 数据生产者效率差。
**不含 `Bin packing`**（同上，且已在前一批说明）。两者**均保留在 Phase C 评价中**。
此选择在结果出现前做出，依据是**已有 headroom 迹象**，**不是** `tree_role_v421()` 的结果。

**seed 47/48/49/50** 一次性冻结。**禁止**按 hash 结果替换、重命名、重跑任何一行
（`run_increment2.sh` 把 task/seed 硬编码，只接受索引）。

---

## 2. 执行配置

与 inc01 及正式 Phase A **逐项相同**：Official（**不传 `--benchmark-config`**）、B=1 K=1、
`maxiter=4`、`mt=1024`、`--reasoning-effort low`、`grok-4.6`、`--max-retries 0`、
`--empty-retry-max-tokens 0`、`--use-archive`、每 run 硬闸 `--max-backend-requests 40`。
每条启动前核验 resolved config；严格串行、禁止并行。

**预算**：每条约 4.5 万 token；**最多 4 条 ≈ 18 万 token**（提前停止则更少）。

---

## 3. 逐条检查清单（每条跑完）

```
1. backend_requests <= 40 且未触闸      （触闸 => 硬失败，停整批）
2. resolved config: consolidate=true / 官方 config 路径 / 无 none / RPBE adapter 未安装
3. archive/index.json 可读，provenance 完整
4. 0-API 重新 materialize（8+ 已有 run + 本批已完成的）
5. 读 phase_b_runnable —— true 则立即停止 Phase A
```

**硬失败即停整批并汇报**：plumbing/config 异常、官方 config 未正确加载、出现
`--benchmark-config none`、RPBE adapter 意外进入搜索、触闸、产物不完整或 provenance 不可验证。
**不是停止理由**：`eligible_lineages=0`、分数低、深度停在 2、角色比例难看。

---

## 4. Phase B → Phase C 的**已知缺口**（必须先补，否则看不到 Ours 分数）

`phase_b_runnable=true` 之后，原计划是「Phase B 训练 Γ → 冻结 Γ → Phase C 跑 predictive」。
**当前代码在这条链上缺两段胶水**：

| 缺口 | 现状 | 证据 |
|---|---|---|
| **保存 Γ** | Phase B 没有任何把训练后 Γ 落盘的路径 | `trainer.run()` 只返回 history |
| **加载 Γ** | Phase C 的 predictive 分支**新建随机 Γ** | `main.py:1985` `_rpbe_gamma = SlottedFusion()`；全库无 `GAMMA_PATH` / `--gamma` / ckpt 加载 |

**结论**：即使 task 侧解封、Phase B 跑起来，**Phase C 现在也测不出「Ours (trained Γ)」**——
它会用一个随机 Γ。这两段胶水是 **0 API 的纯实现工作**，必须补完，Phase C 的那个数字才有意义。

补完之后才是：
```
Phase B（离线，0 API）→ 保存 Γ → Phase C paired sanity：
   同 held-out task / 同 seed / 同 maxiter+token+model
   Official  vs  Predictive(trained Γ)
只看一个核心 task score；Full Context 后置。
```
