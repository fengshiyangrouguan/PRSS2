# PHASE-C AUTOPSY V1 — 首次 treatment divergence 的 0-API 尸检

**性质：Phase-C evaluation protocol correction。不是算法变更，不修改核心 V4.2 方法定义。**
日期：2026-09-18。数据来源：`runs/phaseD/{phaseD_off,phaseD_pred}` 的 `llm_io/outer.jsonl`（0 API）。

---

## 1. 四个冻结事实

### F1. shared-root pair 的 d1 完全一致

两臂的 root candidate 目录 `archive/gen0_seed` **逐字节相同**，
`traces/flow_shop_scheduling.py` 的 md5 均为 `ec4bd6ed607aa2e5c62bf81ad2793c67`。
两臂均由 `--resume` 从同一份 root-only checkpoint 启动，**没有第二次 LLM 调用生成 seed**
（两臂均为 8 次 backend 请求 = 4 iterations × 2，不含 seed 调用）。

### F2. 第一次 post-root Ω prompt 在两臂间 **byte-identical**

```
phaseD_off   Ω prompt md5 = db93a5453cc67fa4   len = 8503
phaseD_pred  Ω prompt md5 = db93a5453cc67fa4   len = 8503
```

即：**在该步，官方 reduction 与 trained-Γ reduction 渲染出了完全相同的 prompt。**

### F3. response 不同 ⇒ 第一次 trajectory divergence 是 **pre-treatment sampling**

同 prompt、同 temperature 0.7 下两臂的 Ω response 不同（2,163 vs 2,576 字符）。
因此 d2 子代的差异**不是** reduction 造成的，而是同 prompt 的随机采样。
下游可见：子代 solver prompt 相差恰好一个 372 字符的块
（`## Additional Context` + `compute_makespan` 那行），而两臂生成的 d2 solver
相似度仅 **0.1158**。

### F4. 该步 reducer 结构上 inert

当前单任务 CO-Bench 设定下，d1 root 的 `U_v` = **1 个 trace + 0 个 injected code**。
reducer 面对**单元素上下文没有选择空间**，因此两臂输出必然相同。

> **推论（本文件的核心）**：`d1 root quality` 与 `treatment 能否在 d2 生效` **无关**。
> 无论 root 的 score 是 0、0.7 还是 0.9，第一次 post-root Ω 都没有可裁剪的上下文。
> 因此「抽到一个更好的 root」**在结构上解决不了** treatment onset 的问题。

---

## 2. 由此产生的协议修订

把 Phase-C 配对从 **shared root** 收紧为 **shared prefix until first treatment-capable cut**。

> **Fork the two arms only at the first cut where the reduction operator has at least
> two eligible context items and can therefore change the rendered prompt. All preceding
> stochastic computation is shared and frozen.**

比 "shared root" 更严格、也更一般：不硬编码 depth，而是硬编码 **first treatment-capable cut**。
本任务下它今天等于 d3；换一个 root 自带多条 traces 的 benchmark，它可能等于 d2。

### 冻结规则

1. Official 与 Predictive **共享所有 treatment 尚不可能生效的 stochastic prefix**。
2. 每次 Ω call 前，检查当前 `U_v` 中 **可供 reducer 选择的真实 context item 数**。
3. 若在该 cut 上两臂的 reduction **结构上不可能产生不同输出**（例如只有 1 个合法 item），
   该 Ω call 属 **pre-treatment prefix**：**只执行一次、结果冻结、两臂共同继承、禁止分别采样**。
4. 到达**第一个 treatment-capable cut** 后才 fork：
   Official → official reduction；Predictive → trained Γ。
5. fork 时**必须断言**：candidate state 逐字节相同、archive prefix 相同、traces 相同、
   `injected_codes` 相同、`parent_id` / `depth` 相同。
6. fork 后第一次 Ω call **必须记录**：pre-reduction `U_v`、Official selected context、
   Predictive selected context、rendered prompt、以及是否**实际**发生 selection/prompt 差异。
7. 若所谓 first-active cut **最终仍渲染出完全相同的 prompt**，**不得**把该步计为 treatment；
   继续共享其 response，直到真正出现 treatment-capable / treatment-different cut。
8. 逐深度表中：**pre-treatment depths 只标 shared baseline**，不用于判断方法效果；
   **只有 treatment-active depth 进入 paired Δ(d)**。
9. 保留 held-out test、**dev-only candidate selection**、无 test leakage、**Γ frozen** 等现有协议。

---

## 3. `d3 Δ = −0.0243` 的处置

**标记为 protocol diagnostic，不进入正式 effect estimate。**

理由：不是「负效果但方差大」，而是更严格的说法——

> 两臂在 treatment 第一次真正能作用**之前**，就因为**同 prompt 的随机 Ω response** 分叉，
> 因此该 pair **没有实现 treatment-onset alignment**。

与上一轮「root 不同」的污染不同：那一轮在 **d1** 就被污染；本轮 d1 一致，
但污染发生在 **inert d2 step**。now 已知应把 shared prefix 再推进一层。

---

## 4. 正式逐深度表的新形状

| pair | depth | treatment active? | Official | Ours | Δ |
|---|---:|---|---:|---:|---:|
| 1 | d1 | no | shared | shared | 0 |
| 1 | d2 | no | shared | shared | 0 |
| 1 | d3 | **yes** | … | … | … |
| 1 | d4 | yes | … | … | … |
| 1 | d5 | yes | … | … | … |

N 个独立 shared-prefix pair 后汇总：

```
Delta_bar(d) = (1/N) * sum_i [ S_i_ours(d) - S_i_official(d) ]
```

**只有 treatment-active depths 进入「我们涨没涨」的主结论。**

---

## 5. 明确不做的事

- **不做**「抽到一个能跑的 root 为止」：那引入 root-selection 问题，且**没有解决**
  treatment inert 的结构原因（F4）。
- **不筛** shared d2 prefix：先构造一次、冻结，然后**直接看**它是否满足
  「first treatment call 的 `U_v` 含 ≥2 个可选 item」。若不满足，**如实报告**
  「该 benchmark/task 下 treatment onset 更晚」，**不为凑 treatment 而重抽**。
- 不修改 OAS / kf.py / split / trainer / Meta^n 主循环。
