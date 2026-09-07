# TGN 结构消融实验报告（Part III）

> **日期**：2026-09-07
> **代码**：feature_abl 分支（见 §9 commit hash）
> **宿主**：官方 twitter-research TGN（vendored）节点分类，wikipedia 数据集
> **任务**：冻结预训练 TGN（t2_pretrain），jointly fine-tune host + Gamma + decoder，
>   val AUC 选模，zero-memory replay 后测 held-out test

---

## 1. Loss 与 1Obs/2Obs 的真实构造（静态审计结论）

训练总 loss：`L = L_task + λ·L_KF`，其中 Ky Fan 谱得分

```
J_τ = tr[(Σ_ZZ + εI)^{-1} Σ_ZP (Σ_PP + εI)^{-1} Σ_PZ]
```

per compressible interface τ（`0 < layer < L`，即 layer1/layer2 for L=3）。
P = 固定未来测量（psi 的 sketch），来自每个 cut 未来真实事件的固定特征。

**1Obs/2Obs 构造**（同一 cut 的连续两个严格未来事件，非跨层父子对齐）：

- 每个 cut `(node, cut_time)` 经 train-only future index 取该 node 之后的头两个
  训练事件 Y1（horizon=1）、Y2（horizon=2），缺失则省略该 horizon 行。
- `1obs`：只发 Y1 行，per-tree 总权重 1 给 Y1。
- `2obs_aligned`：发 Y1 与 Y2 两行，各 0.5 权重，per-tree 总权重 1。
- `2obs_mispaired`：发 Y1 与一个被"错配"的 Y2 —— 见 §3。

**重要口径声明**：当前 JODIE trace 只沿 root 的 SELF spine 记录（一个 root 的
node 在 layer1/layer2 各一个 cut，递归沿用同一时间戳），因此**并不存在**规格
假设的 `Y_v^(2) = Y_p(v)^(1)` 跨层父子 occurrence 对齐。mispaired 臂破坏的是
同一 cut 两级未来观测的**样本级配对**，不是递归 closure 对齐；报告中 mispaired
臂应表述为 "shuffled/mispaired 2nd observation"，不可表述为 "w/o recursive
alignment"。

---

## 2. 三臂协议（完全相同，仅 supervision_mode 不同）

| 项 | 设置 |
|---|---|
| stage-1 起点 | `outputs/t2_pretrain/best.pt`（同一 checkpoint，同 seed） |
| optimizer | head Adam lr 3e-4；repr（host+Gamma）Adam lr 1e-3 |
| KF estimator | `exact_replay`（same-window two-pass，λ=0.088） |
| macro group | `kf_group_batches=56`（与主实验 5-seed exact_replay 相同的 repr cadence） |
| 关窗门槛 | `kf_min_abs=896` |
| 采样 | bs 200, n_layer 3, n_degree 5, n_epoch 20, patience 10, trace_roots 32, evenly_spaced |
| 其余 | ridge_eps 1e-3, sketch_dim 64, rpbe_seed 0 |

### 关于 896 vs 主实验 1024（协议声明）

主实验的 `kf_min_abs=1024` 保留不变。结构三臂强制**共同的两未来合法 cut 交集**，
实际每 macro-group 产生的 unique trees 约 1362（seed0 实测），**并非** 896。
896 仅是允许该窗口通过的下限（放 3% 余量防止 seed 抖动），实际窗口样本数在
每 run 的 summary/metrics 中记录。因此：

> Structural ablations retain the main-run 56-batch update cadence. Because
> the common two-future-valid intersection yields ≈994–1362 unique trees per
> group, the acceptance floor is set to 896; actual per-window sample counts
> are reported. 896 is the closing floor under the stricter shared cut set,
> not the estimator's sample count.

---

## 3. 2obs_mispaired 的受约束实现（完美匹配）

mispaired 在 (tau, role-of-Y2, 自适应粗时间桶) 内对 horizon-2 事件求**二分图
完美匹配**，约束：

1. donor ≠ receiver（每 cut 收到的是别的 cut 的 Y2）；
2. 分配的 Y2 严格晚于 receiver 的 cut time（合法未来）；
3. 整体一一置换 → **Y2 multiset 精确保持**（与 aligned 逐事件相等）。

匹配不存在的桶**从三臂共同删除**（surviving cut 集三臂一致）。置换由
`(rpbe_seed, batch_seed, bucket)` 确定性决定，exact replay 两遍恢复同一置换。

单元测试断言（`test/test_rpbe_structural.py`）：
- aligned vs mispaired Y2 multiset **Counter 相等**；
- mispaired `unchanged_fraction == 0`（每 cut 的 Y2 都变）；
- 每分配 Y2 `outcome_time > 接收 cut time`；
- 三臂 cut-set 一致（含桶删除后）。

---

## 4. 启动门禁结果（seed0，1 macro-group × 三臂）

`python -m scripts.ablation_gate --seed 0 --group-batches 56 --min-trees 896`

| 门禁项 | 结果 |
|---|---|
| 每 tau M_unique_trees | layer1/layer2 均 **1362**（记录实际值） |
| below_threshold_groups | 0（窗口全部关闭） |
| 三臂 cut-set hash | 一致 `1afcc1…` |
| 三臂 Y1 multiset hash | 一致 `996196…` |
| aligned vs mispaired Y2 hash | 一致 `be4e3c…` |
| mispaired unchanged_fraction | **0**（5982/5982 全变） |
| 非法（非未来）Y2 分配 | 0 |
| replay planned/matched | 2724 / 2724（missing 0） |
| repr aux 梯度 | 有限非零（≈5.40） |

训练期间 fail-fast：任一 macro-group 任一 tau 关窗低于 896 → `RuntimeError`
终止（`--kf-fail-below-threshold`），杜绝静默丢窗口。全量 run 未触发。

---

## 5. 下游任务结果（每 arm 自身 val-AUC 选择 checkpoint）

（待补充训练完成后填写最终表）

---

## 6. 正确解读与限制（Part III.7）

- §5 是每 arm **自身** checkpoint 的下游任务指标，**不可**作为决定性结构对比。
- 决定性对比须用**共同 held-out aligned-2Obs audit**（相同审计目标评估三臂，
  见规格 Part III.7）——尚未实现为脚本，另行交付。
- 1obs 与 2obs 的 cut 交集相同（见门禁），因此任务差异不来自 cut 数量。

---

## 7. 运行命令 / 输出目录

```
bash run_structural_abl.sh --seeds 0 1 2 3 4 --outroot struct_abl --parallel 3
# 补充：seed0/1obs 缺失臂 + seed2 三臂重跑
bash run_structural_abl.sh --seeds 0 2 --outroot struct_abl --parallel 3
# 门禁
python -m scripts.ablation_gate --data wikipedia --data-dir old/processed_tgn_data \
    --seed 0 --gpu 0 --pretrained-checkpoint outputs/t2_pretrain/best.pt \
    --group-batches 56 --min-trees 896 --hash-batches 120
# 汇总
python -m scripts.summarize_struct_abl --outroot struct_abl --seeds "0 1 2 3 4"
```

输出：`outputs/struct_abl/seed{N}/{1obs,2obs_aligned,2obs_mispaired}/`
（config.json / summary.json / metrics.jsonl / monitor/）；门禁：
`outputs/struct_abl_v2/gate/ablation_gate.json`。

---

## 8. 失败 / skip / NaN / protocol deviation 记录

| 项 | 记录 |
|---|---|
| seed0/1obs | 首轮全量未产出（目录缺失）→ 已补跑 |
| seed2 | 按用户要求三臂重跑（结果并入最终表） |
| 协议 deviation | `kf_min_abs` 主实验 1024 → 结构臂 896（§2 说明）；`kf_group_batches=56` 与主实验一致 |
| motivation probe（Part II） | 按用户决定弃用新建的多输出回归 probe；现象审计改用仓库原 `audit_pir`（root label 口径），其结果因正例稀疏估计失败不计入机制结论（详见 §0 备注） |

### §0 备注：audit_pir 退休

`audit_pir.py` 全量结果（seed0 task_only）R0<RZ、I_available<0 不可作结论：
该脚本存在 calibration 未用独立集 / Platt logit 输入错 / 三 probe 容量不一致 /
bootstrap 用 set 丢重复 等实现缺陷，root-label PIR 在 wiki 0.14% 正例下估计失败。
主 loss 不因此失效（P 含 counterpart/role/Δt/horizon/path/query-type，非仅 0/1 标签）。

---

## 9. Commit / 复现索引

（待补：训练与结果对应的 feature_abl commit hash）
