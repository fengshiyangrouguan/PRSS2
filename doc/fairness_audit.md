# Vanilla vs Ours 训练与推理流程公平性审计（UCI + Wikipedia）

- 审计日期：2026-09-13
- 代码基线：develop_UCI @ 781bb67（UCI 线）+ 同分支 main 基底（Wiki 线）
- 官方参照：BenchTemp `train_link_prediction.py` / `evaluation/evaluation.py`
  （服务器 `/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep/`）
- 结论速览：**Wiki 线评测完全同口径；UCI 线存在一处实质口径差异
  （官方 batch-mean AP/AUC vs 我们 global 单次计算），其余为协议设计内差异。**

---

## 1. 数据协议（两线一致 ✓）

| 项 | 官方 | 我们 | 一致 |
|---|---|---|---|
| split | 时间分位 70/15/15（train/val/test） | `uci_link.py:6` 同 70/15/15 分位 | ✓ |
| 负采样 | `RandEdgeSampler(full_data, seed=0/2)`，val seed=0、test seed=2，评测前 `reset_random_state()` | `uci_eval.py` val seed=0 / test seed=2，一次性生成同分布负例 | ✓（同 seed 同分布） |
| 负采样池 | 全流 dst 池（transductive 主协议） | 同 | ✓ |
| memory 协议 | 每 epoch 起始 `__init_memory__()`；val 从 train-end 状态评分；final test 前 reset + replay train + replay val | `uci_eval.evaluate_test` 同构（reset → replay train → replay val → test） | ✓ |
| 评测邻居查找器 | `full_ngh_finder`（全图） | `full_finder` 切换同构 | ✓ |
| 模型模式 | `model.eval()`（dropout 关） | `tgn.eval()` 同 | ✓ |
| 泄漏防护 | — | wiki 线 `delta_scale` 只取 train 时差（`train_jodie.py:290` 注释明示防泄漏）；BoundaryMaps 固定（maps_sha 校验） | ✓（我们更严） |

## 2. 评测协议

### 2.1 UCI Link Prediction —— **存在口径差异（需修复/声明）**

| | 官方 `eval_edge_prediction` | 我们 `uci_eval.evaluate_split` |
|---|---|---|
| 指标聚合 | **batch 级**：每个 batch（200 正 + 200 负）内算 AP/AUC，最后 `np.mean(val_ap)` | **global**：全部 batch 的正负分拼接后一次 `sklearn` AP/AUC |
| 差异性质 | 每 batch 独立排序域 | 全流单一排序域 |
| 影响方向 | 两者对同一模型分数不严格相等（AP/AUC 是排序域的非线性函数）；平衡数据下差异通常 <0.01，但**方向未实测** | |

**修复建议（三选一）**：
1. `uci_eval` 增加 `agg="batch_mean"` 模式，用同口径重算 vanilla 的官方 checkpoint；
2. 或反之用 global 口径重算我们的数字；
3. 论文表格脚注声明口径差异并给出同口径对照的实测差。

### 2.2 Wiki Node Classification —— 完全一致 ✓

官方 `eval_node_classification`：`pred_prob` 全流数组 → 一次 `roc_auc_score` / `average_precision_score`。
我们 `jodie_loop.evaluate_split`：`np.concatenate(labels/probs)` 后一次 `metric_bundle`。
**同口径（global）**，无差异。

## 3. 训练协议差异清单（设计内差异 + 利己性方向）

| 项 | vanilla（官方协议） | ours（我们协议） | 利己性方向 |
|---|---|---|---|
| 优化器 | 单 Adam 全参数，每 batch step（210 步/epoch） | 双优化器：head 每 batch、repr 每宏组（6 步/epoch）+ `--repr-lr 3×` 补偿 | **对 ours 有利**（但源自审阅方认可的 Eighth-review 设计） |
| 训练预算 | n_epoch 50、patience 3（实际 4-33ep 早停） | 50/80ep、patience 80（跑满不早停） | **对 ours 有利**（更多训练预算；官方 patience=3 是其文献协议） |
| 选模 | 每 epoch val（batch-mean AP）→ best | 每 epoch val（global ap_all）→ best@budget | 选模指标口径同 2.1 差异 |
| 学习率 | lr=1e-4 | lr=1e-4 + repr-lr=3e-4 | ours 有补偿 |
| RPBE 组件 | 无 | compressor + KF 窗口 + tree-wise 投影（κ=0.05） | 方法本身 |
| Wiki host | 冻结 host（官方 stage2）+ 只训 decoder | 联合微调 host + RPBE | **对 ours 有利**（更多可训参数）——但官方 stage2 冻结是文献协议，我们的协议内对照臂是 taskonly |

**设计原则**（非隐藏操作）：
- **vanilla = 官方文献对齐臂**：完全用官方协议（代码零改动副本），用于与文献数字（BenchTemp 0.8914±0.0138）对标；
- **taskonly = 协议内对照臂**：与 ours 同 runner、同 cadence、同预算、同评测器，仅去掉 RPBE aux——**这才是 ours 的公平对照**；
- 表格中 ours 对 vanilla 的差值反映"方法 + 协议"的联合差异，对 taskonly 的差值才是"方法本身"的增益。

## 4. RNG 与初始化

- 三臂均 `seed_all(seed)`（python/numpy/torch 同 seed）。
- ours/taskonly 同 runner：参数构造顺序一致、RNG 消耗路径一致（已验证 epoch 0 的 train_link_loss 起点几乎相同：1.0233 vs 1.0222）。
- vanilla（官方脚本）有自己的 RNG 路径——与我们的 runner 天然不同，但同为 `_seed_all` 起点。
- 初始化公平性：ours 与 taskonly 严格同源；vanilla 独立但协议不同源（文献对齐臂，不参与协议内配对）。

## 5. 推理流程（test 评测路径）

| 步骤 | 官方 vanilla | 我们 ours/taskonly |
|---|---|---|
| checkpoint 选择 | val batch-mean AP 的 best（训练内） | val global ap_all 的 best（训练内） |
| test 前 memory | 训练结束态？→ 官方 test 前 reset 并 replay（`main` 中 val 后 restore） | `evaluate_test`：reset → replay train → replay val → score test ✓ |
| 打分 | batch-mean AP/AUC（4 协议变体，主表用 Transductive） | global AP/AUC（等价 Transductive） |
| 邻域 | full finder | full finder ✓ |

**唯一推理口径差异仍为 2.1 的 batch-mean vs global。**

## 6. 结论与行动项

1. **Wiki 线：公平**。评测同口径、数据同协议；训练差异全部是"官方 stage2 冻结 vs 我们协议"的设计内差异。
2. **UCI 线：一处实质差异**——官方 batch-mean 指标 vs 我们 global 指标。必须统一口径后出正式对比表。
3. **无"暗箱利己"**：所有差异都可在代码中逐项溯源；有利方向两边都有（训练预算/补偿对 ours 有利；官方早停短训练对 vanilla 不利但那是官方文献协议原样保留）。
4. 建议行动：
   a. `uci_eval` 加 `agg` 参数（batch_mean / global），对 vanilla 的官方 checkpoint 用两口径重算，量化差异；
   b. 正式表格 ours-vs-taskonly 用同口径（已是同评测器 global ✓）；
   c. ours-vs-vanilla 的表格脚注写明"vanilla 为官方协议口径（batch-mean + patience 3）"。

## 附录：证据位置

- 官方 batch-mean：`benchtemp/.../evaluation/evaluation.py::eval_edge_prediction`（`np.mean(val_ap)`）
- 官方 node-class global：同文件 `eval_node_classification`（全流一次 sklearn）
- 我们 UCI：`src/rpbe/training/uci_eval.py::evaluate_split`（concatenate 后一次 sklearn）
- 我们 Wiki：`src/rpbe/training/jodie_loop.py::evaluate_split`（concatenate 后 metric_bundle）
- split/负采样：`src/rpbe/data/uci_link.py`（quantile [0.70, 0.85]）
- 选模/早停：`scripts/official_uci_vanilla.py`（EarlyStopMonitor patience=3、val_ap）vs `scripts/train_uci_link.py`（ap_all、budget）
- 双优化器/补偿：`scripts/train_uci_link.py`（opt_split head_lr/repr_lr）
