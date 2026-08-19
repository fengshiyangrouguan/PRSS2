# Predictive Relation-State Sheaf（PRSS）实现

这个目录是 `PRSS_method_spec_v1_1_flexible_dims.md` 的独立实现包。它不是普通 TGN 预训练后的压缩器：PRSS 在训练第 0 步就安装在每个递归接口上，父节点从始至终只接收固定宿主宽度的商状态 `z = R_tau h`。

## 已实现的规格约束

- `k_tau` 只从宿主接口读取；TGN 中等于原始 embedding width，Wikipedia 官方配置下是 172。PRSS 核心没有固定 32 维或其他全局商宽度。
- nonlinear candidate lift 产生 `d_tau >= k_tau` 的候选状态，投影矩阵始终为 `R_tau: k_tau x d_tau`，行正交。
- training-only outside encoder 严格排除当前节点候选 `h_v`，只使用根场景信息、父局部信息、关系/时间和兄弟分支摘要。
- conditional reader 生成 `B(C)`，响应对候选状态保持线性；reader 的 Gram 使用 EMA 更新，周期性 `eigh` 取每个接口自己的 top-`k_tau`。
- 谱更新不参与普通 backward；更新后做 orthogonal Procrustes 对齐，减小同一子空间内的坐标跳变。
- 主损失严格为 task、structured response 和 normalized spectral tail 三项。更强 unrestricted reader 只作诊断，用独立 optimizer，不能反向影响主模型。
- validation/test 冻结 Gram 与 SVD；inference 关闭 trace、outside encoder 和 reader，只走宿主模型与 `z=Rh`。
- TGN 递归适配器包装官方 `GraphEmbedding` 聚合器，不复制或替换其 attention/aggregation 逻辑；每个子调用返回父节点前立即变成宿主宽度商状态。
- Wikipedia CSV 的前四列按 `user_id,item_id,timestamp,state_label` 读取，后续所有列原样作为每次 interaction 的 edge feature；预处理不会把 172 维 edge feature 错当节点静态 feature，也不静默写死 172。

## 目录

- `prss/config.py`：逐接口维度契约及超参数校验。
- `prss/lift.py`：非线性 lift、线性/恒等消融。
- `prss/outside_context.py`：training-only outside 上下文递推。
- `prss/reader.py`：conditional matrix reader 和 unrestricted diagnostic reader。
- `prss/spectral.py`：EMA Gram、top-k eig、Procrustes、谱与子空间诊断。
- `prss/system.py`：神经模块和 block-coordinate 谱更新的统一入口。
- `prss/recursive.py`：宿主无关的 compressed-first 树递归执行器。
- `prss/integrations/tgn.py`：官方 TGN 递归接口和链路响应场景桥接。
- `prss/ablations.py`：七个具名、可执行的 method/ablation policy。
- `experiments/synthetic_tree.py`：规格要求的已知子空间 `D,E -> A -> Y` 识别测试。
- `experiments/preprocess_jodie.py`：严格、两遍、低内存的 JODIE CSV 到官方 TGN 文件转换器。
- `experiments/train_tgn_prss.py`：Wikipedia/UCI/Enron 等 TGN 格式自然数据训练入口。
- `prss/monitoring.py`：数学不变量、机制证据、梯度/状态健康和 TensorBoard 监控。
- `experiments/compare_prss_runs.py`：matched-seed 效应、bootstrap CI 与证据 gate。
- `MONITORING.md`：完整监控指标、告警语义和结论边界。

## 本地验收

```bash
cd PRSS-method
python -m pytest -q
python experiments/synthetic_tree.py --output outputs/synthetic_tree.json
```

当前固定 seed 的 Phase 2 验收结果保存在 `outputs/synthetic_tree.json`：PRSS 恢复出的最大主角度为 `0.00256 rad`，真实三维预测子空间覆盖率为 `0.9999968`。这是实现识别测试，不是 Wikipedia 科学证据。

## AutoDL：从原始 Wikipedia CSV 到完整训练

推荐上传后的布局：

```text
/tmp/PRSS/
  wikipedia.csv                  # JODIE 原始真实交互 CSV
  tgn/                           # 官方 TGN 源码（本工作区版本含可指定输入/输出的预处理器）
  PRSS-method/
```

然后运行：

```bash
cd /tmp/PRSS/PRSS-method
RAW_CSV=/tmp/PRSS/wikipedia.csv \
TGN_DIR=/tmp/PRSS/tgn \
PROCESSED_DIR=/tmp/PRSS/processed_tgn_data \
CANDIDATE_DIM=256 EPOCHS=50 \
bash run_autodl.sh
```

脚本先执行单元测试和 Phase 2，再在缺少处理文件时调用包内的严格预处理器，最后从随机初始化直接训练 `full` PRSS。预处理器逐行验证固定 feature 宽度、有限数和非递减时间，使用两遍扫描及 `.npy` memmap，避免把整个 172 维 CSV 复制成 Python 对象；输出 manifest 会记录实际 edge-feature 维数。它不会先跑 vanilla TGN，也不会做训练后的离线 SVD。`CANDIDATE_DIM` 必须大于等于宿主给出的 `k_tau`；Wikipedia 的 `256 -> 172` 是真正的维度压缩。

若 TGN 和数据不在推荐目录，可只通过 `TGN_DIR`、`RAW_CSV`、`PROCESSED_DIR`、`OUTPUT_ROOT` 环境变量改路径，无需改代码。

## 完整消融

```bash
cd /tmp/PRSS/PRSS-method
TGN_DIR=/tmp/PRSS/tgn \
PROCESSED_DIR=/tmp/PRSS/processed_tgn_data \
EPOCHS=50 SEEDS="0 1 2" \
bash run_ablations_autodl.sh
```

七个变体含义如下：

| variant | 实际训练行为 |
|---|---|
| `fixed_random` | 保留相同 response supervision，但固定随机行正交 `R`，不更新投影、不用谱损失 |
| `pca` | 保留相同 response supervision，用训练候选状态协方差更新 `R`，不是 reader Gram |
| `direct` | 保留相同 response supervision，把 `R` 注册成参数并由 task loss 直接学习 |
| `linear_reader_svd` | context 到 `B(C)` 为线性映射，再做 reader-Gram SVD |
| `no_nonlinear_lift` | 用单层线性 lift，保留 reader-Gram SVD 和两项辅助损失 |
| `neural_svd_no_spec` | 保留非线性 lift、conditional reader 与 SVD，去掉 spectral-tail loss |
| `full` | 完整 PRSS |

脚本按 `_SUCCESS.json` 断点续跑，不会覆盖已经完整结束的 seed。工程 smoke 可显式使用 `--max-train-interactions` 等参数，但这种截断结果不能作为方法证据。

## 输出和诊断

每次自然数据训练目录包含：

- `config.json`：宿主/候选维度、压缩率、变体的真实 policy、数据和训练参数。
- `metrics.jsonl`：逐 epoch task/response/spectral/unrestricted response、AP/AUC、response gap、全部接口谱诊断。
- `best_model.pt`：模型、PRSS 状态和 TGN memory/pending raw messages。
- `results.json`：最佳验证、测试指标、最终谱和 inference contract。
- `_SUCCESS.json`：仅在完整训练、最佳 checkpoint 恢复和测试均完成后生成。

每个 run 还会生成 `monitor/step_metrics.jsonl`、`epoch_metrics.jsonl`、`alerts.jsonl`、TensorBoard events、逐 epoch 的 `R/G/eigenvalues` 快照和九宫格 `mechanism_dashboard.png`。详细解释见 [MONITORING.md](MONITORING.md)。

谱诊断逐接口记录 eigenvalue spectrum、`energy@r`、`tail@r`、principal angles、projector distance、Gram 更新数和 SVD 更新数；训练日志还记录 `||B||_F`，用于识别 reader collapse。排名 `r` 由当前接口 `k_tau,d_tau` 推导，不使用全局常数。

规格给出的 `lambda_spec = 0, 0.01, 0.1, 0.5, 1.0` 敏感性实验可断点续跑：

```bash
TGN_DIR=/tmp/PRSS/tgn \
PROCESSED_DIR=/tmp/PRSS/processed_tgn_data \
bash run_lambda_sensitivity_autodl.sh
```

## 因果边界

对一条时间为 `t` 的正/负链路场景，宿主历史查询只允许 `< t` 的交互。outside 根 metadata 中的 role 只编码“source tree / candidate tree”，绝不编码“positive / negative”；正负 target 只进入 response loss。当前节点自己的候选状态不会进入它的 context。验证和测试不能更新 Gram、`R` 或 reader。

## 数据集范围和结果解释

`train_tgn_prss.py --data NAME --data-dir DIR` 可运行任何已按官方 TGN 文件格式生成的自然数据，包括 Wikipedia、UCI/CollegeMsg 或 Enron。UCI/Enron 尚未出现在当前工作区，因此这里只提供同一严格训练入口，不伪造 Phase 4 数字。

在当前机器上已经用真实 Wikipedia 处理数据做过受限 interaction 工程 smoke，验证了 172 维宿主输出、192 维候选、训练期 Gram/SVD 更新及推理期完全关闭 reader/outside；它不计入论文实验。完整规模与多 seed 消融应在 AutoDL 上运行后，才能判断规格第 30 节的自然数据性能验收是否成立。
