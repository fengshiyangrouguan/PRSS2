# PRSS 监控与方法证据协议

监控分成三层，不能互相替代：

1. **数学/工程不变量**：证明实现没有偏离 PRSS 定义。
2. **机制证据**：证明 reader、predictive spectrum 和递归商状态确实在工作。
3. **比较证据**：证明 full PRSS 在相同 seed、数据切分和宿主宽度下优于消融。

单次训练健康只能通过前两层的一部分，不能被报告为“方法有效”。最终结论由多 seed 配对报告给出。

## 每个监控点记录什么

### 训练与梯度

- task、structured response、unrestricted response、normalized spectral tail、total loss；
- lift、structured reader、outside encoder、direct projection、TGN host 的梯度 L2 norm；
- candidate state 的向量范数、逐坐标 batch 标准差和 finite fraction；
- reader `||B(C)||_F` 的 mean/std/min/max。

这些指标可以识别 loss 爆炸、lift/reader 没接入计算图、reader collapse、candidate collapse 和 NaN/Inf。

### Response quality

- structured matrix reader accuracy/NLL；
- unrestricted MLP reader accuracy/NLL；
- structured/unrestricted NLL ratio 与差值。

默认 warm-up 后 ratio 大于 2 触发 warning。它不终止训练，因为这可能是模型表达能力不足的科学发现，而不是实现错误；但有该告警时不能声称当前 lift 已充分线性化 future response。

### Predictive spectrum 与商空间

逐接口记录：

- 完整 eigenvalue spectrum；
- `energy@floor(k/4)`、`energy@floor(k/2)`、`energy@k`、`energy@min(2k,d)`；
- normalized spectral tail；
- Gram trace、effective rank、Gram symmetry residual；
- 当前部署 `R` 对**本 batch reader Gram**的 predictive-energy coverage；
- 当前统计量 Gram 上的 coverage；
- 相邻更新的 principal angles、projector distance；
- Gram 和 SVD 更新次数。

本 batch reader-Gram coverage 用于在线趋势；验证和测试还会在冻结 `R`、禁止更新 Gram/SVD 的条件下，对只读 response batches 计算 held-out predictive-energy coverage。多变体报告只比较这项 held-out coverage。PCA 自己的 covariance eigenvalue energy 不能冒充 predictive energy，因此两者在日志中分开命名。

### 强制不变量

以下情况默认立即终止训练：

- 任意被监控 loss、candidate 出现 NaN/Inf；
- 非 direct 变体的相对 `||R R^T-I||_F/sqrt(k)` 超阈值；
- EMA Gram 失去对称性。

以下情况记录 warning：

- reader/candidate collapse；
- structured reader 明显落后 unrestricted reader；
- 应收到梯度的模块没有梯度；
- 相邻 predictive subspace 跳变过大。

可以用 `--no-fail-on-monitor-error` 暂时收集故障现场，但这种 run 的健康状态不会通过证据门槛。

### 推理隔离实测

最终测试后会额外执行一个标准 inference probe，并实际比较前后计数及输出 tensor：

- outside trace 是否被创建；
- Gram 更新数是否改变；
- SVD 更新数是否改变；
- source/destination/negative 的真实输出宽度是否等于宿主 `k`。

验证/测试中的 response diagnostic 可以只读调用 outside reader，但不得更新 Gram/SVD；这与标准部署 inference 的完全关闭分别记录。

## 监控产物

每个 run 的 `monitor/` 目录包含：

```text
monitor/
  monitor_config.json
  step_metrics.jsonl
  epoch_metrics.jsonl
  alerts.jsonl
  monitor_summary.json
  projection_snapshots/epoch_XXXX.pt
  tensorboard/events.out.tfevents...
  mechanism_dashboard.png
```

`projection_snapshots` 同时保存每个接口的 `R`、EMA Gram 和 eigenvalues，可离线复算 projector drift 或审计 checkpoint。

## 在线查看

```bash
tensorboard \
  --logdir /tmp/PRSS/PRSS-method/outputs/autodl/wikipedia/full/monitor/tensorboard \
  --bind_all --port 6006
```

AutoDL 中再把 6006 端口映射出来。默认每 50 step 做一次较重的矩阵监控；可用 `--monitor-every 20` 提高时间分辨率，或在性能受限时调大。每个 epoch 仍会固定保存投影快照和验证指标。

AutoDL 脚本遇到没有 `_SUCCESS.json` 的旧目录时不会把新日志追加进去，而会先将其改名为 `_incomplete_时间戳`。故障现场得到保留，同时新的 seed 证据不会被旧 step 混入。

静态九宫格证据图可重新生成：

```bash
python experiments/render_monitor_report.py \
  --run-dir outputs/autodl/wikipedia/full
```

## 多 seed 消融结论

`run_ablations_autodl.sh` 完成后自动生成：

```text
outputs/ablations/wikipedia/evidence_report.json
outputs/ablations/wikipedia/evidence_report.md
```

报告对相同 seed 计算：

- `full - fixed_random` 的 ΔAP、ΔAUC、Δpredictive-energy coverage；
- `full - PCA` 的对应差值；
- `full - direct learned projection` 的对应差值；
- paired bootstrap 95% CI、win rate 和 matched-seed 数；
- full 的 Gram/SVD 更新、monitor health、推理隔离和维度匹配 gate。

结论状态严格分为：

- `supported_by_completed_paired_ablations`：必要 run 齐全、机制 gate 通过，且三个主要 ΔAP 的 95% CI 下界均大于 0；
- `promising_but_ci_crosses_zero`：均值为正，但区间仍跨 0；
- `incomplete_evidence_missing_runs`：不能因缺失消融而下结论；
- `not_supported_by_current_runs`：当前结果不支持方法优势。

至少使用 3 个 matched seeds，正式实验建议 5 个以上。原始 vanilla TGN 仍须单独放入论文 task-performance 表；full 对 direct 的比较负责排除“更大 lift/参数量”解释，不能替代 vanilla 性能基线。
