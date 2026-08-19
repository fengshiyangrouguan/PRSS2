# Wikipedia × 官方 TGN：预测混叠与低维可压缩性诊断

这套代码只验证两个先于新方法存在的命题：冻结的 TGN 递归状态是否遗漏仍影响未来的历史信息，以及这部分 future-relevant information 是否呈现低秩结构。`tgn/method/` 故意为空；这里没有 synthetic toy，也没有实现或评估我们的最终方法。

## 已钉死的数据与任务语义

- 原始文件是 `wikipedia.csv`，已实测为 157,474 条自然交互；每行前四列为 `user_id,item_id,timestamp,state_label`，随后正好 172 维 interaction/edge feature。
- 预处理使用官方 TGN 的 bipartite reindex：0 保留为 padding，用户为 1–8227，页面为 8228–9227，interaction index 为 1–157474。
- 主任务完全沿用官方 self-supervised future-link prediction：每个真实 interaction 是正样本，并从页面节点集合中采一个随机负 destination。`state_label` 不是这个 self-supervised 任务的 Y。
- 当前正边的 172 维 feature 不进入当下 link decoder，也不会泄漏进 probe context；它只按官方逻辑用于该事件之后的 memory message。`X^(K)` 只读取查询时间之前的历史边。
- 数据划分、inductive node holdout、TGN memory/no-memory、AP/AUC、`n_layer` 与 `n_degree` 都沿用官方仓库。官方基线固定在 commit `d55bbe678acabb9fc3879c408fd1f2e15919667c`。

## 代码结构

```text
PRSS/
  wikipedia.csv
  processed_tgn_data/             # 本机已完成的官方格式预处理结果
  tgn/                            # twitter-research/tgn 官方代码 + 最小 instrumentation
    modules/embedding_module.py   # 只读 observer；不开启时没有任何额外路径
    diagnostics/
      extract_cuts.py             # 冻结权重、从零按时间重放并导出 h/C/X/Y
      conditional_residual.py     # M0、MK、local permutation、paired bootstrap
      collision_analysis.py       # (h,C) 近邻匹配与 future divergence
      predictive_rank.py          # cross-fitted G、predictive SVD、held-out RRR
      summarize_baselines.py      # L=1/2/3 × memory/no-memory AP/AUC
    method/README.md              # 明确说明方法暂不实现
    run_autodl.sh                 # 完整 AutoDL 入口
```

## AutoDL 完整运行

把整个 `PRSS` 放到 AutoDL 的 `/tmp` 下，例如 `/tmp/PRSS`。进入 TGN 目录：

```bash
cd /tmp/PRSS/tgn
pip install -r requirements_diagnostics.txt
bash run_autodl.sh
```

脚本默认：

- 对 memory/no-memory、`L=1,2,3`、seed 0/1/2 分别训练官方 TGN；
- 以 seed 0 的同一个冻结 `L=1` 状态做 `K=1,2,3` conditional residual curve；
- 分别跑 `structure+time`、`structure+time+edge`、全部真实 pre-aggregation 信息三版；
- 对 full TGN 和 TGN-no-memory 都做 collision 与 predictive SVD/RRR；
- 输出到 `/tmp/PRSS/outputs_tgn_diagnostics`。

可通过环境变量改位置或训练轮数：

```bash
RAW_CSV=/tmp/PRSS/wikipedia.csv \
DATA_DIR=/tmp/PRSS/processed_tgn_data \
OUTPUT_DIR=/tmp/PRSS/outputs_tgn_diagnostics \
N_EPOCH=50 SEEDS="0 1 2" GPU=0 \
bash run_autodl.sh
```

不要把 `N_EPOCH=1`、小样本 cap 或单 seed 的工程 smoke 输出当研究结果。正式结论至少使用默认完整数据、正常 early stopping 与多个 baseline seeds。

## 为什么导出必须按时间重放

官方 `state_dict` 会保存 memory tensor 和 last-update tensor，但不会保存 Python 层的 pending raw-message queues。直接加载 checkpoint 后随机访问 test 行，会得到不完整且时间上错误的状态。

`extract_cuts.py` 因此只加载学习到的参数，随后把 memory 清零，并用最终冻结权重按以下顺序重放：

1. train interactions（训练 neighbor finder）；
2. validation interactions（full neighbor finder）；
3. test interactions（full neighbor finder）。

batch size 强制与训练 manifest 一致，因为官方 TGN 的 memory update 频率依赖 batch 边界。每个样本的 `h,C,X` 都在当前事件写入 memory 之前取得。

## `h`、`C`、`X^(K)` 的精确定义

- `h`：冻结 TGN 对 source 在查询时刻输出的最终 temporal embedding；正负 continuation 共用完全相同的 `h`。
- `C`：候选 destination 的冻结 temporal embedding，加标准化 query time。正样本和负样本只有 continuation 不同。
- `X_structure^(K)`：每 hop 的 frontier/edge/unique-neighbor/branching 和 `log(1+Δt)` 固定统计。
- `X_edge^(K)`：每 hop 历史 edge feature 的 mean、standard deviation、most-recent 三组完整 172 维统计。
- `X_state^(K)`：官方 aggregate 之前的 layer-zero lower states；每 hop 保存 frontier mean、neighbor mean、neighbor std。full TGN 使用 raw node feature + 当前 effective memory；no-memory 使用 raw node feature。

官方 `GraphEmbedding` 在递归到 neighbor 时继续使用原始 query timestamp，而不是父边 timestamp。history extractor 精确复刻了这个行为。padding node 不进入描述统计。

数据按 interaction pair 保存：一个 `X/h` 对应 observed destination 和 sampled negative 两个 continuation。这样不会把大历史 tensor 复制两份，也保证 paired bootstrap 的抽样单位就是 interaction，而不是把相关的正负行拆开。

## 四项主诊断

### 1. Conditional residual

`conditional_residual.py` 训练相同的一层浅 probe：

- `M0: P(Y|h,C)`；
- `MK: P(Y|h,C,X^(K))`；
- `MK-shuffle`：相同输入宽度、隐藏宽度、网络与选中的 weight decay，只把 pair-level X 在相近时间分位 × 同为 user 类型下的历史活跃度分位内做 derangement。

weight decay 只由 validation NLL 选择；test 在模型选择结束后评一次。主量为 pair-level

```text
ΔNLL_K = NLL(M0) - NLL(MK)
```

并报告 paired bootstrap 95% CI。期望 `ΔNLL_K>0` 且 CI 不跨 0，同时 shuffle 增益消失。

### 2. Near-collision

在标准化 `(h,C)` 空间找来自不同 interaction history 的最近邻，分别报告 `d_h`、`d_C` 与 rich test probe 的 `|p_i-p_j|`。`collision_summary.json` 对多组 `epsilon_h,epsilon_C,tau` 给出 collision-rate curve，不依赖单一阈值。

### 3. Predictive SVD

共同 continuation contexts 来自 train 的正负 context。训练 history 的 signature 由 pair-level K-fold cross-fitted rich probes 产生；validation/test signature 由只在 train 拟合并经 validation 选择的 full probe 产生。矩阵 `G` 先按 train column mean 中心化，避免总体正例率形成虚假的 rank-1 结论；rank basis 只在 `G_train` 上拟合，报告 validation/test reconstruction error。

### 4. Reduced-rank regression

RRR 输入是未经过当前 `h` 压缩的 `X`，目标是 cross-fitted/held-out future signature。`X_train` 先标准化并白化；ridge 只看 validation；随后对白化空间的回归系数做 rank constraint，报告 rank 到 train/validation/test relative MSE 的曲线。默认 whitening PCA 为 256 维并把解释方差写进结果；设 `--whiten-rank 0` 可用完整数值秩。

## 关键输出

```text
outputs_tgn_diagnostics/
  depth_support/tgn_depth_summary.json
  depth_support/tgn_depth.png
  memory/
    cuts/manifest.json
    conditional/conditional_residual.json
    conditional/conditional_residual.csv
    collision/collision_summary.json
    collision/collision_scatter.png
    predictive_rank/predictive_rank.json
    predictive_rank/predictive_rank.png
  no_memory/...
```

“大方向成功”必须同时满足：深历史的 `ΔNLL` 为正且 CI 不跨 0；local permutation 后增益显著消失；很小 `d_h,d_C` 下仍存在非平凡 future divergence；predictive SVD 与 held-out RRR 都出现 `r << raw/signature dimension` 的稳定 elbow。普通 TGN 的 AP/AUC-vs-depth 只作为 supporting evidence，不代替上述主证据。

## 已完成的工程校验

- 完整读取并预处理真实 Wikipedia CSV，核对 172 维、行数、ID 区间和时间单调性；
- 在真实 Wikipedia batch 上比较只读 observer 开启/关闭，TGN 输出逐元素完全一致（最大差 0）；
- full-memory 与 no-memory 各跑过真实数据 1-epoch 工程 smoke；
- 两种模型都完成冻结权重的全时间重放和 cut export；
- conditional residual、collision、cross-fitted predictive SVD/RRR 均在真实数据的 capped smoke 上端到端运行并生成合法输出。

这些 smoke 只证明代码路径可运行，不构成论文现象的统计证据。
