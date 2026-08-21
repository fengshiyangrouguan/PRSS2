# A0 使用文档（用户 + Claude Code 双读者版）

> 用途：A0（条件矩预测商训练线）的操作手册与交接文档。上半部分是给人看的白话指南；下半部分是给 Claude Code 的精确协议与坑清单。
> 更新日期：2026-08-21
> 理论依据：《递归预测商_检索审计与A0算法_2026-08-20.md》（仓库根目录）

---

# 第一部分

## 1. A0 是什么（一句话）

A0 是 PRSS 项目的新一代方法：**从"切开的计算树"上被动学习一个 r 维的预测状态**——状态要能预测未来结果，还要能自己在树上递归更新（不需要宿主的高维 memory）。它分四阶段：

```
阶段 A：校准 R —— 学"历史里哪些方向最能预测未来"（条件未来矩 + rank-r SVD，一次性冻结）
阶段 B：校准 B̂ —— 学"r 维状态怎么递归更新"（凸 ridge，一次性冻结）
阶段 C：审计 —— 五类失败证书（G0 数据可识别 / G1 可压缩 / G2 闭合 / G3 深树稳定 / G4 真实收益）
阶段 D：对比 —— A0 读取头 vs 宿主 decoder 同一次前向双头训练 + 部署审计（偏差/吞吐/内存）
```

与旧方法（v2.0 spectral 压缩器）的本质区别：**旧 R 学"输出变化大的方向"（无监督），新 R 学"对未来结果有用的方向"（判别式），并且 A0 在 r 维上递归而不是每步解压回宿主维度。**

## 2. 代码地图

```
src/prss/a0/                    # A0 核心（新包）
├── probes.py                   # 固定探针：a(C) 上下文随机投影、φ_Y 标签 one-hot、根标签传播
├── quotient.py                 # 阶段 A：加权矩累积 + 白化 rank-r SVD → R 冻结
├── operators.py                # 阶段 B：χ 交互特征（meanpool / TensorSketch）+ ridge → B̂ 冻结
├── weights.py                  # 阶段 A 可选：密度比重要性权重 + ESS
├── audit.py                    # 阶段 C：残差/闭合/增益/leverage/门判定（G0-G4）
src/prss/training/a0_loop.py    # 四阶段编排 + 双头训练 + 递归部署审计
scripts/train_a0.py             # 独立入口（不动 train.py / train_jodie.py / PRSSCore）
scripts/run_a0_synthetic.py     # 5 个合成实验 + 图（本机可跑）
test/test_a0_quotient.py        # 27 项单测（纯 torch，本机）
test/test_a0_loop.py            # 2 项端到端冒烟（云端 GPU，numpy bridge）
```

## 3. 快速上手：合成实验（本机，无 GPU 依赖）

```powershell
python -m scripts.run_a0_synthetic            # 全部 5 个实验
python -m scripts.run_a0_synthetic --exp mod8 # 单个
```

输出：数据 `outputs/a0_synthetic/*.json`（含 `_gate_matrix.json` 审计门矩阵）、图 `outputs/plots/a0_synth_*.png`。

**合成实验结果锚点**（论文素材）：

| 实验 | 关键数字 | 说明了什么 |
|---|---|---|
| mod-8 深度扫描 | 唯一可分状态数 2→4→8→8；一步基线坍缩 | 多步未来条件矩 > 一步监督 |
| XOR | 带上下文 dist=1.0000 / 无上下文 0.0022 | 必须条件于上下文 |
| shared-DAG | p_share=1.0 时条件数 6.6e12 恰在 OOD 误差 6.13 处 | G2 审计门能预警外推失败 |
| 路径增益 | 局部闭合同水平（0.865/0.862）根误差 1.0/9.5；式(14) 界全覆盖 | 深树误差由增益决定 |
| 圆周 | r=64 残差 0.736 vs 秩 3 系统 1e-6 | 线性压缩有硬边界 |

## 4. 真实数据实验（云端 GPU）

```bash
# 云端（ssh -p 13743 root@connect.weste.seetacloud.com）
cd /root/autodl-tmp/PRSS2
/root/miniconda3/bin/python -m scripts.train_a0 \
    -d wikipedia \
    --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/pretrained/tgn-wikipedia.pth \
    --output outputs/a0/r32_seed0 \
    --r 32 --d-context 32 --n-epoch 10 --seed 0 \
    --gate-mode stop --g0-min-ess-frac 0.2
```

**常用参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--r` | 32 | 压缩维数（预算） |
| `--d-context` | 32 | 上下文探针宽度 |
| `--frac-a/-b/-c` | 0.2 | train 流内三阶段窗口比例（D 默认用全 train，`--d-slice-only` 可选严格版） |
| `--use-weights` | 关 | 重要性权重 + 真 G0（开时 A 窗口前 20% 估密度比） |
| `--chi-mode` | meanpool | `sketch` = TensorSketch 交互特征（配 `--sketch-s`） |
| `--deploy-events` | 0 | 部署审计事件数（>0 开启 r 维递归部署测量） |
| `--gate-mode` | report | `stop` + 阈值才真正拦截 |
| `--g0..--g4` | None | 各门阈值（`--g0-min-ess-frac`/`--g1-max-rank-tail`/`--g2-max-closure-resid`/`--g3-max-gain-product`/`--g4-min-auc-delta`） |
| `--max-train/val/test` | 0 | 冒烟截断（0=不限） |

**输出四件套**（与 JODIE 线同约定）：`config.json`（一次性）、`metrics.jsonl`（D 阶段每 epoch 一行）、`summary.json`（含 audit 表 + 部署审计 + 双头指标）、`_SUCCESS.json`（哨兵，status=complete/stopped + stop_reason）。

## 5. 全部入口参数速查（五入口全量）

### 5.1 `scripts/train_a0.py`（A0 四阶段线）

| 参数 | 默认 | 说明 |
|---|---|---|
| `-d/--data` | wikipedia | 数据集名（ml_{name}.csv 三件套） |
| `--data-dir` | 必填 | 数据目录（本机 `old/processed_tgn_data`） |
| `--pretrained-checkpoint` | 必填 | 预训练 TGN 权重（云端 `outputs/pretrained/tgn-wikipedia.pth`） |
| `--output` | 必填 | 输出目录（四件套） |
| `--seed` / `--gpu` | 0 / 0 | |
| `--bs` | 100 | 批大小（时间序连续切片） |
| `--n-degree` | 10 | 邻居数（决定 preagg 宽度） |
| `--n-head` / `--n-layer` | 2 / 2 | 宿主注意力头/层数（必须与预训练一致） |
| `--drop-out` / `--message-dim` / `--memory-dim` | 0.1 / 100 / 172 | 宿主超参（必须与预训练一致） |
| `--r` | 32 | **压缩维数（固定预算）** |
| `--d-context` | 32 | 上下文探针宽度（u 宽 = 2×d_context） |
| `--lambda-x` | 1e-4 | 阶段 A 白化 ridge |
| `--lambda-gamma` | 1e-3 | 阶段 B 交互设计 ridge |
| `--lambda-audit` | 1e-3 | 阶段 C 审计 ridge |
| `--chi-mode` | meanpool | `sketch` = TensorSketch 交互特征 |
| `--sketch-s` | 64 | sketch 模式桶数（χ 宽 = 3s） |
| `--deploy-events` | 0 | >0 开启 r 维递归部署审计（前 N 事件） |
| `--frac-a/--frac-b/--frac-c` | 0.2 | train 流内 A/B/C 窗口比例（和必须 <1） |
| `--d-slice-only` | 关 | D 阶段只用 C 后剩余行（默认全 train） |
| `--trace-roots` | 16 | 每 batch 追迹根数（A/B/C 采样密度） |
| `--trace-mode` | evenly_spaced | 根选择：`evenly_spaced` / `positive_first` / `off` |
| `--use-weights` | 关 | 重要性权重 + 真 G0（A 窗口前 20% 估密度比） |
| `--weight-calib-frac` | 0.2 | 权重校准占 A 窗口比例 |
| `--w-min` / `--w-max` | 0.1 / 10.0 | 权重截断区间 |
| `--lr` | 3e-4 | D 阶段双头学习率（各自 Adam） |
| `--n-epoch` / `--patience` | 10 / 3 | D 阶段轮数与早停耐心（双头独立早停） |
| `--selection-metric` | auc | 早停指标：`auc` / `ap` |
| `--gate-mode` | report | `stop` = 门失败即终止（不换求解器回环） |
| `--g0-min-ess-frac` | None | G0：ESS 占比下限（如 0.2） |
| `--g1-max-rank-tail` | None | G1：谱尾上限 |
| `--g2-max-closure-resid` | None | G2：闭合残差上限 |
| `--g3-max-gain-product` | None | G3：路径增益积上限 |
| `--g4-min-auc-delta` | None | G4：A0−baseline AUC 差下限 |
| `--monitor-every` / `--no-fail-on-monitor-error` | 50 / 关 | 监控频率 / 容忍监控报错 |
| `--max-train/--max-val/--max-test` | 0 | 冒烟截断（0=不限） |

### 5.2 `scripts/train_jodie.py`（JODIE 节点分类线，旧谱压缩 5 变体）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--variant` | spectral | `vanilla` / `random` / `pca` / `direct` / `spectral`（vanilla = 纯官方宿主无 PRSS） |
| `-d/--data` | wikipedia | `wikipedia` / `reddit`（mooc/lastfm 标签全 0 跑不了） |
| `--data-dir` / `--pretrained-checkpoint` / `--output` | 必填 | 同上 |
| `--seed` / `--gpu` / `--bs` | 0 / 0 / 100 | |
| `--n-degree` / `--n-head` / `--n-layer` | 10 / 2 / 2 | 宿主（与预训练一致；官方 reddit checkpoint 是 L=1） |
| `--n-epoch` / `--lr` / `--patience` | 10 / 3e-4 / 3 | 监督微调（BCE 无负采样，负例=自然 label 0 行） |
| `--drop-out` / `--message-dim` / `--memory-dim` | 0.1 / 100 / 172 | 宿主超参 |
| `--finetune-host` | 关 | 冻结宿主 = 官方协议；开 = 微调宿主 |
| `--selection-metric` | auc | 早停指标 `auc` / `ap` |
| `--candidate-dim` / `--candidate-hidden` | 256 / 128 | 候选加宽与 builder 隐宽 |
| `--context-dim` / `--reader-hidden` | 64 / 128 | outside 上下文与 reader 隐宽 |
| `--lambda-resp` / `--lambda-spec` | 1.0 / 0.1 | 响应/谱损失系数 |
| `--gram-ema` | 0.05 | 谱 Gram 指数移动平均系数 |
| `--spectral-warmup` / `--spectral-interval` | 200 / 200 | 谱求解热身/间隔（步） |
| `--spectral-step-size` | 0.25 | Grassmann 步长 |
| `--trace-roots` / `--trace-mode` | 8 / positive_first | 追迹根数/B1 钩子（`positive_first` / `evenly_spaced` / `off`） |
| `--no-early-stop` | 关 | B4 诊断钩子：跑满 n-epoch 不早停 |
| `--grad-clip` | 5.0 | 梯度裁剪 |
| `--monitor-every` / `--checkpoint-every` | 50 / 50 | 监控/滚动 checkpoint 频率（0=关） |
| `--resume-from` | 空 | 滚动 checkpoint 恢复（含 memory backup） |
| `--no-fail-on-monitor-error` / `--max-train/--max-val/--max-test` | 关 / 0 | |

### 5.3 `scripts/train.py`（TGB 链路预测线，5 变体）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | 空 | YAML 实验文件（提供默认值） |
| `--variant` | spectral | `vanilla` / `random` / `pca` / `direct` / `spectral` |
| `--dataset` | tgbl-wiki | TGB 数据集名 |
| `--seed` / `--gpu` / `--output` | 0 / 0 / 必填 | |
| `--epochs` / `--bs` / `--lr` / `--patience` | 30 / 200 / 1e-4 / 5 | 训练超参 |
| `--n-neighbors` | 10 | 采样邻居数 |
| `--mem-dim` / `--time-dim` / `--emb-dim` | 100 / 100 / 100 | 宿主 memory/时间/嵌入维度 |
| `--candidate-dim` / `--candidate-hidden` | 256 / 128 | 同上表 |
| `--context-dim` / `--reader-hidden` | 64 / 128 | 同上表 |
| `--lambda-resp` / `--lambda-spec` | 1.0 / 0.1 | 同上表 |
| `--gram-ema` | 0.05 | 同上表 |
| `--spectral-warmup` / `--spectral-interval` | 100 / 100 | 同上表（TGB 线默认 100） |
| `--spectral-step-size` / `--trace-roots` | 0.25 / 8 | 同上表 |
| `--freeze-host` | 关 | 冻结宿主模式（旧架构语义） |
| `--grad-clip` | **0.0（保持！）** | ⚠️ 此宿主梯度合法可达 ~1e9，开裁剪会冻死训练（loss 钉在 1.44） |
| `--monitor-every` / `--checkpoint-every` | 100 / 0 | 监控/滚动 checkpoint |
| `--resume-from` / `--continue-from` | 空 | 恢复滚动 checkpoint / 从 best.pt 继续（新优化器新早停） |
| `--no-fail-on-monitor-error` / `--max-train/--max-val/--max-test` | 关 / 0 | |

### 5.4 `scripts/inference.py`（TGB 推理）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` | 必填 | train.py 产出的 best.pt（config.json 自动同目录读取） |
| `--split` | test | `val` / `test` |
| `--output` | 必填 | 推理结果目录 |
| `--gpu` / `--bs` | 0 / 200 | |

### 5.5 `scripts/run_a0_synthetic.py`（合成实验）与辅助脚本

| 参数 | 默认 | 说明 |
|---|---|---|
| `--exp` | 全部 | `mod8` / `xor` / `shared_dag` / `path_gain` / `circular` 单个跑 |

辅助：`plot_tb_json.py --json --outdir [--epoch-steps]`（TB JSON 画图）、`tb_export.py --logdir --outdir [--epochs-per-step]`（TensorBoard 导出 PNG）、`protocol_ab.py --checkpoint [--gpu] [--output]`（协议 A/B 诊断，输出 JSON）。

## 6. 调参指南

A0 的设计让调参有章可循：**每个参数都有对应的审计指标当导航，不是盲调**。

### 6.1 第一档：真正需要调的（A0 线核心 5 个）

| 参数 | 默认 | 调什么 | 调参技巧（审计导航） |
|---|---|---|---|
| `--r` | 32 | 压缩预算 | **看 `rank_tail_max` 定**：跑完 C 阶段后——tail 大（>0.2）说明 r 不够用，往上加；tail 小且 r=32→64 分数不涨，说明 32 已饱和。正式实验建议直接跑 r∈{16,32,64} 三点画 Pareto（维度 vs AUC），论文里这也是呈现方式，不浪费 |
| `--lambda-gamma` | 1e-3 | 阶段 B 设计正则 | **看 `sibling_support` 条件数**：cond < 100 → λ 可降到 1e-4；cond 1e7+（wiki 真实数据很可能，冒烟实测 8700 万）→ λ 加大到 1e-2~1e-1，否则 B̂ 在共线方向过拟合噪声 |
| `--lambda-x` | 1e-4 | 阶段 A 白化正则 | 诊断信号：solve 后 R 行范数爆炸 / z 幅度异常 → λ 太小；C_xx 条件差（特征共线）→ 加到 1e-3~1e-2 |
| `--frac-a` | 0.2 | A 窗口比例 | **看 quotient 的 n（行数）**：n 应远大于 p=172。wiki 全量 0.2 够（~5 万行）；数据 <2 万行时 A 提到 0.3（R 是最重要的估计，值得多吃） |
| `--trace-roots` | 16 | A/B/C 采样密度 | 统计质量 ∝ 这个数。时间允许就往上加（32）；冒烟时减小到 4~8 |

### 6.2 第二档：实验设计型选项（不是调参，是对照组）

| 参数 | 用法 |
|---|---|
| `--chi-mode meanpool/sketch` | meanpool 保基线，sketch 作升级对照（邻居多时表达力强）。两个都跑，比较 closure_residual |
| `--use-weights` | **先关跑一轮** → 若结果异常或怀疑上下文与历史强耦合 → 开权重跑第二轮对比。开权重要看 `ess_frac_min`：<0.2 说明修正勉强（G0 预警） |
| `--d-context` | 与 r 配合：u 宽 = 2×d_context 必须 ≥ 预期的有效秩，否则 M 被 clamp。看谱的显著值数量，d_context 至少取显著秩的一半以上 |
| `--deploy-events` | 调完所有参数后最后开，只看不调（部署偏差/吞吐是产出不是输入） |

### 6.3 第三档：不要动的（动了就破坏对比口径）

- **宿主超参**（`--bs/n-degree/n-layer/message-dim/memory-dim/drop-out`）：必须与预训练 checkpoint 一致，改了权重就对不上；
- **D 阶段默认**（`--lr 3e-4 / --n-epoch 10 / --patience 3`）：与 JODIE 线同口径，baseline decoder 的 0.8871 self-check 锚点依赖这些默认值——**改了锚点就复现不了，连 baseline 是不是对都不知道**；
- **TGB 线 `--grad-clip` 必须 0**（代码注释里实测踩过：这个宿主梯度合法到 1e9，开裁剪直接冻死训练）；
- **门阈值不是调参**：先 `--gate-mode report` 跑出真实值，再按"真实值 × 1.5"定阈值写进预注册，之后 `stop` 模式执行。

### 6.4 通用方法论（六条）

1. **先跑后调**：A0 的审计表就是调参地图——每个参数有专属诊断指标（r↔rank_tail、λ_gamma↔cond、frac_a↔n），永远先看数再动手；
2. **一次只改一个**：双头设计下 A0 参数只影响 A0 头、baseline 不动——单次变更的因果干净；
3. **小数据验证合法性**：`--max-train 3000` 冒烟先确认参数组合不爆数值，全量只跑确定配置（省 GPU 时间）；
4. **λ 的尺度启发式**：λ 要和设计矩阵特征值尺度匹配——cond 大就加大 λ，判断标准是 B̂ 的 condition_number 和 leverage 分数回到合理区间；
5. **调参目标是被审计，不是刷分**：目标 = G0–G3 过线 + G4 非负，不是无脑最高 AUC——这是文档的哲学，也防审稿人质疑"增益是调参调出来的"；
6. **seed 是最后一道**：固定投影 P_c 和 trace 采样都受 seed 影响，定参后 3 seeds 报方差。

**实用建议**：wiki 全量第一轮先跑 `--r 32` + `--gate-mode report`（不带门阈值），跑完看审计表 → 用 6.1 的第 1、2 条调整 λ/r → 第二轮再跑 `stop` 模式 + r=64。这样 6 个 run 里前 2 个是"诊断 run"，后面 4 个才是正式对比 run，不浪费。

## 7. summary 的审计表怎么看

| 字段 | 白话 | 健康标准 |
|---|---|---|
| `audit.ess` / `ess_frac_min` | 有效样本占比（权重模式） | 越接近 1 越好，<0.2 说明上下文与历史强耦合（G0） |
| `audit.rank_tail_max` | 截断到 r 维丢掉的预测能量占比 | 小 = r 够用（G1）；冒烟小数据下恒 0 是退化不是 bug |
| `audit.proper_score_regret_by_tau` | 每层：压缩 vs 富历史的 log/Brier 分数差 | 接近 0 = 压缩无损；按层看哪层损失大 |
| `audit.closure_residual_max` | B̂ 递归预测与真值的偏差 | 大 = 递归不闭合（G2） |
| `audit.sibling_support.*` | 设计矩阵条件数 + leverage | cond 爆表 = 兄弟共线（真实数据常见，见坑 5） |
| `audit.path_gain_product` | 各层 Lipschitz 增益的深度乘积 | >1 = 深树误差放大（G3） |
| `deployment.deploy_vs_rich_deviation_mean` | r 维递归状态 vs 宿主前向状态的偏差 | 小 = 部署保真 |
| `deployment.state_bytes_per_node_*` | 每节点状态内存（r×4B vs host×4B） | A0 的成本优势证据 |

## 8. 验证口径与锚点（预注册判据）

- **A0 vs vanilla = run 内部双头**：同一次前向训练 A0 readout（z_root，r 维）与 baseline decoder（x_root，宿主维），各自 val 早停，同 test 段。`delta_auc = A0 − baseline` 直接可读。
- **baseline decoder 是内置 self-check**：wiki 上应复现 **0.8871 ± 0.002**（train_jodie vanilla 锚点）。不复现 = 宿主装配有 bug，该 run 的 A0 数字一并作废。
- **A0 vs 旧 spectral（0.9073）= 跨 run**：共享同一数据集切分/checkpoint/评分/协议，差异只有模型。结论分两层：指标层 + 成本层（A0 只传 r=32/64 维 vs 旧 spectral 解压回 172 维）。
- **TGB wiki 五变体锚点**（test_mrr）：spectral 0.3897 > pca 0.3633 > vanilla 0.3369 > random 0.3190 > direct 0.2936。
- 正式实验跑前先写好 `docs/a0_preregistration.md`（文档 11.4 五个"若…则…"断言 + 阈值，commit 留时间戳）。

## 9. 云端环境速查

| 项 | 值 |
|---|---|
| SSH | `ssh -p 13743 root@connect.weste.seetacloud.com`（RTX 5090 D 32GB） |
| 代码 | `/root/autodl-tmp/PRSS2`；python `/root/miniconda3/bin/python`（torch 2.8.0+cu128） |
| 数据 | `old/processed_tgn_data/`（wikipedia 三件套） |
| 预训练权重 | `outputs/pretrained/tgn-wikipedia.pth`（从旧 `jodie_wikipedia/vanilla_seed0/best.pt` 的 `model.tgn` 键提取，38 键含 affinity_score） |
| 云端 git | **fetch/pull 不可用**（https 无凭据）→ 一律 scp 按文件同步 |

---

# 第二部分：给 Claude Code 的协议与坑

## 10. 操作约定（必须遵守）

1. **耗时操作先征得用户同意**（训练/下载/全量测试）；本机单测与合成实验可直接跑。
2. **部署流程**：本地开发 + 本机单测 → commit + push（GitHub）→ scp 到云端对应路径 → 云端 `pytest` + 冒烟 → 下载 `outputs/` 归档本地。
3. **云端不可覆盖的文件**（云端有本地修改）：`DEVELOPMENT v2.0.md`、`configs/experiments/*.yaml`、`requirements.txt`、`scripts/inference.py`。scp 只同步指定文件。
4. **零改动红线**：`src/prss/core.py / auxiliary.py / outside.py / reader.py / compressors/ / state.py / config.py`、`scripts/train.py`、`scripts/train_jodie.py`。A0 的全部逻辑在 `src/prss/a0/` + `training/a0_loop.py` + `scripts/train_a0.py`。
5. **图表 CJK 限制**：本机 matplotlib 缺中文字体，图内 suptitle/标签用英文。
6. **测试分工**：本机 `pytest test/test_a0_quotient.py`（纯 torch）；`test/test_a0_loop.py` 与 `test_jodie_*` 数值测试需要 numpy bridge → 云端跑；本机 `test_tgb_smoke` 因 tgb 未装必然失败（环境问题，非回归）。

## 11. 已知坑清单（新会话必读）

1. **numpy bridge 本机坏**（torch.from_numpy 失败）→ 数值测试必须云端；纯 torch 构造可本机。
2. **padding 邻居孤儿 occurrence**：jodie_tgn 为 mask 掉的 node-0 邻居也建了 occurrence 但父不引用 → `propagate_root_labels` 覆盖不到。A/B/C 阶段一律只处理 roots 可达的 occ（probes.py 的 stack_by_tau 内跳过，loop 的父遍历 `if occ.occurrence_id not in oid_labels: continue`）。
3. **累积缓冲设备**：A0Quotient/OperatorRidge/ResidualAccumulator 的缓冲构造在 CPU，首次 accumulate 时跟随数据设备（CUDA）。
4. **官方 TimeEncode 输出 3 维** `(1, 1, time_dim)`——索引取 `[0, 0]` 才是向量。
5. **真实数据兄弟共线**：NeighborFinder 邻居重叠 → χ 设计矩阵 cond 可到 1e7-1e13（冒烟实测 8700 万）。这是 shared-DAG 合成实验预测的真实现象，靠 `sibling_support` 审计诊断，不是 bug。
6. **rank 会被 M 的秩上限 clamp**（min(m, p)）：u 宽 m=2·d_context，p=host_dim；扫 rank 网格时用 `q.r_matrix.shape[0]` 取实际 rank。
7. **trace 消费时机**：下一 batch 前向会重建 `_trace`——每 batch 前向后立即消费 trace 再 clear。
8. **预测度量是唯一正确判据**：rank-r 截断下裸欧氏距离会"假分离"（坐标旋转不唯一）。分离度/闭合一律用 Σ 加权距离（= 预测差异，文档 5.5 式 8）。
9. **密度比判别器必须带交互特征**：配对 vs 打乱的负样本不改特征边际 → 一阶特征与标签正交 → IRLS 在原点梯度恒 0（鞍点）。weights.py 已内置交互（小维度外积 / 大维度随机投影交互）。
10. **部署 lift 方向**：`P_lift = (RRᵀ+εI)⁻¹R`（r×p），用法 `z @ P_lift` → host 宽度。
11. **冒烟小数据的退化现象**（非 bug）：3000 行 cap 下 rank_tail=0（正例太少）、cond 爆表（B 窗口 6 batch）、deploy 偏差大——全量数据才有统计意义。

## 12. 未完成工作清单（接手方向）

| 类别 | 项 | 依据 |
|---|---|---|
| 理论 | 式 (13)-(15) 路径增益证明、式 (17) 统一误差分解、四个候选主贡献 | 文档 13.2 / 16.5 / 8.3 |
| 文献 | P1-P7 七个文献包检索 + 第 15 节研究 prompt 执行 | 文档 14/15 |
| 实验 | wiki 全量 r=32/64 × seeds（预注册先行）、enron、11.3 全基线列表 | 文档 11 |
| 文档 | `docs/a0_preregistration.md` 预注册预测 | 文档 11.4 |

## 13. 版本记录

- 2026-08-21：A0 训练线初版（commit 44e289a）→ GPU 修复（6e795d2）→ 合成实验套件（8919845）→ 方法补全五件套（7849986：proper-score 分层 / leverage-OOD / 重要性权重+G0 / TensorSketch / 递归部署 v2）。云端 29 项测试全绿，冒烟四件套跑通。
- 2026-08-21：使用文档首版（1c63c16）+ 五入口全参数速查（5.1–5.5）+ 调参指南（第 6 节，三档分类 + 六条方法论）。
