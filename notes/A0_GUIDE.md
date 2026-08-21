# A0 使用文档（用户 + Claude Code 双读者版）

> 用途：A0（条件矩预测商训练线）的操作手册与交接文档。上半部分是给人看的白话指南；下半部分是给 Claude Code 的精确协议与坑清单。
> 更新日期：2026-08-21
> 理论依据：《递归预测商_检索审计与A0算法_2026-08-20.md》（仓库根目录）

---

# 第一部分：给人看的

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

## 5. summary 的审计表怎么看

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

## 6. 验证口径与锚点（预注册判据）

- **A0 vs vanilla = run 内部双头**：同一次前向训练 A0 readout（z_root，r 维）与 baseline decoder（x_root，宿主维），各自 val 早停，同 test 段。`delta_auc = A0 − baseline` 直接可读。
- **baseline decoder 是内置 self-check**：wiki 上应复现 **0.8871 ± 0.002**（train_jodie vanilla 锚点）。不复现 = 宿主装配有 bug，该 run 的 A0 数字一并作废。
- **A0 vs 旧 spectral（0.9073）= 跨 run**：共享同一数据集切分/checkpoint/评分/协议，差异只有模型。结论分两层：指标层 + 成本层（A0 只传 r=32/64 维 vs 旧 spectral 解压回 172 维）。
- **TGB wiki 五变体锚点**（test_mrr）：spectral 0.3897 > pca 0.3633 > vanilla 0.3369 > random 0.3190 > direct 0.2936。
- 正式实验跑前先写好 `docs/a0_preregistration.md`（文档 11.4 五个"若…则…"断言 + 阈值，commit 留时间戳）。

## 7. 云端环境速查

| 项 | 值 |
|---|---|
| SSH | `ssh -p 13743 root@connect.weste.seetacloud.com`（RTX 5090 D 32GB） |
| 代码 | `/root/autodl-tmp/PRSS2`；python `/root/miniconda3/bin/python`（torch 2.8.0+cu128） |
| 数据 | `old/processed_tgn_data/`（wikipedia 三件套） |
| 预训练权重 | `outputs/pretrained/tgn-wikipedia.pth`（从旧 `jodie_wikipedia/vanilla_seed0/best.pt` 的 `model.tgn` 键提取，38 键含 affinity_score） |
| 云端 git | **fetch/pull 不可用**（https 无凭据）→ 一律 scp 按文件同步 |

---

# 第二部分：给 Claude Code 的协议与坑

## 8. 操作约定（必须遵守）

1. **耗时操作先征得用户同意**（训练/下载/全量测试）；本机单测与合成实验可直接跑。
2. **部署流程**：本地开发 + 本机单测 → commit + push（GitHub）→ scp 到云端对应路径 → 云端 `pytest` + 冒烟 → 下载 `outputs/` 归档本地。
3. **云端不可覆盖的文件**（云端有本地修改）：`DEVELOPMENT v2.0.md`、`configs/experiments/*.yaml`、`requirements.txt`、`scripts/inference.py`。scp 只同步指定文件。
4. **零改动红线**：`src/prss/core.py / auxiliary.py / outside.py / reader.py / compressors/ / state.py / config.py`、`scripts/train.py`、`scripts/train_jodie.py`。A0 的全部逻辑在 `src/prss/a0/` + `training/a0_loop.py` + `scripts/train_a0.py`。
5. **图表 CJK 限制**：本机 matplotlib 缺中文字体，图内 suptitle/标签用英文。
6. **测试分工**：本机 `pytest test/test_a0_quotient.py`（纯 torch）；`test/test_a0_loop.py` 与 `test_jodie_*` 数值测试需要 numpy bridge → 云端跑；本机 `test_tgb_smoke` 因 tgb 未装必然失败（环境问题，非回归）。

## 9. 已知坑清单（新会话必读）

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

## 10. 未完成工作清单（接手方向）

| 类别 | 项 | 依据 |
|---|---|---|
| 理论 | 式 (13)-(15) 路径增益证明、式 (17) 统一误差分解、四个候选主贡献 | 文档 13.2 / 16.5 / 8.3 |
| 文献 | P1-P7 七个文献包检索 + 第 15 节研究 prompt 执行 | 文档 14/15 |
| 实验 | wiki 全量 r=32/64 × seeds（预注册先行）、enron、11.3 全基线列表 | 文档 11 |
| 文档 | `docs/a0_preregistration.md` 预注册预测 | 文档 11.4 |

## 11. 版本记录

- 2026-08-21：A0 训练线初版（commit 44e289a）→ GPU 修复（6e795d2）→ 合成实验套件（8919845）→ 方法补全五件套（7849986：proper-score 分层 / leverage-OOD / 重要性权重+G0 / TensorSketch / 递归部署 v2）。云端 29 项测试全绿，冒烟四件套跑通。
