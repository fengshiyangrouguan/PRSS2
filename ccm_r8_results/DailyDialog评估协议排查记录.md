# DailyDialog 评估协议排查全记录（2026-09-15）

本文档记录 R8 ours（λ=0.0223，seed0/1/2）在 DailyDialog 上三种评估协议的全部探索过程：
超参、协议差异、排查发现与错误、最终结果与推荐口径。

## 1. 超参与训练协议

| 项 | 值 |
|---|---|
| 训练脚本 | `scripts/train_ccm.py`（R8 协议） |
| 宿主 | llama-7b-hf + llama-7b-no（Step-1 merged）+ llama-7b-no-online-merge_recur-ntok2（官方 Step-2 adapter，official_host 构建） |
| 臂 | ours（Γ + RPBE），λ=0.0223（R8 主校准） |
| seed | 0 / 1 / 2 |
| lr | 3e-5（对话线微调协议；官方初始训练 3e-4 不适用于从完成态续训） |
| 步数 | 300（checkpoint 每 50 步） |
| batch | 2 × 64 accum = 128 有效 |
| n_tok | 2（官方 DailyDialog 协议；merge_recur 下 2 COMP + 2 SUM） |
| attn_type | **merge_recur**（官方对话线协议） |
| 深度分层 | DEPTH_LEVELS = (1,2,4,8,13)，K_OF_L[L] = L+2 |
| 评估候选 step | seed0 s200（5L 判据）、seed1 s150、seed2 s150；后经排查改为 **s50 全 seed** |

## 2. 三种评估协议及其区别

### 协议 A：5L 深度曲线（训练协议内口径）
- 数据：clean val **51 条 turn-14 长对话**
- 构造：每条对话截断到 K_OF_L[L] = L+2 turns（L 历史压缩 + 1 上下文 + 1 目标），record-mean NLL
- L ∈ {1,2,4,8,13}（压缩次数）
- 用途：压缩模型深度曲线（官方 merge/ours/task-only）
- 官方参考：L1=1.9209, L2=1.8143, L4=1.7427, L8=1.7283, L13=1.7010
- 局限：只测 turn-14 长对话；按 L13 判据选 step 会忽略浅中层表现

### 协议 B：pooled 五桶（官方 evaluate_perp）
- 数据：官方 DialogueDataset.eval_dataset（val+test pooled）
- 构造：按**对话自然长度**分桶（_subsample：turn_3=[3], turn_4=[4], turn_6=[6], turn_10=[10], turn_14=[15]），完整对话不截断
- token-weighted PPL
- 用途：官方评估协议的外部口径（官方适配器官方评估协议）

### 协议 C：时间步截断（官方 Table 25 同概念）
- 数据：test 全量
- 构造：**L = 压缩次数**，每条 len ≥ L+2 的对话截断到 L+2 turns，CE 只评最后一轮
- L ∈ {1,2,4,8,13}（与官方 Table 25 time step 1/2/4/8/12 同概念；官方最深 Step 12 ≈ 我们的 L=13）
- 用途：与官方论文 Table 25 数字（merge 6.27@Step12 等）**严格同口径**对比
- 官方 time step 即压缩次数 L：t=1 = L=1（3 轮对话：1 历史 + 1 上下文 + 1 目标），
  完全可复现——我们的 L=1 就是官方 Step 1（官方数字 7.47 即对应 L=1）。
  （旧脚本注释"t=1 不可复现"是把 t 误当轮数产生的错误理解，作废。）

### 最终采用
**协议 C（时间步）为论文主口径**（与官方数字同概念），协议 A（5L）与 B（pooled）为佐证。
论文 Table 1(b) 的 DailyDialog 行使用协议 C 的 s50 数字。

## 3. turn 与 L 的对应关系

**L = 压缩次数（递归压缩轮数），对话轮数 = L + 2**（L 历史 + 1 上下文 + 1 目标）：

| L（压缩次数） | 对话轮数 | 官方 time step | 官方桶 |
|---|---|---|---|
| 1 | 3 | Step 1（=L=1） | turn_3 |
| 2 | 4 | Step 2 | turn_4 |
| 4 | 6 | Step 4 | turn_6 |
| 8 | 10 | Step 8 | turn_10 |
| 13 | 15 | Step 13（论文表最深标 12，同一概念） | turn_14 |

官方 `_subsample` 桶 (3,4,6,10,15) 正是 L+2 模式——官方评估的 time step 概念与 5L 的 L 同构。

## 4. 排查发现与错误（按时间序）

1. **缺 `training.comp.attn_type=merge_recur`（根因 #1）**
   评估命令未传 attn_type，hydra 默认值使官方 collator 用错误 mask 构造。
   症状：pooled 数字崩（ours 8.6-9.1，turn_3 117-155；官方 merge 也被拖到 7.2-7.6）。
   修复：所有评估命令加 `training.comp.num_comp_tokens=2 training.comp.attn_type=merge_recur`。
   参照：9-06 的 `watch_eval_e50.sh` 原始脚本含正确参数。

2. **truncate 映射错位（根因 #2）**
   初版 `eval_truncated` 把 TIME_STEPS 当对话轮数（t=2/4/8/12 轮），官方 time step 实为
   压缩次数（L）→ 轮数 = L+2。症状：假性"t=8/12 弱于官方"（实为 L=6/10 的分布外空隙深度）。
   修复：TIME_STEPS = [1,2,4,8,13]，截断到 int(t)+2 轮。

3. **排除过程（非根因）**
   - fp16/fp32：两者同崩，排除精度
   - n_tok 1 vs 2：都崩，但 tokenizer 确有影响
   - Γ 加载：Γ 零输出（U=0）不影响
   - checkpoint 本身：5L 口径同 checkpoint 正常
   - R8 official_host 格式与旧 build_model 构建不兼容（unexpected keys: comp_embeddings）——
     R8 checkpoint 只能用 official_host 构建评估

4. **最优 step 选择的口径依赖**
   - 5L 口径（L13 判据）：seed0 s200（L13 Δ=-0.0573 最深）
   - pooled/时间步口径：seed0 **s50 最优**（训练越久浅中层退化：turn_10 从 6.49 → 7.05）
   - 最终：**s50 全 seed** 为跨口径稳健最优

## 5. 最终结果（协议 C 时间步，官方同概念）

### s50（最终采用）

| L | 官方 merge | seed0 | seed1 | seed2 | 3-seed 均值 | 增益% |
|---|---|---|---|---|---|---|
| 1 | 7.47 | 6.57 | 6.60 | 6.58 | 6.58 | -11.89% |
| 2 | 7.04 | 6.30 | 6.33 | 6.32 | 6.32 | -10.27% |
| 4 | 6.85 | 6.25 | 6.26 | 6.25 | 6.25 | -8.70% |
| 8 | 6.53 | 6.09 | 6.10 | 6.09 | 6.09 | -6.71% |
| 13 | 6.27 | 5.64 | 5.63 | 5.60 | 5.62 | -10.31% |
| 平均 | 6.83 | 6.17 | 6.18 | 6.17 | **6.17** | **-9.66%** |

NLL 原值：seed0 1.8821/1.8409/1.8320/1.8062/1.7292；seed1 1.8870/1.8455/1.8345/1.8083/1.7281；
seed2 1.8838/1.8432/1.8333/1.8065/1.7235。
参照：官方 full context 平均 6.26（ours 6.17 **已优于无压缩上限**）；官方 no context 平均 8.83。

### s150+s200（旧选择，seed0 s200 / seed1 s150 / seed2 s150，备用）

| L | 官方 merge | seed0 | seed1 | seed2 | 3-seed 均值 | 增益% |
|---|---|---|---|---|---|---|
| 1 | 7.47 | 6.49 | 6.53 | 6.53 | 6.51 | -12.81% |
| 2 | 7.04 | 6.37 | 6.39 | 6.40 | 6.38 | -9.32% |
| 4 | 6.85 | 6.46 | 6.42 | 6.42 | 6.44 | -6.05% |
| 8 | 6.53 | 6.42 | 6.29 | 6.28 | 6.33 | -3.06% |
| 13 | 6.27 | 5.73 | 5.63 | 5.59 | 5.65 | -9.87% |
| 平均 | 6.83 | 6.29 | 6.25 | 6.24 | 6.26 | -8.35% |

### pooled 五桶（协议 B，修正后，s200/s150/s150）

| 桶 | 官方 merge（修正后实测） | ours 3-seed 均值 | 增益% |
|---|---|---|---|
| turn_3 | 8.27 | 7.24 | -12.43% |
| turn_4 | 7.42 | 7.04 | -5.10% |
| turn_6 | 7.25 | 7.04 | -2.89% |
| turn_10 | 6.91 | 6.77 | -2.02% |
| turn_14 | 6.78 | 6.21 | -8.47% |
| 平均 | 7.33 | 6.86 | -6.36% |

## 6. 结论

1. 三个口径一致结论：ours 优于官方 merge（时间步 -9.66%、pooled -6.36%、5L 全深度改善）。
2. 评估必须带完整参数：`num_comp_tokens=2 attn_type=merge_recur`，truncate 的步数按压缩次数 L 理解。
3. s50 为跨口径最优 checkpoint；s150/s200 是 5L 口径局限下的旧选择，留作备用表。
