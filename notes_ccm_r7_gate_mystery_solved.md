# CCM 第七轮 gate step0 排查：运行间差异谜团 = 聚合口径混淆（已定论）

日期：2026-09-07。任务 #74（step0 identity gate）排查结论。

## 谜团与表象

- official json（turn14_audit_val_official.json，12:23 生成）：L1 record-mean = **1.920893**
- repro_official_l1.py（12:58，逐条复制 run_official_audit 路径）：**1.9208930800**，与 json 逐条 nll 一致（0 diff，token_count 3–48/行）
- diag_gate_ab.py（12:46）与 repro_ab_nll.py（13:03）：ref/host token-weighted mean = **2.068211**（per-L "tokens=862"）
- 曾误判为"同一代码两次运行产出不同模型/不同 cohort"，排查过 RNG（data.py valset 无任何随机消费，文件顺序确定性）、adapter 中途改写（result/dialog mtimes 全部 00:57 前，无改写）、进程间非确定性——全部排除。

## 真相（无需任何修复）

**两个值不是"两次运行"，而是同一模型同一批 51 行的两种聚合口径**：

| 口径 | 公式 | L1 结果 |
|------|------|---------|
| record-mean（evaluate_arm / official json / gate nlls()） | `mean(per-record loss_sum/token_count)`，51 条对话等权 | 1.920893 |
| token-weighted（AB agg） | `Σloss_sum / Σtoken_count`，长句权重大 | 2.068211 |

- "tokens=862" 是 **51 行 × 每行 3–48 tokens 的 L1 总和**，不是每行 862。单行 token_count = 目标句 u14 的有效 token 数（几到几十），与 L 无关（official json 内 L=1..13 同行 token_count 相同，可核）。
- 两口径差 0.147 来自行间 token 数严重不均衡（3–48）+ 每行 nll 与长度相关。
- 判定身份/一致性（AB_GATE_PASS）用哪种口径都有效：ref vs host 在 5L×51 全部 batch 上 max_logit diff = **0.00e+00**（255/255 行逐位相同），任何口径下 nll 都相同。

## 对第七轮 gate 的裁定

- **task@step0 臂模型 == official merge**：由 AB 逐位证明（build_official_host 官方宿主继承 12:13 修复后，host(gamma=False) 与 ref 权重、前向逐位一致）。
- 官方参考值主口径锁定 **record-mean（evaluate_arm 系）**：L1=1.920893, L2=1.811304, L4=1.766103, L8=1.744795, L13=1.743476（51 行/每 L）。
- 注意：AB 只覆盖了 cond 的 host(gamma=False)；cond 臂 = host + attach_gamma（零输出 Γ），step0 无记忆消费时 Γ 不应参与——需 gate_official_host.py 正式重跑五 L 三臂确认（cond 与 task 的 record-mean 差 < 1e-4）。
- 旧 gate 失败值（task=2.012331 vs ref=1.920893）来自 12:13 修复前或旧口径混用，不作数；以本次 AB + gate 重跑为准。

## 教训（写进未来做法）

1. **报告 NLL 必须带口径标签**：record-mean vs token-weighted，二者不可混比（短句行 3 tokens vs 长句行 48 tokens 时两口径可差 ~0.15）。
2. "同一脚本两次运行结果不同"类谜团，先查**输出文件里的中间量（token_count 等）**是否自洽，再怀疑 RNG/模型。
3. per-L 聚合口径统一为 evaluate_arm 的 record-mean（官方 json 与 gate 同源），AB 类脚本若用加权只用于身份比较，不得与主表混比。

## 追加（同日 14:00-15:00）：真正的 GATE_FAIL 根因 = evaluate_arm 多模型驻留污染

口径谜团解决后，gate_official_host.py 重跑仍 GATE_FAIL（nma1 复现且数值与旧实例逐位一致：
ref=1.920893 task=2.012331 cond=2.039071）。同机 quick AB（L1，ref+task 同驻、手写循环）PASS。
triple_check（单进程 ref+task 同驻，手写循环 vs evaluate_arm 各跑一遍）给出铁证：

| 调用 | L1 | 与 gate 的关系 |
|------|----|----|
| hand_rolled(ref) | 1.920893 | = 官方 json / gate.ref ✓ 真值 |
| evaluate_arm(ref) | 2.012331 | = gate.task 的值！ |
| hand_rolled(task) | 2.039071 | = gate.cond 的值！ |
| evaluate_arm(task) | 2.049668 | 全新值（gate 无第 4 模型） |

## 定论（同日 16:30）：根因 = vendored collator 就地 mutation，非模型、非评估路径

**Root cause（一行代码，官方 repo 自带 bug）**：`third_party/ccm/src/data/dialogue/data.py`
`_concat_dialog()` 内 `token_ = dialog[i]` 拿的是**共享 dataset 行的引用**，而 merge_recur 协议下
`token_ += sum_token` 是 **list 就地 extend** —— 每次 collator 调用都把 sum_token 永久追加进
内存中的 dialog 行。官方流程每个 item 只 collate 一次（每进程一轮评估）所以从不暴露；
我们的门禁把三臂（ref/task/cond）+ 手写循环在**同一进程同一 collator 同一批 dialogs** 上
反复评估 → 第二轮起每行已带多余 sum_token，input_ids 逐轮变长 → NLL 漂移。

**prove_mutation.py 实锤**：同一 item collate 两次，dialog 行长度 10→14→18（每次 +4），
两次 batch shape 44 vs 48；修复后 10→10→10，两次 batch 逐位相同。

**全表自洽**（"进程内第 k 次完整评估" = 第 k 轮污染）：

| 观测值 | 解释 |
|--------|------|
| gate ref=1.920893（第 1 轮） | 行干净 → 真值 |
| gate task=2.012331（第 2 轮） | 行已带 sum_token → 假警报 |
| gate cond=2.039071（第 3 轮） | 假警报 |
| triple evaluate_arm(ref)=2.012331 == gate task | 同为第 2 轮污染 |
| triple hand(task)=2.039071 == gate cond | 同为第 3 轮污染 |
| order A2=1.9267（仅 1 行预污染） | 只有 dialog0 行脏 |
| row_probe：evaluate_arm 在 hand 之后 → 255/255 行全偏 | 第 2 轮 vs 第 1 轮 |

**附带修正：旧官方 json 本身也是脏的**（L 内层循环，L1 先行 → 行 12 被 L1 污染后 L2/L4/L8/L13
再用）。仅 L1=1.920893 是真值（每 dialog 的 L1 是该 dialog 行的首次使用）。L2+ 的旧参考值
（1.811304/1.766103/1.744795/1.743476）作废。

**修复**：`_concat_dialog` 改 `token_ = list(dialog[i])`（拷贝行，concat 语义本就不该 mutate 输入），
已加 `LOCAL FIX (gate_r7)` 注释。备份 `data.py.bak_gatefix`。

## 最终裁定：step0 identity GATE_PASS（五 L × 三臂，d 全 0）

gate_official_host.py 修复后重跑（gate_official_host_r7_FIXED.log，2026-09-07）：

| L | ref | task | cond | d_task | d_cond |
|---|-----|------|------|--------|--------|
| 1 | 1.920893 | 1.920893 | 1.920893 | 0.00e+00 | 0.00e+00 |
| 2 | 1.814313 | 1.814313 | 1.814313 | 0.00e+00 | 0.00e+00 |
| 4 | 1.742679 | 1.742679 | 1.742679 | 0.00e+00 | 0.00e+00 |
| 8 | 1.728306 | 1.728306 | 1.728306 | 0.00e+00 | 0.00e+00 |
| 13 | 1.700960 | 1.700960 | 1.700960 | 0.00e+00 | 0.00e+00 |

GATE_PASS。**task@step0 == cond@step0 == official merge 在 logit 级成立**（五 L 全部
d < 1e-4 门禁，实际为 0）。**新的干净官方参考值表**（取代 12:23 json）：L1=1.920893、
L2=1.814313、L4=1.742679、L8=1.728306、L13=1.700960（record-mean，51 clean-val dialogs）。
第七轮 step0 identity 门禁 **正式 PASS** → 双臂训练放行（同一 official-merge-adapter 起点，
lr 3e-5）。

## 教训
1. 报告 NLL 必须带口径标签（record-mean vs token-weighted）。
2. **评估基建的"多轮复用"要防输入可变性**：collator 若有就地拼接（+=），同一 dataset 多轮
   评估会累积污染 —— 遇"第 k 次调用结果不同"先查 collator/dataset 是否被 mutate，
   再怀疑模型与 RNG。prove_mutation 式"同一 item collate 两次比 input_ids"是 2 分钟定案手段。
3. "与官方数值不一致"的排查链应含：参考值本身是否在污染序列中生成（旧 L2+ json 即此例）。
