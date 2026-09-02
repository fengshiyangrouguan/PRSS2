# CCM-merge × RPBE 跨域实验：L0–L6 开发汇报

> 分支 `develop_CCM`｜vendored 官方仓库 snu-mllab/Context-Memory @ `a89dd08`（MIT）｜DailyDialog × LLaMA-7B
> 汇报日期：2026-09-02｜状态：**L0–L6 交付完毕，L6.5 门禁 1–2 PASS，Gate 4（步时判定）阻塞待仲裁**
> 实验证据：`docs/ccm_experiments/ccm_outputs_dump.json`（云端 summary 全量）、`docs/ccm_experiments/parity_gate2.json`（Gate 2 权重级 parity）、`docs/ccm_experiments/step_timing_gate4.json`（Gate 4 计时原始证据）

---

## 0. 实验定位与规格基线

本实验验证 TGN 主实验（exact_replay vs vanilla **+1.53 点、5/5 seed 全正**）结论向 LLM 的迁移：在官方 CCM-merge（ICLR 2024，snu-mllab/Context-Memory）宿主上检验 RPBE「future-sufficient 递归压缩」——用 LLM 自注意力运算产生的中间状态（SUM K/V）经固定适配器构造记忆观测 `z_v`，以两条 horizon 预测行监督递归记忆 `M_t`，让记忆只保存未来可验证的信息。

规格文档链（均已入库）：

| 文档 | commit | 内容 |
|---|---|---|
| CCM-LLM 跨域实验最终规格 v1 | `f023c5f` | 宿主 CCM-merge + DailyDialog + Γ 替换 g_update + 2Obs 单 p_v 监督 |
| 对接计划 v2（审阅修订） | `e6e21a4` | P0-1 跨 microbatch 两遍 replay、P0-2 两条 horizon rows、P0-3 冻结源修正 |
| 计划 v3（窗口门槛审计） | `9209d78` | N 扫描选最小稳定阈值、suffix-depth matched cohort、L5 符号/缩放封口 |

审阅裁决后的三个 P0 修正全部落地为代码事实：

- **P0-1**：单 dialogue 单 cut 无法计算协方差（中心化后全零）→ L5 采用跨 microbatch 的 exact 两遍 optimizer-window replay；
- **P0-2**：两个 horizon 不能拼成单 p_v（改变 C_PP 与白化几何）→ 一 cut 两条 horizon rows、共享 cut_id、权重 0.5/0.5，进同一个 Ky-Fan 窗口；
- **P0-3**：冻结源 = 已完成五 seed TGN 实验的实际目标——`z_v = J_mem(M_v)`（LLM 必需的固定适配器，TGN identity 是其特例）+ 两条 P rows；主实验**删除** rho_fix/witness/r_B/未被 loss 消费的 U/B 结构。新增 `configs/ccm/frozen_method.json` 机器可读冻结（ridge=1e-3、m=64、horizon_weights=[0.5,0.5]、rpbe_map_seed=0、λ=r_eff 规则）。

三臂定义（全程不变）：**ours**（CCM-merge + Γ 残差 + RPBE 监督）、**gamma_task_only**（同架构但 w_RPBE=0，仅 task CE——隔离 Γ 本身效果的对照）、**ccm_merge**（官方算术平均 merge，无 RPBE，复现官方协议）。

---

## 1. L0 — Fork 与协议清理（commit `df0a4c4`、`938813a`）

- vendor 官方仓库锁定 `a89dd08e2c9587ec9c6c3ad339bb154c33e6b41a`（MIT 保留），进 `third_party/ccm/`，不改官方内部文件（后续所有 patch 都在 src/rpbe 侧挂钩）；
- 双数据模式：**official-reproduction**（保留官方 pooled 复现）+ **clean-split**（`DialogueDataset` 增官方 train/validation/test 三拆分，`938813a`）；主实验全程用 clean-split；
- 不碰 tokenizer / preprocessing / random-prefix 采样。

## 2. L1 — 官方复现（按用户决定跳过，兼容层照常交付）

计划原案：Step1 LoRA → Step2 merge_recur 跑通、官方 pooled PPL 与论文量级一致、depth 1/2/4/8/12 官方评估作为次要复现结果。

**实际状态：L1 复现阶段按用户决定跳过**（资源与排期权衡），但代码层的必要前置——`845858d` 兼容层——完整交付：vendored CCM 是 2023 年底代码（transformers 4.31 / accelerate 0.2x），需适配本机/云端环境 transformers 4.44、peft 0.4、accelerate 1.x、py3.12（mask shim 内联、`MLP(config)`、RoPE 签名、Union 容器覆盖、collator dict、wandb Settings、`DIALOG_MIRROR` 本地数据镜像）。

**后果如实记录**：官方 pooled PPL 数字缺失，L9 报告中的「官方量级可比性」将由 ccm_merge 臂自身承担（三臂同码同机，merge 臂即官方协议的活参照）；官方协议正确性由 L6.5 Gate 2 权重级 parity（对撞官方 HF Trainer）兜底并 PASS——见 §10.2。

## 3. L2 — Γ 残差接入（commit `9181f1f`、`0953073`）

**设计**：`src/rpbe/hosts/ccm/gamma_residual.py` — Γ 在 merge 位置替换 g_update 的「可学习残差」角色：

```
R_θ(x) = s · U(tanh(V([prev; cur; feat])))   （x = 上轮记忆、当前 SUM 行、时间特征）
V: Linear(2D+16 → 64)  U: Linear(64 → D, bias=False)   TIME_FREQS=8, log-spaced sin/cos
s = 标量参数，zero-init 门控（保证 step 1 = 官方 merge，避免全零无梯度）
```

**关键工程点**：

- **zero-init 安全**：U/V 小随机、仅门控 s=0——首步数值上严格等于官方 arithmetic-mean merge，训练从官方轨迹出发；
- **任意实际 k 的 recurrence**：官方实现假设 T≤12；训练 random-prefix 采样实际 k 可远超（观测到 k 到几十），写成逐轮递归 scan（`ccm_patch.py` 挂载点），补 k>14 测试；
- **工程闭环 6 项**（`9181f1f` 验收全过）：① PEFT wrapping 后 Γ requires_grad 确认；② optimizer param groups 含全部 Γ；③ checkpoint 显式保存/恢复 Γ；④ roundtrip 容差一致；⑤ paired seed hash 含 LoRA+Γ+embedding 模式；⑥ 首个 step 后 Γ 参数确实离开 zero-init。
- `0953073` 修复：attach_gamma 需跟随模型设备（`.cuda()` 后再挂载时 Γ 参数留在 CPU——云端冒烟抓出）。
- torch.compile：三臂一致禁用（规格默认），避免动态 recurrence 的编译等价负担。

**发现**：Γ 只在 merge 的 M_t 路径上，不触碰 COUNT/SUM K/V 提取路径 → 数据流与 gamma_task_only 完全一致（Gate 1 用 data_flow hash 证实，§10.1）。

## 4. L3 — Host state adapter（commit `196ab84`）

- 确定性 collator metadata：k、utterance token spans、padding 后 COMP/SUM positions、sample/cut IDs——进模型前剥离，模型侧按固定索引重算，杜绝顺序依赖；
- SUM K/V 提取：改 LlamaAttention 输出路径，带梯度取每 turn 的 SUM 行 → `J_mem`：**固定**（requires_grad=False）layer/head/slot → 结构化 CountSketch（128 维，主实验无 witness）→ 归一化得 `z_v`；
- 主实验按 P0-3 冻结源**不构造 witness / rho_fix / r_B**；
- Test C（强化版，验收过）：替换 cut 后**全部**未来输入 u_{k−2}, u_{k−1}, u_k（等长替换），M_{k−3} 与 z_v 完全不变——因果隔离成立的数值证明。

## 5. L4 — 2Obs record（commit `196ab84`，v2 改规格后实现）

- 仅 k≥4 的 dialogue 参与 RPBE（k=3 保留 task CE、w_RPBE=0）；v = k−3；
- **一 cut → 两条 horizon rows**（共享同一 cut_id、权重各 0.5）：

```
row1: (z_v, p_{v,1} = TensorSketch([1;χ₁]⊗φ₁), ω=0.5, cut_id)
row2: (z_v, p_{v,2} = TensorSketch([1;χ₂]⊗φ₂), ω=0.5, cut_id)   χ₂ 含 one-update 标识
```

- 两条 rows 进同一个 Ky-Fan 窗口（clustered correction 走既有 WeightedWelford 路径，K 抵消/白化几何不变）；
- CountSketch 容量 524288 → 128（结构化，复用 maps.py 构造模式）。

## 6. L5 — 跨 microbatch 两遍 replay + 训练器（commit `fe80e89`、`b1eb064`、`410560a`）

**训练循环**（`scripts/train_ccm.py`）：

- **pass1（no-grad 统计）**：逐 microbatch 累计 WeightedWelford → 窗口关闭时 kf_adjoint 出 M/β；
- **pass2（exact replay）**：恢复输入/RNG/模型状态，重放窗口内全部样本 → task CE + `kf_vjp_batch`（λ 按 **r_eff 校准规则**：training-only 初始完整窗口校准一次，之后所有 seed 冻结同一 λ）→ 一次 optimizer step；
- 优化器窗口 = **≥128 有效 cuts 自适应收集**（按 k≥4 的 dialogue 计数，非机械 128 microbatch）；
- 三臂同一任务样本、同一 scheduler cadence、paired seeds（沿用 TGN 的 RNG 协议）；
- `410560a`：无 cut 样本（k<4）跳过 pass1 forward——有效 cut 率 ~50% 时省 ~25% 时间（也是后续所有步时测量的基线版本）。

**L5 修复链**（smoke/审阅抓出的真 bug，全部入库）：

- `fe80e89`：LoRA 包装走官方 peft_custom 优先（**comp_mask 与 peft 0.4 的兼容**，官方 Trainer 同款路径）+ 行号索引/重复 CE 修正；
- `b1eb064`：fp16 训练需 autocast（peft_custom 的 LoRA 分支 fp32，无 autocast 时与 fp16 base 混合报错）+ PEFT 包装后重设 comp/sum token。

## 7. L6 — 测试矩阵（commit `d2143f3`）

**构造级（本机 CPU）**：176 passed——Test A（zero-init 数值复现算术 merge）/B（Ours 训练 = no-grad 统计 forward + 有梯度 replay forward）/C（因果隔离，§4）/D/F/E（窗口、双 rows、record）+ L6 新增 5 项（RPBE/Γ/LoRA 梯度全非零、checkpoint roundtrip、pass1/pass2 状态与 RNG 重放、窗口非退化、eager/compile 一致性口径）。

**云端 37 项**全部通过（GPU 权重级，4.44 / 本机 python 版本差异下的同一套测试）。

**三臂权重级 smoke 发现 5 个真实 bug**：

1. **`__eou__` turn 拆分（最关键）**：`DIALOG_MIRROR` 最初把每个字符当一个 turn → 序列长 3.6k token → fp16 图 OOM。修复 `d2143f3`：按 `__eou__` 拆分 turns。此前 L6 三臂 smoke 一直挂在这个图上；
2. peft 0.4 的 comp_mask 兼容（§6 已修）；
3. fp16+autocast 缺失（§6 已修）；
4. 可变 utterance ordinal mask（随实际 k 变化的 mask 构造错误）；
5. token re-registration（PEFT 包装后 token id 需重注册）。

smoke 通过口径：显存 13.6GB 稳定（7B fp16 训练合理量级）、task loss 前 2 步三臂逐位一致（paired-seed RNG 协议验证）、有限值/梯度生效。

## 8. L6.5 — 门禁体系与两轮审阅修复（commit `5a73c1c` → `4e04c6b`）

### 8.1 P0-1：跨窗口 RNG 闭合（`c522770`）

pass1 终点状态（含 CUDA RNG）捕获 → pass2 重放前恢复，保证 replay 与正式训练「同一世界线」；pass2 从逐样本 collect_rows 精简为 `collect_replay_z`（不重放窗内 RNG 序列、只重建 z 流），避免二次随机化。**不碰 builder/χ/p**（审阅红线）。

### 8.2 P0-2：官方协议对齐 + Gate 2 parity（`c522770` → `e69fba5`）

对齐项：AdamW wd=0、cosine + 3% warmup、mean-loss/grad-accumulation 归一化、grad-clip 1.0、token-normalized CE、shifted mean CE（vendored `ccm_llama` ~L922 shift 语义）、`attention_mask_comp` 由 SUM token 推导并 fold 进 attention mask。

**÷128 事件（Gate 2 过程中最重要的协议发现）**：官方 HF Trainer 用 accelerate 1.14，其 `Accelerator.backward` 内部执行 `loss = loss / gradient_accumulation_steps`。一度误删（`e95d50a` 曾判定「官方不除」）→ 实测梯度大 128 倍、fp16 backward 上溢 220/256 nan、GradScaler skip step、arm B 全 0 delta——**删除法是错方向**；恢复 ÷len(pending) 才与官方同尺度且防溢出（`e69fba5` 定案，docstring 修正早期错误结论）。CE 语义随之从 unshifted（~12.0）转为 official shifted（~2.43）——数字跨度的剧变是协议对齐，不是方法差异。

**Gate 2 parity PASS**（`docs/ccm_experiments/parity_gate2.json`）：256 参数权重级对撞官方 HF Trainer 单步 vs 自定义循环——worst rel diff **1.42e-4**；分布桶 exact 138 / <1e-4 117 / <1e-2 1 / ≥5e-2 0；无 nan、无 skip step；参数量、scheduler 相位、autocast 内 CE 全部逐字复刻。

### 8.3 性能诊断工具（`4e04c6b`）

`CCM_PROFILE=1` 分段计时注入单窗口打印（pass1_fwd/collect_rows/window_add/close_replay/pass2_fwd/collect_z/surrogate/pass2_bwd/grad_step，默认关闭）——Gate 4 的测量基础。

## 9. 实验档案（云端 11 runs 全量落盘）

`docs/ccm_experiments/ccm_outputs_dump.json`（自 /root/autodl-tmp 拉回，字段完整）：

| run（目录） | 代码版本 | CE（语义） | 窗口/数据量 | kf_closed | data_flow hash |
|---|---|---|---|---|---|
| ccm_gate/ours | 410560a | 12.039（unshifted） | 1865 mb | 7 | 93e167935ab5 |
| ccm_gate/gamma_task_only | 410560a | 12.039 | 1865 mb | 7 | 93e167935ab5 |
| ccm_gate/ccm_merge | 410560a | 11.849 | 896 mb | 0 | 4e871a32b375 |
| ccm_gate2/ours | e69fba5 (HEAD) | **2.4333**（official shifted） | 1865 mb | 7 | 93e167935ab5 |
| ccm_gate2/gamma_task_only | e69fba5 | **2.4325** | 1865 mb | 7 | 93e167935ab5 |
| ccm_gate2/ccm_merge | e69fba5 | **2.4282** | 896 mb | 0 | 4e871a32b375 |
| ccm_gate2/prof_ours | e69fba5 + CCM_PROFILE | 2.4543（3 步） | 796 mb / 3 win | 3 | — |
| ccm_gate2/v1b_ours（漂移对照） | 410560a 重跑 | 12.0391 | 1865 mb | 7 | — |
| ccm_gate2/v3_ours（4b5d2f4） | 4b5d2f4 | 12.0390 | 1865 mb | 7 | — |
| ccm_parity | e69fba5 | — | 256 params | — | — |

要点：① shifted 语义下 ours(2.4333) ≈ gamma_task_only(2.4325) ≈ merge(2.4282)——首窗口 CE 无法区分方法（zero-init Γ + 热身期 + 监督未及显效，属预期）；② **ours == gamma_task_only 的 data_flow hash 逐字节一致**——RPBE 两臂共享同一任务样本/状态流，差异只在监督信号（对照有效性）；③ kf_closed merge=0：官方 merge 臂无 RPBE 窗口，符合设计。

## 10. L6.5 Gate 判定结果

### 10.1 Gate 1（数据流一致性）PASS
data_flow.jsonl 落盘（`5a73c1c`），hash 链见 §9 表。ours 与 gamma_task_only 一致 ⇒ Γ 不改变宿主状态流，三臂对比干净。

### 10.2 Gate 2（官方协议 parity）PASS —— 见 §8.2

### 10.3 Gate 3（smoke 语义）PASS
窗口语义 = 正式窗口语义（k≥4 有效 cut、≥128 trees 关闭），不是先前误以为的 8-cut 门槛——见 §11.2 的修正。

### 10.4 Gate 4（步时判定）**BLOCKED** —— 完整证据链

**目标**：Ours step time ≤ 1.5× CCM-merge（计划 v2 的硬标准；声称开销 1.33×）。**实测 5×**（99–102 s vs 18.8 s，同机同轮）。拆解（`docs/ccm_experiments/step_timing_gate4.json`）：

**① 版本二分（消去代码差异）**——控制机器漂移的双重实验：

| 版本 | 内容 | steps 2–6 mean | per-step 序列 |
|---|---|---|---|
| V1 | 410560a（首轮 gate run） | **85.7 s** | — |
| V1b | 410560a 重跑（同代码） | 92.5 s | [103.5, 90.1, 93.2, 93.7, 94.1, 91.6, 79.2] |
| V3 | 4b5d2f4（Γ scan 性能重构） | 99.3 s | [114.1, 99.1, 101.3, 99.2, 98.2, 98.8, 88.0] |
| V2/HEAD | e69fba5 | 101.5 s | — |
| merge（对照） | 同轮同机 | **18.4 / 18.8 s** | — |

结论：**机器漂移 ±8% 真实存在**（V1 85.7 → V1b 92.5，同代码隔 3h）；漂移修正后 4b5d2f4 仍是 ~+7% 净回退（V1b 92.5 → V3 99.3，相邻时段唯一变量），但相对 5× 差距是次量级。

**② 窗口阈值真相（规格误读修正）**：`KFMomentWindow._threshold = max(min_ratio × rank_dim, min_abs)`，rank_dim = min(z_dim=128, m=64) = 64，min_ratio=2.0 → **128 unique trees 关闭**（config "threshold": 128.0）。此前以为的 kf_min_cuts=8 门槛**从未生效**（8 < 128）；gate smoke 即正式窗口语义。

**③ 48% cut rate → 2.07× data**：1865 packs / 7 步（k<4: 969、k≥4: 896）→ ours ~266 mb/step vs merge 128 mb/step；k≥4 才产生监督行，pass1 只对它们做统计 forward（~128 mb），pass2 却要对窗口内**全部** ~266 mb 重放 fwd+bwd。

**④ probe 分段（CCM_PROFILE，每窗口 ~100 s）**：

| 段 | 耗时 | 含义 |
|---|---|---|
| pass1_fwd | 18.5 s | 128 mb（仅 k≥4）× fwd |
| pass2_fwd | 35.1 s | ~266 mb × fwd |
| pass2_bwd | 44.7 s | ~266 mb × bwd |
| RPBE 端合计（collect/window/adjoint/surrogate/grad_step） | ~1 s | 可忽略 |

⇒ 模型 fwd/bwd 是 100% 瓶颈。且 ours 单 mb 成本 ~0.30 s（fwd+bwd）≈ 2× merge 的 0.144 s——ours/gamma 臂每 mb forward 挂着 Γ 逐轮递归 scan + mem_callback 逐层提取，merge 臂不运行 RPBE（第二倍率因子）。

**⑤ 结构下界**：compute units = pass1 128×1 + pass2 266×2 = 660 vs merge 256 → **2.57×**。即使 Γ 附加成本归零也不可低于此 ⇒ **计划 v2 的 1.33× 隐含「~100% cut rate」假设错误**（1.33× 仅在 pass2 数据量 = pass1 数据量、即全部 mb 都产监督行时成立）；1.5× 目标在结构上不可达。

**待仲裁选项**（已提交 review-arbitration，`step_timing_gate4.json` 也收录）：

1. **回滚 4b5d2f4**（去 ~7% 净回退，恢复 410560a 步时）；
2. **窗口门槛审计**：min_ratio 2.0 → 1.0 → 64 trees → 每窗口 data 降至 ~135 mb/step 量级——但窗口变小伤 Welford 统计质量（C_ZZ 更噪），且估算仍在 2.6× 量级、min_ratio 不在 frozen_method.json 中（需规格批准，v2/9209d78 未授权）；
3. **SDPA/flash 等模型级加速**：不动 RPBE math，三臂同等受益（压缩 ② 的 2× 因子）；
4. **重校指标本身**：ours 1 step 消费 2.07× data 且监督行 265 条 vs merge 128 条——1 step 的信息量 ≈ merge 的 2 steps，1.5× 步时标准需与信息论对账后再定。

## 11. 关键发现汇总（跨 L0–L6）

1. **协议对齐是无声的深水区**：÷128（accelerate 内部除法）、shifted mean CE、attention_mask_comp fold、AdamW wd=0——任何一处缺失都让「对照」训练不同的东西；parity 的 1.42e-4 是这些细节全部对齐后的结果。
2. **窗口语义与规格假设不符**：128 unique trees（非 8 cuts）→ 有效 cut rate 只有 48% → 两遍 replay 承担 2.07× data——1.33× 开销声明建立在错误前提上（§10.4 ⑤）。
3. **RPBE 附加成本在模型路径而非统计端**：adjoint/Welford/surrogate 合计 ~1 s/窗口，可忽略；Γ scan + mem_callback 使 ours 每 mb forward ~2× merge。
4. **数据流 hash 证明对照有效性**：ours == gamma_task_only（93e167935ab5）≠ merge——三臂差的就是监督信号与 merge 机制本身。
5. **机器漂移 ±8% 使单点测量不可信**：任何性能判定必须同机同时段串行 + 版本二分（本报告 V1/V1b 对照即为模板）。
6. **本轮 5 个 smoke 真 bug 中 `__eou__` turn 拆分是最大的单点**：一个解析错误让训练跑在完全不同的序列结构上（每字符一 turn），且只有权重级 smoke 能抓到。

## 12. 当前状态与下一步

- **已完成**：L0–L6（测试 176+37 项全绿、三臂 smoke 通过）；L6.5 门禁 1–2 PASS、Gate 3 语义 PASS；Gate 4 证据链完整但**判定阻塞**——1.5× 目标结构上不可达（2.57× compute units 下限 + ~2× 单 mb 成本）。
- **等待用户对 Gate 4 的仲裁**（§10.4 四选项其一或组合）；仲裁落地前 L7 pilot 受硬标准阻塞不开跑。
- **L7–L9**（pilot 步时 ≤1.5× → 5-seed 正式 → matched-cohort depth NLL 曲线）在裁决后继续；报告口径、评估方法（per-token NLL 差 Δ_log(T) 防「随深度放大」假象）已按计划 v2 定稿。

## 附：commit 索引（main..develop_CCM，按交付序）

```
df0a4c4  [vendor] CCM 官方仓库 a89dd08（MIT）
938813a  [ccm] L0：clean_split（官方 train/validation/test）
845858d  [ccm] L1 兼容层：transformers 4.44/peft 0.4/accelerate 1.x/py3.12 + DIALOG_MIRROR
9181f1f  [ccm] L2：Γ 残差（zero-init 门控 + 任意 k 递归 + Test A + 工程闭环 1-6 + frozen_method.json）
0953073  [ccm] L2 修复：attach_gamma 跟随模型设备
196ab84  [ccm] L3-L6：host adapter + 2Obs 记录 + train_ccm 两遍 exact replay + 测试矩阵
fe80e89  [ccm] L5 修复：peft_custom 优先（comp_mask × peft 0.4）+ 行号/重复 CE
b1eb064  [ccm] L5 修复：fp16 autocast + PEFT 后重设 token
d2143f3  [ccm] L6 修复：__eou__ turn 拆分（3.6k-token 序列 OOM 根因）+ 移除 smoke 探针
c522770  [ccm] L6.5 审阅修复：P0-1 RNG 闭合 + P0-2 官方协议对齐 + 三臂 data_flow hash
5a73c1c  [ccm] L6.5 gate：data_flow.jsonl 落盘 + ccm_parity.py
410560a  [ccm] L6.5 修复：skip-pass1（k<4 无统计 forward）+ merge 输出路径遮蔽修复
4b5d2f4  [ccm] L6.5 性能：Γ scan → SUM-row gather + 寄存器递推（严格等价，gold 测试验证）
1a5e6c9/ d264f94/ e95d50a  [ccm] L6.5 parity 链：参数补全、out 遮蔽、官方 loss 三处对齐
e69fba5  [ccm] L6.5 gate2 定案：÷accumulation 是官方语义（accelerate 1.14 实测）——恢复 ÷128
4e04c6b  [ccm] L6.5 诊断：CCM_PROFILE 分段计时（本汇报写作时 HEAD）
```
