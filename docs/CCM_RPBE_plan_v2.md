# CCM-merge × RPBE 对接实现计划 v2（审阅修订版，develop_CCM 分支）

## Context

TGN 主实验已完成（exact_replay vs vanilla +1.53 点、5/5 全正）。按《CCM_LLM_recursive_memory_final_plan_20260831.md》做 LLM 跨域实验：官方 CCM-merge（ICLR 2024，snu-mllab/Context-Memory，MIT，vendored commit `a89dd08`）上验证"future-sufficient 递归压缩"的迁移。

**审阅裁决**：整体设计成立，但三个 P0 问题必须先改规格（L4/L5 暂停编码）；**L0/L1 立即开**。

**三个 P0 修正（本计划已吸收）**：
1. **P0-1**：单个 dialogue 单 cut 无法计算协方差（中心化后全零）→ L5 采用**跨 microbatch 的 exact 两遍 optimizer-window replay**（≥128 有效 cuts，按有效 cut 数自适应收集，三臂同任务样本同 cadence）。训练开销约 1.33×，仍满足 ≤1.5× 标准。
2. **P0-2**：两个 horizon **不能拼成单 p_v**（改变 C_PP 与白化几何）→ 一个 cut 两条 horizon rows、共享 cut_id、权重 0.5/0.5，进同一个 Ky-Fan 窗口。
3. **P0-3**：文档混用两代规格 → **冻结源 = 已完成五 seed TGN 实验的实际目标**：z_v = J_mem(M_v)（LLM 必需的固定适配器，TGN identity 是其特例）+ 两条 P rows；**主实验删除 rho_fix/witness/r_B/所有未被当前 loss 消费的 U/B 结构**。新增 `configs/ccm/frozen_method.json` 机器可读冻结文件（ridge=1e-3、m=64、horizon_weights=[0.5,0.5]、rpbe_map_seed=0、lambda_calibration=r_eff 规则，显式写死并记录来源，禁止静默读默认值）。

## 前置条件

1. 算力：DailyDialog 官方称 3090 24GB 可训；两遍 replay 后 pilot 先测峰值显存（当前实例 32.6GB），不够再租 A100。
2. LLaMA 权重：**必须原 LLaMA-7B（`llama-7b-no` 基线）**，不能用 LLaMA-2（否则三臂全部重跑、官方 PPL 不可比）；HF 转换 + path_config.py 指向。
3. Vendored commit 固定 `a89dd08e2c9587ec9c6c3ad339bb154c33e6b41a`。

## 目录/模块设计

```
third_party/ccm/                    # vendored 官方 repo（commit 锁定）
src/rpbe/hosts/ccm/
  adapter.py                        # SUM K/V 提取 → CutRecord 行（z_v 来自 J_mem）
  gamma_residual.py                 # Γ：R_θ(x) = s·Uσ(Vx)，U/V 小随机、仅 gate s=0 zero-init
  ccm_patch.py                      # merge 替换钩子（支持任意实际 k 的 recurrence，非仅 T≤12）
src/rpbe/llm/
  utterance_embed.py                # 冻结 input embedding 均值 + 固定 CountSketch
  mem_lift.py                       # J_mem（post-bottleneck 固定 lift；主实验无 witness）
  dialogue_records.py               # DialogueCutBuilder：一 cut → 两条 horizon rows（共享 cut_id）
scripts/
  train_ccm.py                      # 三组配置、paired seeds、1000 steps、两遍 replay 训练路径
  eval_ccm_depth.py                 # matched-cohort depth 曲线 + reference 评估
test/test_ccm_*.py                  # A-H + 新增 5 项（见 L6）
configs/ccm/frozen_method.json      # 冻结方法配置（机器可读）
```

**复用（冻结）**：`_score_from_covs`/`kf_adjoint`/`kf_vjp_batch`/`WeightedWelford`（loss.py）；CountSketch 构造模式（maps.py）；CutRecord 行约定与 clustered W2（records.py/loss.py）；paired-seed RNG 协议（checkpoint.py）。**注意**：现有核不消费 witness——所以主实验不构造 witness。

## 任务分解（审阅重排版）

### L0 — Fork 与协议清理（立即开）
- vendor 官方 repo、commit `a89dd08` 锁定；MIT 保留
- 双模式：**official-reproduction**（保留 pooled val+test 复现模式）+ **clean-split**（主实验用官方 train/validation/test）
- 不碰 tokenizer/preprocessing/random-prefix 采样

### L1 — 官方复现（立即开）
- Step1 LoRA → Step2 merge_recur 跑通；官方 pooled 模式 PPL 与论文量级一致
- depth 1/2/4/8/12 官方评估正常（作次要复现结果）
- 锁定 backbone = 原 LLaMA-7B

### L2 — Γ 接入（含完整工程闭环）
- merge 处挂 Γ：M_t = 均值 + R_θ(M_{t−1}, h_t, t)
- **R_θ 安全 zero-init**：U/V 小随机、仅 gate s=0（避免全零无梯度）
- **任意实际 k 的 recurrence**（训练 random prefix 可 >12；时间编码与 recurrence 支持任意 k，加 k>14 测试）
- **工程闭环 6 项**：① PEFT wrapping 后 Γ requires_grad 确认 ② optimizer param groups 含全部 Γ ③ checkpoint 显式保存/恢复 Γ ④ roundtrip 容差一致 ⑤ paired seed hash 含 LoRA+Γ+COMP/SUM embedding ⑥ 首个 step 后 Γ 参数确实变化
- torch.compile：要么 padded masked scan 并测 eager/compile 等价，要么三臂一致禁用 compile
- 每 run 显式锁 n_tok=2
- RoPE：测试同类 COMP slot 各轮位置基准一致
- 验收：Test A（zero-init 数值复现算术 merge）、Test B 改为 **Ours 训练 = 一次 no-grad 统计 forward + 一次有梯度 replay forward**（推理仍单 forward）、参数量 <1M

### L3 — Host state adapter
- 确定性 collator metadata：保留 k、utterance token spans、padding 后 COMP/SUM positions、sample/cut IDs（进模型前剥离）
- 提取带梯度的 SUM K/V → J_mem（layer/head/slot 索引进固定 hash、结构化 CountSketch、requires_grad=False）→ z_v
- **主实验不加 witness**
- 验收：Test C 强化版——替换 cut 后**全部**未来输入 u_{k−2}, u_{k−1}, u_k（保持替换长度相同），M_{k−3} 与 z_v 完全不变

### L4 — 2Obs record（改规格后）
- 仅 k≥4；v=k−3
- **一个 cut → 两条 horizon rows**：
  - row1：(z_v, p_{v,1}, ω=0.5, cut_id)
  - row2：(z_v, p_{v,2}, ω=0.5, 同一 cut_id)
- p_{v,1}=TensorSketch([1;χ₁]⊗φ₁)、p_{v,2}=TensorSketch([1;χ₂]⊗φ₂)（χ₂ 含 one-update 标识）
- 进同一个 Ky-Fan 窗口（clustered correction 走既有 WeightedWelford 路径）
- k=3：task CE 保留、w_RPBE=0
- 验收：Test D/F/E

### L5 — Loss 集成（跨 microbatch exact 两遍 replay）
- 优化器窗口 = ≥128 有效 cuts（k≥4 的 dialogue），按有效 cut 数自适应收集（非机械 128 microbatch）
- pass1 no_grad：累计 WeightedWelford → kf_adjoint
- 恢复输入/RNG/模型状态 → pass2：task CE + kf_vjp_batch → 一次 optimizer step
- 三臂使用完全相同任务样本与 scheduler cadence
- λ 不继承数值：用 **r_eff 校准规则**（training-only 初始完整窗口校准一次，然后所有 seed 冻结同一 λ）
- 验收：pass1/pass2 状态与 RNG 重放一致、窗口非退化（C_ZZ 全零即失败）、训练开销 ≤1.5×

### L6 — 测试 A-H + 新增 5 项
新增：① RPBE/Γ/LoRA 梯度全非零 ② checkpoint roundtrip ③ pass1/pass2 状态与 RNG 重放 ④ window 非退化 ⑤ eager/compile 一致性
构造级本机跑，权重级 GPU 跑

### L7 — 1-seed pilot（先 smoke 后 full）
- 先 5-20 optimizer steps 功能 smoke（有限值/梯度/状态/性能/辅助目标生效）——**单 seed PPL 排序不作停止条件**
- 再三臂各 1000 steps 单 seed pilot
- 硬标准：Ours step time ≤1.5× CCM-merge；>2× 查规格 §21 禁令

### L8 — 5-seed 正式实验
- 15 runs（3 配置 × 5 paired seeds）
- 报告：test PPL mean±std、per-seed paired PPL、Ours−Γtask-only 配对差、效率 4 项

### L9 — 最终评估
- **matched-cohort depth**：选 clean-test 中支持最大深度（≥12）的同一批 dialogue，每个对话都评 1/2/4/8/12（官方各 depth 原始集合作次要复现）
- **主指标 = per-token NLL 差 Δ_log(T)**（PPL 的指数会使同一 NLL 改善在高难度深度表现为更大的绝对差——防"随深度放大"假象）；PPL 曲线继续展示
- evaluation-only reference（No Context/Full Context/CCM-concat/released merge）
- 稳定性记录：‖R_t‖/‖M_t^base‖ 与 memory norm 随深度变化（防递归残差爆炸）
- 输出 per-seed JSON/CSV；不按 test 改超参

## 关键风险

| 风险 | 对策 |
|---|---|
| 两遍 replay 显存/时间 | pilot 先测；理论 1.33×，硬标准 ≤1.5× |
| LLaMA-7B 权重（非 LLaMA-2） | 原版权重获取困难则三臂全部重跑（与官方 PPL 不可比），需先确认权重可得性 |
| 有效 cuts 不足 128 | 按有效 cut 自适应收集；DailyDialog 短对话 k=3 比例需 L1 审计 |
| torch.compile 与动态 recurrence | 三臂一致禁用 compile（默认）或 padded masked scan 测等价 |
| Γ 递归残差爆炸 | 记录 norm 比值；gate zero-init + 梯度裁剪沿用 TGN 协议 |

## 验证方式

1. L1：官方 pooled PPL 与论文量级一致
2. L6：Test A-H + 新增 5 项全绿
3. L7：smoke 有限值/梯度生效；pilot step time ≤1.5×、aux 生效、Γ 离开 zero-init
4. L8：5-seed 配对差 + NLL depth 曲线符合"深层递归信息保持"
5. 全程：commit hash、frozen_method.json、per-seed 指标落盘

---

# v3 修订（第二轮审阅封口，2026-09-01）

## 三点必须写进规格的修正

### 修正 1：窗口门槛不能用 128——TGN 实际冻结值是 1024

`2m=128` 只是维数层面的最低门槛。TGN 五 seed 正式实验实际使用 `kf_min_abs=1024`（`run_multiseed_TGN.sh` 未覆盖，`RPBConfig` 默认），窗口按**独立 tree** 计数。LLM 直接改 128 不能写成"继承 TGN"，且 d_z=m=64、N=128 处于小样本 CCA 虚高区。

**修正**：
- L1 后增加 **estimator validity audit**（不训练、不看 PPL）：固定初始 checkpoint + training split，扫 `N ∈ {128, 256, 512, 1024}`，检查：real J vs dialogue-level permutation J、J/min(d_z,m) 是否接近虚假满秩上限、条件数、有效秩、相邻窗口规模 exact-gradient cosine、Cholesky failure rate。
- 预先规定"选最小稳定 N"后冻结。这是统计估计器有效性检查，不是 LLM 性能 sweep。
- `frozen_method.json` 新增：
  ```json
  {"kf_min_abs": "<audit-selected>", "kf_min_ratio": 2.0,
   "window_unit": "unique_dialogue_id"}
  ```
  窗口门槛数**独立 dialogue**，不数两条 horizon rows。

### 修正 2：L5 明确 loss 符号与跨 microbatch 缩放

- 含 K 个 microbatches 的优化器窗口目标：
  `L_group = (1/K)·Σ_b L_task,b − λ·J_norm`，单 CCM 接口下 `J_norm = J/min(d_z,m)`。
- 每 microbatch 反传：`L_task,b/K − (λ/min(d_z,m))·s_b`（Σ∇s_b = ∇J），窗口末**一次** optimizer step。
- **禁止**依赖 HF 按固定 `gradient_accumulation_steps=128` 自动除（自适应 K 会重复/错误缩放）。
- 明确：用官方每 dialogue 的 CE 标量再除实际 K；`p/means/adjoints` 全部 detach；k=3 只有 task 项；训练恰好 1000 个完整 optimizer windows；checkpoint 只落窗口边界；三臂读取同一 `sample_id + random_k + window_boundary` manifest。

### 修正 3：matched cohort 必须固定同一预测目标

官方各 depth 取不同长度前缀 → 不同深度预测不同 utterance。改为**固定 current/target、只裁剪历史**（suffix-depth）：

| Depth | 压缩历史 | 当前输入 | 固定目标 |
|---|---|---|---|
| 1 | u₁₂ | u₁₃ | u₁₄ |
| 2 | u₁₁:₁₂ | u₁₃ | u₁₄ |
| 4 | u₉:₁₂ | u₁₃ | u₁₄ |
| 8 | u₅:₁₂ | u₁₃ | u₁₄ |
| 12 | u₁:₁₂ | u₁₃ | u₁₄ |

- depth 12 需要 **≥14 turns**（不是 ≥12）。
- 主机制量 `Δ_log(T) = NLL_Γtask(T) − NLL_Ours(T)` 在**完全相同的 target tokens** 上计算。
- **预注册**深浅对比：`(Δ_log(8)+Δ_log(12))/2 − (Δ_log(1)+Δ_log(2))/2`，禁止结果后肉眼定义趋势。

## 五个规格细节补齐

1. **L4 精确索引写回**（Test D 唯一真值）：v=k−3；C₁=u_{k−2}, Y₁=u_{k−1}；C₂=(u_{k−2}, u_{k−1}, one-update), Y₂=u_k。
2. **frozen_method.json 扩展冻结**：kf_variant=full_balancing、d_z/J_mem 输出维、J 归一化规则、d_c/d_f、全部 CountSketch/TensorSketch seeds、Γ rank/activation/gate shape、K/V 是否分模块、layer/head/slot 共享规则、时间编码、RPBE 接收梯度的参数集合、checkpoint_selection=final_step_1000。
3. **J_mem 不 detach 输入**：`requires_grad=False` 只表示固定映射参数不训练；pass2 必须保留 `M_v → J_mem(M_v) → J` 的梯度路径。K/V flatten 顺序固定为 `[layer, K_or_V, head, COMP_slot, head_dim]`。
4. **L6 Γ 梯度口径修正**：gate s=0 时 step 0 的 U/V 梯度为零是**正常**。正确断言：① step 0 Γ 模块总梯度非零、gate 梯度非零 ② gate 更新后后续 step 的 U/V 梯度非零 ③ Γ 输出开始偏离零。
5. **pass1/pass2 缓存 collated tensors**：不重新调用带随机 `sample_dialog` 的 collator；先缓存确定 collated CPU tensors，恢复 RNG 后只重放模型。断言：`row_ids_pass1 == row_ids_pass2`、`cut_ids_pass1 == cut_ids_pass2`、`max_abs(z_pass1 − z_pass2) <= dtype 容差`、`RNG_after_pass2 == RNG_after_one_control_forward`。

## Test A 再强化

- 官方 `sum_attn_mask @ key/value_states` 继续作为**精确 base mean**；Γ 只递归计算 residual；最终写回 `official_mean + residual`（避免 FP16/FP32 顺序递推的加法序差异）。
- Test A 比较：每层/每 head/每 slot 的 merged K/V + 最终 logits + gate=0 时与官方 CCM 一致。

## 最终裁决表

| 模块 | 裁决 |
|---|---|
| L0/L1 | 立即开 |
| L2/L3 | 可并行实现，先冻结 Γ/J_mem 完整结构参数 |
| L4 | 补回 C₁/Y₁/C₂/Y₂ 后开 |
| L5 | 补窗口审计 + loss 公式 + 可变 K 缩放后编码 |
| L7–L8 | estimator validity audit 通过后启动 |
| L9 | 改为固定 target 的 suffix-depth 评估 |
