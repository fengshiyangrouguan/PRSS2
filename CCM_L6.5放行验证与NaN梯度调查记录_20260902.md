# CCM-merge × RPBE：L6.5 放行验证与首窗口 NaN 梯度调查记录

日期：2026-09-02（云端实例 nma1:54969，develop_CCM 分支）
背景：裁决 #2 放行四项验收 → 云端三臂单窗口验证 → 发现首窗口后参数全零 → NaN 调查链 → 根因定案（GradScaler 冷启动）→ 审阅裁决指出 AMP skip 后 scheduler 协议错误 → 修复 + 三臂 3 窗口重新验收六项全过
**当前状态：L6.5 PASS**（accept5 三臂 3 窗口六项验收全过，见 §4.5/§5）
相关：`configs/ccm/frozen_method.json`（冻结方法）、`scripts/train_ccm.py`（诊断插桩 commit `5d7dd7c` + AMP-skip 修复 commit `6ae50dd`）

---

## 1. 背景：放行验收四项（裁决 #2）

| # | 验收项 | 方法 |
|---|--------|------|
| 1 | 全量本机测试 | pytest/unittest 全绿 |
| 2 | 云端三臂 cadence 对撞 | data_flow / paired_seed / boundary hash 一致；ours `aux_terms==128` |
| 3 | calibration 参数不变 | calibrate-only run：params_unchanged、0 次 optimizer/scheduler step |
| 4 | ours(λ>0) vs gamma_task_only **非零更新差** | 同样本同 cadence 跑两臂，diff final.pt |

验收触发条件：所有验收跑都在云端执行；**任何 run 启动前需用户授权**。

---

## 2. 触发现象与初始证据

**现象**：首个三臂单窗口验收跑（max_windows=1）完成后 diff 三个 `final.pt`，发现 `n_different_tensors=0`——三个臂的最终参数**完全相同且停在初始化**：

- 32 个 gate `s` 张量（Γ 的 zero-init gate）**全零**
- 全部 `lora_B`（peft 零初始化）**全零**
- summary.json 显示 `"steps": 1`

**初始误判**：曾解读为"surrogate 梯度缺失导致 ours 退化成 gamma"（呼应 P0-1 的历史问题）。此解读在后续被推翻。

**同时澄清的两件事**：
1. `"steps": 1` 是无条件计数器——`scaler.step()` 检测到非有限梯度时会**跳过** optimizer 更新，但 `step += 1` 照样执行。`steps:1` 不能证明参数更新过。
2. `aux_abs=0.000e+00`（数值零代理）是**预期行为**：`aux = -λ·Σ[(g·z).sum() − (g·z.detach()).sum()]` 恒为零、精确梯度 −λ·g——不能作为"梯度缺失"证据。

**第一个硬线索**：云日志 `grad_norm=nan (scaled)` —— 梯度是 NaN。结合 gate `s` 全零 ⟹ **首次 grad_step 被 GradScaler skip**（skip 条件即梯度非有限）。

---

## 3. 诊断插桩（CCM_AUX_DIAG=1 门控，train_ccm.py）

问题：现有插桩只在 scaled 域看 norm，无法定位 (a) 哪个参数携带 NaN、(b) unscale 后是否仍 NaN、(c) 参数是否真的更新。新增（全部 ENV 门控，正式 run 默认关闭）：

1. `bad_grad_names(params, k=3)`：返回前 k 个 grad 非有限的参数名。
2. pass2 loop 内每个 microbatch backward 后：`AUXDIAG mb={i} BAD after bwd: [...]`。
3. window 汇总处：`AUXDIAG win-grads norm=... (scaled) scale=... bad=...`。
4. `grad_step()` 内：
   - `unscale_` 之后加**新检查点**（commit `5d7dd7c`，区分 scaled 溢出 vs raw NaN）：`AUXDIAG unscale norm=... bad=...`。
   - step 前后 `params_digest` 对比 → `AUXDIAG grad_step applied=True/False scale_now=...`。

**部署**：`scp scripts/train_ccm.py` → 云端 `/root/autodl-tmp/scripts/train_ccm.py`（云端无 git 直传目录，走 /tmp 中转）。

---

## 4. 实验时间线

### 4.1 diag1（ours 单臂，max_windows=1）——首次带插桩跑

结果：单窗口即确认 `grad_norm=nan (scaled)`、首次 grad_step skip、gate 全零。
未决问题：NaN 是 (A) fp16 大规模 scaled 梯度溢出（通病）还是 (B) aux 路径特有（z-grad = −λ·g 经 fp16 图传播）？

### 4.2 diag2（gamma_task_only + ours 双臂，max_windows=1，CCM_AUX_DIAG=1）——对照

**启动事故**：/tmp/ccm_diag2.sh 参数名误用下划线（`--model_name_or_path`），而 train_ccm.py 用连字符定义（`--model-name-or-path`）→ argparse 报两个 required 缺失、启动即退。修复：全部选项改连字符，重发成功。

**结果（决定性对照）**：

| 观察 | gamma_task_only (λ=0) | ours (λ=6.768e-4) |
|------|----------------------|-------------------|
| BAD after bwd | 每个被检查的 mb 都报 BAD | 每个被检查的 mb 都报 BAD |
| BAD 参数 | layer0 `q_proj.lora_A/lora_B`（+ k_proj.lora_A 第 3 名） | **完全相同** |
| win-grads | norm=nan (scaled) scale=65536 | 相同 |
| grad_step | applied=**False** scale_now=32768 | 相同 |
| hash 对照 | — | `paired_seed`/`data_flow`/`boundary` 与 gamma **逐字节相同**；`aux_terms=128` ✓ |

**关于 BAD 计数的严格表述**（审阅修正）：梯度是跨 mb 累计的——一旦某个早期 microbatch 写入 inf，后续 backward 检查会持续看到 inf。因此"257/257 mb 全部 BAD"的严格含义是：**非有限梯度最迟在第一个被报告 BAD 的 microbatch 出现，并在整个累计窗口内持续存在**，不能解读为 257 个 mb 各自独立溢出。这不影响"NaN 在 task 路径而非 aux 路径"的结论（gamma λ=0 同模式 + `aux.requires_grad=False` 的 mb 也 BAD）。

**发现 1**：gamma（纯 task、λ=0）与 ours 完全同模式 ⟹ **NaN 与 aux 路径无关**，是 task backward 的普遍问题（排除假设 B）。

**发现 2**：连 `aux.requires_grad=False`（该 mb 无 aux 项）的 mb 也 BAD ⟹ 再次确认与 aux 无关。

**发现 3（副产品）**：验收(2) cadence 对撞在双跑中直接达成（hash 全同 + aux_terms=128）。

**三臂 cadence 完整 JSON 证据**（审阅要求：三臂数据并列展示，不得只列双臂）：

第一轮三臂单窗口（outputs/ccm_accept/{merge,gamma,ours}/summary.json，修复前协议）：

| arm | data_flow_len | data_flow_hash | boundary_hash | task_microbatches | task_valid_tokens | aux_terms | kf_closed | lambda_kf |
|---|---|---:|---|---|---:|---:|---:|---:|
| ccm_merge | 257 | 802649…aff7 | 87a0cb…4000 | 257 | 4233 | 0 | 0 | 1e-3（臂内不消费） |
| gamma_task_only | 257 | 802649…aff7 | 87a0cb…4000 | 257 | 4233 | 128 | 1 | 1e-3（臂内强制 0） |
| ours | 257 | 802649…aff7 | 87a0cb…4000 | 257 | 4233 | 128 | 1 | 6.768e-4（frozen） |

三臂 `data_flow_hash`/`boundary_hash`/`task_microbatches`/`task_valid_tokens` **逐字节相同** → 相同对话流、相同窗口边界、相同任务暴露。amp_skips 字段在修复前协议中不存在（旧协议未记录），修复后的三臂 3 窗口（accept5）带完整字段见 §4.5。

**遗留问题**：BAD 是 scaled 域现象；需要知道 unscale 后（raw）梯度是否有限 → 设计 diag3。

### 4.3 diag3（ours 单臂，max_windows=3，unscale 检查点）——根因判定

**结果（三窗口全链证据）**：

| 窗口 | win-grads (scaled) | unscale 后 | grad_step |
|------|-------------------|-----------|-----------|
| 1 | norm=**nan** @ scale=65536 | bad 仍在（q/k/v_proj lora） | applied=**False** → scale 减半 32768 |
| 2 | norm=**4705.76**（有限） | norm=**0.1436** bad=**[]** | applied=**True** |
| 3 | norm=**3275.86**（有限） | norm=0.1000 bad=[] | applied=**True** |

配套证据（final.pt / summary）：
- 32/32 个 gate `s` **已离开零初始化**（s[0] absmax=2.999e-5——2 次更新的合理量级）
- 254/256 LoRA 参数已变（剩 layer31 两个 `lora_B` 恰为零，非阻塞）
- task CE per-token：2.4649 → 2.4568（下降）
- steps=3、aux_terms=384（3 窗口 × 128 ✓）

**根因判定（正式）**：**GradScaler 冷启动**，非代码 bug。机制：
- GradScaler 初始 scale=65536；fp16 autocast 反向中，中间张量梯度 > ~1 即超 fp16 上限（65504）溢出成 inf；
- inf 一旦产生不可逆 → 叶子梯度 inf → `unscale_` 后 inf/65536 **仍是 inf**（这解释了为什么窗口 1 的 unscale 检查 BAD 仍在——这是"scaled 溢出"的特征，不是 raw NaN）；
- `scaler.step()` 检测非有限 → skip 本次更新 → **scale 自动减半**（65536→32768）；
- scale=32768 下 raw 梯度（norm~0.14，单分量 <0.15）完全不溢出 → 窗口 2 起 `applied=True` 正常训练。

**判别方法**（区分 scaled 溢出 vs raw NaN 的判据）：raw NaN（真 bug）在 scale 减半后**永不恢复**；scaled 溢出在 scale 降到合适值后**立即恢复有限**（diag3 窗口 2 即恢复）。

官方 HF Trainer（vendored CCM 同款 AMP 协议）同样机制——首次 step skip 是正常冷启动，无人在意因为长 run 有大量 step。

**教训**：单窗口验证只有 1 次 grad_step，恰好撞上冷启动 skip → "参数全零"是**假警报**。**放行/验收类跑必须 ≥2 窗口**。

### 4.4 diag4（gamma_task_only + ours 双臂，max_windows=2）——验收(4) 完成

**结果**：

| 指标 | gamma_task_only | ours |
|------|----------------|------|
| steps | 2（win1 skip → win2 applied） | 2（同） |
| unscale 后 win2 | norm=0.1459 bad=[] | norm=0.1439 bad=[] |
| data_flow_hash | f860e8...2fc6 | **相同** |
| boundary_hash | 8448e7...1d5e | **相同** |
| mean_task_ce | 2.4425 | 2.4425（前 2 步 task 一致验证 RNG/cadence 协议） |

**final.pt 逐参数 diff（验收(4) 判定证据）**：
- key 集合相等（384 keys）
- **158/384 参数有非零差异**，top：各层 `v_proj/q_proj/o_proj.lora_B` maxdiff ≈ 2.0e-5
- **32/32 个 gate `s` 有非零差异**（max ≈ 1.99e-5）

两臂从相同初始化出发、任务样本/窗口边界逐字节相同、skip 节奏相同——唯一差别是 ours 多出 `λ·g_KF`（aux）梯度 → 参数分歧是 **aux 项真实生效并改变更新路径**的直接证据。差异量级 ~1e-5 与"1 次 grad_step × lr=3e-4"相符。

### 4.5 AMP-skip scheduler 协议修复（审阅裁决 pending 项）+ accept5 三臂 3 窗口重新验收

**审阅裁决指出的小而真实的协议错误**：修复前 `grad_step()` 无条件执行 `scheduler.step()`——当第一窗口 overflow 时 `scaler.step()` 已 skip optimizer 更新，但 scheduler 仍前进一步。HF Trainer 4.44.2 的正式行为是：只有 optimizer 没被 skip 时才推进 scheduler（global step 仍增加）；Accelerate 文档明确 overflow skip 时不应改变学习率。因此 diag4 第二窗口能立即产生参数变化，部分原因是第一窗口虽 skip 却提前推进了 warmup scheduler。

**修复**（scripts/train_ccm.py）：
- `grad_step()` 返回 `applied`；AMP skip 判定 = `scaler.is_enabled() and scale_after < scale_before`（scale 减半是 Accelerator 同款信号；**不用 params_digest**——digest 无法区分"optimizer 以 lr=0 执行"与"optimizer 被 skip"）
- skip 时不推进 scheduler；`scheduler.step()` 只在真实 optimizer 执行时调用
- 三计数分离：`steps`（global/window attempts，HF global_step 口径，skip 也 +1）、`optimizer_steps_applied`（真实 optimizer 执行数）、`amp_skipped_steps`（overflow skip 数）；三者 + `amp_scale` 全部写入 summary.json、_SUCCESS.json、checkpoint.pt（含 resume 恢复）
- 口径结论：若论文声称"1000 次实际 optimizer update"，必须跑到 `optimizer_steps_applied=1000`；否则应写"1000 global/training steps"

**本机验证**：py_compile 通过；GradScaler 语义模拟（CPU：正常步 scale 不变→applied、inf 步 65536→32768→skipped、enabled=False 恒不 skip）通过；test_ccm_* 四个构造级测试 38 passed。

**accept5 三臂 3 窗口重新验收**（串行：三臂并行 ~41GB > 实例 32.6GB；max_windows=3；CCM_AUX_DIAG=1）——验收 6 项：

1. 三臂 data-flow/boundary/microbatch/token 完全一致
2. 三臂共同的冷启动 skip 数一致
3. `aux_terms=384`（3 窗口 × 128）
4. calibration 参数仍不变（先前已过，不受本修复影响——calibration 分支在 grad_step 之前）
5. 至少一次非零 LR 的实际 optimizer update 后，ours 与 gamma 参数产生非零差
6. scheduler step 数等于 `optimizer_steps_applied`，而非 global step

**accept5 结果（六项全过 → L6.5 正式 PASS）**：

| arm | steps | optimizer_steps_applied | amp_skipped_steps | scheduler_steps | amp_scale | data_flow_hash | boundary_hash | aux_terms |
|---|---|---:|---:|---:|---:|---|---|---:|
| ccm_merge | 3 | 2 | 1 | 2 | 32768 | d13058…4d09 | 5d3fbf…428f | 0 |
| gamma_task_only | 3 | 2 | 1 | 2 | 32768 | d13058…4d09 | 5d3fbf…428f | 384 |
| ours | 3 | 2 | 1 | 2 | 32768 | d13058…4d09 | 5d3fbf…428f | 384 |

六项逐条：
1. ✓ 三臂 data_flow_hash/boundary_hash 逐字节同；data_flow_len=796、task_microbatches=796、task_valid_tokens=12986 三臂全同
2. ✓ 三臂冷启动 skip 数一致：amp_skipped_steps=1、optimizer_steps_applied=2
3. ✓ aux_terms=384（3 窗口 × 128；merge 不消费 aux=0 正常）
4. ✓ calibration 参数不变（先前已过；calibration 分支在 grad_step 之前，本修复不影响）
5. ✓ 至少一次非零 LR 真实 update 后 ours vs gamma 非零差：**158/384 参数非零差、32/32 gate s 非零差**（maxdiff ~2e-5，与 diag4 同量级）
6. ✓ scheduler_steps == optimizer_steps_applied == 2，而非 global steps=3 → **修复生效的直接证据**

**三臂 grad_step 轨迹（逐窗口）完全一致**：

| 窗口 | AUXDIAG 轨迹 | 语义 |
|------|-------------|------|
| 1 | applied=False **amp_skipped=True** | 冷启动 overflow skip，scheduler 不推进（修复前会错误推进） |
| 2 | applied=False amp_skipped=False | optimizer 执行但 **lr=0**（scheduler s=0，warmup 首步）——审阅者预言的"digest 无法区分 lr=0 执行与 skip"的活例子 |
| 3 | applied=True amp_skipped=False | lr=1e-5，真实参数更新 |

mean_task_ce 三臂同 2.456878（任务轨迹一致）。

---

## 5. 最终结论

1. **NaN 根因**：GradScaler 冷启动（初始 scale=65536 下 fp16 反向中间梯度溢出 → 首步必 skip → 自动降 scale 后恢复）。**非方法 bug、与 aux 无关、与三臂差异无关。**
2. **"三臂停在 θ₀"是假警报**：单窗口跑只有 1 次 grad_step、恰好是冷启动 skip；正式 1000-step run 前 1-2 步 skip 后正常。
3. **AMP skip 后 scheduler 无条件推进是协议 bug**（审阅裁决）：已修复为 Trainer 4.44.2 口径（skip 不推进 LR），三计数分离进 summary/checkpoint。**修复前的四验收成立结论中，"冷启动不用管"的说法只对一半**——skip 本身正常，但 skip 时的 scheduler 处理必须符合官方协议。
4. **四验收状态（修复前口径）**：

| # | 验收项 | 状态 | 证据 |
|---|--------|------|------|
| 1 | 本机全量测试 | ✓ | 177 passed / 26 skipped |
| 2 | 三臂 cadence 对撞 | ✓ | 三臂 hash 逐字节同（见 §4.2 表）、aux_terms=128/窗口 |
| 3 | calibration 参数不变 | ✓ | params_unchanged、0 次 step |
| 4 | ours vs gamma 非零更新差 | ✓ | diag4：158/384 参数非零差、32/32 gate s |

5. **当前状态：L6.5 PASS（正式）** —— RPBE replay、λ 校准、aux 生效、matched cadence 核心实现已通过；AMP-skip-aware scheduler 修复后三臂 3 窗口六项验收全过（accept5，见 §4.5）。下一步：L7 单 seed pilot（先 5-20 step smoke，再 1000-step full，待授权）。
6. **gate s 离开零初始化**（Γ 训练生效）与 **task CE 下降**（训练真正推进）双确认。
7. **教训沉淀**：验收跑 ≥2 窗口（单窗口撞冷启动 skip）；NaN 判别看 unscale 后有限性 + scale 减半后是否恢复；`steps` 计数器 ≠ 参数更新证据；**scheduler 只在 optimizer 真执行时推进（HF 口径）**；AMP skip 判定用 scale 减半（Accelerator 同款信号），不用参数 digest。

---

## 6. 产物清单

**云端**（/root/autodl-tmp/outputs/ccm_accept/）：
- `ours_diag / gamma_diag`：首轮（nan 证据）
- `ours_diag2 / gamma_diag2`：双臂对照（BAD 全 257 mb、hash 全同）
- `ours_diag3`：三窗口根因判定（unscale 检查点、gate 离开 0）
- `gamma_diag4 / ours_diag4`：双臂 2 窗口验收(4)（158/384 非零差）
- 每目录含 summary.json / config.json / data_flow.jsonl / final.pt

**代码**：
- `scripts/train_ccm.py`：CCM_AUX_DIAG 诊断插桩（commit `5d7dd7c`，ENV 门控）
- 启动脚本 /tmp/ccm_diag{2,3,4}.sh（云端）

**下一步（L7，待授权）**：先 5-20 optimizer steps 功能 smoke → 再 single-seed 1000-step pilot（先 smoke 后 full；不直接启动正式 run）。
