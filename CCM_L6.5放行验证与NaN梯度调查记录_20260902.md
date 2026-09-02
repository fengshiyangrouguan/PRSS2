# CCM-merge × RPBE：L6.5 放行验证与首窗口 NaN 梯度调查记录

日期：2026-09-02（云端实例 nma1:54969，develop_CCM 分支）
背景：裁决 #2 放行四项验收 → 云端三臂单窗口验证 → 发现首窗口后参数全零 → NaN 调查链 → 根因定案 → 四验收全过 → **L6.5 PASS**
相关：`configs/ccm/frozen_method.json`（冻结方法）、`scripts/train_ccm.py`（诊断插桩 commit `5d7dd7c`）

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
| BAD after bwd | 257/257 mb 全 BAD | 257/257 mb 全 BAD |
| BAD 参数 | layer0 `q_proj.lora_A/lora_B`（+ k_proj.lora_A 第 3 名） | **完全相同** |
| win-grads | norm=nan (scaled) scale=65536 | 相同 |
| grad_step | applied=**False** scale_now=32768 | 相同 |
| hash 对照 | — | `paired_seed`/`data_flow`/`boundary` 与 gamma **逐字节相同**；`aux_terms=128` ✓ |

**发现 1**：gamma（纯 task、λ=0）与 ours 完全同模式 ⟹ **NaN 与 aux 路径无关**，是 task backward 的普遍问题（排除假设 B）。

**发现 2**：连 `aux.requires_grad=False`（该 mb 无 aux 项）的 mb 也 BAD ⟹ 再次确认与 aux 无关。

**发现 3（副产品）**：验收(2) cadence 对撞在双跑中直接达成（hash 全同 + aux_terms=128）。

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

---

## 5. 最终结论

1. **NaN 根因**：GradScaler 冷启动（初始 scale=65536 下 fp16 反向中间梯度溢出 → 首步必 skip → 自动降 scale 后恢复）。**非代码 bug、与 aux 无关、与三臂差异无关。**
2. **"三臂停在 θ₀"是假警报**：单窗口跑只有 1 次 grad_step、恰好是冷启动 skip；正式 1000-step run 前 1-2 步 skip 后正常。
3. **四验收全过 → L6.5 PASS**：

| # | 验收项 | 状态 | 证据 |
|---|--------|------|------|
| 1 | 本机全量测试 | ✓ | 177 passed / 26 skipped |
| 2 | 三臂 cadence 对撞 | ✓ | diag2/diag4 hash 逐字节同、aux_terms=128/窗口 |
| 3 | calibration 参数不变 | ✓ | params_unchanged、0 次 step |
| 4 | ours vs gamma 非零更新差 | ✓ | diag4：158/384 参数非零差、32/32 gate s |
4. **gate s 离开零初始化**（Γ 训练生效）与 **task CE 下降**（训练真正推进）双确认。
5. **教训沉淀**：验收跑 ≥2 窗口；NaN 判别看 unscale 后有限性 + scale 减半后是否恢复；`steps` 计数器 ≠ 参数更新证据。

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
