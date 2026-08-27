# 云端训练两次数值崩溃：报告 + 外部审阅修正（2026-08-27）

**任务背景**：正式矩阵（wiki，L=3，n_degree=5，bs=100）三个任务并行。任务①（s0 全 RPBE）与任务③（s1 全 RPBE）在阶段一 epoch 18/19 首次崩溃；Cholesky 修复后续跑，任务①在 epoch 20 第二次崩溃。

**重要**：初版报告将两次崩溃诊断为"同一个根因（z 尺度漂移）的两种数值表现"。**外部审阅推翻了这一诊断**：J>172 在数学上不可能仅由尺度缩小导致，Cholesky pivot 非正也不等于某个原始方向的方差塌缩。修正后的结论是：**崩溃暴露的是"白化统计实现的不一致 + 高维小样本 CCA 饱和"，不是已证实的表示分解坍缩**。下文按"崩溃事实 / 初版错误诊断 / 审阅修正"三段记录，修复方案（§5）按审阅意见重写。

---

## 1. 崩溃一：Cholesky 非正定（原代码）

### 1.1 报错原文

```
torch._C._LinAlgError: linalg.cholesky: The factorization could not be
completed because the input is not positive-definite
(the leading minor of order 171 is not positive-definite).
```

- 发生位置：阶段一 epoch 18/19 的窗口关闭时刻（任务①③同速崩溃，间隔 ~40s）
- 涉及矩阵：**172×172 的 C_ZZ**（state_dims[tau] = 172 = 每层压缩宽度）

### 1.2 代码位置（崩溃时的实现）

```
src/rpbe/loss.py — _cholesky_retry（原实现，commit 13697e4 前）
调用链：
  KFMomentWindow.add (loss.py)
    → _j_from_covs
      → _cholesky_retry(czz + _ridge(eps, czz)·I)
        → torch.linalg.cholesky  ← 抛 _LinAlgError
```

### 1.3 初版诊断（错误的部分）

初版报告声称：崩溃机制是"训练使 z 结构退化——第 171 个方向方差 ≈ 0（最后方向塌缩模式）"，佐证是"各层共享递归信号、层间高度相关、Linear head + GELU 饱和压平方向"。

**审阅修正**：
- **Cholesky pivot 非正 ≠ 原始第 171 个特征方向方差为零**。Cholesky 的 leading minor 是"前 k 行前 k 列子矩阵的行列式"，与某个原始坐标的方差没有直接对应；pivot 非正可能来自任意线性组合（z 各方向高度混合）或纯浮点误差，不能据此断言"最后方向塌缩"。
- **"各层高度相关"推断不成立**：各 τ 的协方差是分别计算的，层间相关不会使单个 C_ZZ 奇异。
- **"Linear head + GELU 饱和"描述错误**：该 head 是线性的（无 GELU）；且压缩器输出不是导致 Cholesky 失败的充分机制。
- **Weyl 兜底表述错误**：Weyl 定理说对称矩阵加 `a·I` 后最小特征值 ≥ a **仅在原最小特征值 ≥ −a 时成立**；A 可能已有负特征值，正 a 并不保证 Cholesky 成功（初版兜底"必然成功"的表述不成立）。

### 1.4 可信的部分

- 协方差矩阵（样本估计，M=344 < 充分样本量）在训练中**条件数恶化**是事实（cond_zz 诊断可见）；
- 原 `_cholesky_retry` 的 escalate 次数少、最后一次调用不在 try 里——即使按"提高成功率"的目的，实现也粗糙（会让崩溃直接杀死进程，数小时进度丢失）。

---

## 2. 崩溃二：J 越界（Cholesky 修复后暴露）

### 2.1 报错原文

```
MONITOR_ERROR code=kf_score_out_of_bounds tjo:layer0 J=245.091446 dim=172
RuntimeError: monitor invariant failed [kf_score_out_of_bounds]:
tjo:layer0 J=245.091446 dim=172
```

- 发生位置：resume 续跑的 epoch 20 窗口关闭时刻（epoch 19 完整跑完，J=62.6）
- 违反的不变量：`0 ≤ J_τ ≤ d_τ`（d_τ = 172）

### 2.2 初版诊断（数学错误，已推翻）

初版推导崩溃链条：z 塌缩到 scale a ≈ 1e-10 → ridge 跌到 absolute jitter 之下 → `(C_ZZ + ridge)⁻¹ ≈ (a + 1e-12)⁻¹·I ≈ 1e12·I` → C_ZP ~ √a → `J ≈ a/1e-12 ~ 100+`。

**审阅修正（逐条）**：
- **`(a+1e-12)⁻¹` 的数值估算错误**：a=1e-10 时 `(a+1e-12)⁻¹ ≈ 9.9e9`，不是 1e12；
- **C_ZP ~ √a 的尺度传递漏掉了**：`J ≈ a·(a+δ)⁻¹ = a/(a+δ) ≤ 1`。更一般地，对尺度因子 c，**`J(c) ∝ c²/(c²+δ)`，c→0 时 J→0，不爆炸**；
- 因此 **J=245.09 > 172 绝不可能仅由 z 尺度缩小产生**。它必然意味着：协方差三元组 (C_ZZ, C_PP, C_ZP) 相互不一致（例如来自不同的去重 mask / 分母 / 时间对齐）、联合半正定性被数值破坏（moment 大数相减的灾难性消减，float32）、或 solve 实现错误。

### 2.3 更严重的统计问题（审阅者发现）

**高维小样本 CCA 饱和**：M=344 个样本、d_z=172、d_p=256 时，去中心化后的样本矩阵秩 ≤ M−1=343，而 (z,p) 联合子空间维数为 d_z+d_p=428，**强制子空间相交 dim = d_z+d_p−(M−1) = 85**。独立噪声下 `J_noise ≈ 128.2`——epoch 19 的 J=62.6 很可能主要是这个几何偏差，**不能作为"学到了未来信息"的证据**。

---

## 3. 两次崩溃的归属（审阅结论）

| 现象 | 初版归因（错误） | 审阅修正归因 |
|---|---|---|
| Cholesky leading minor 171 非正 | 第 171 维坍缩 | 协方差统计实现不一致 / 条件数恶化，pivot 非正 ≠ 坐标坍缩 |
| J=245.09 > 172 | z 尺度漂移到 1e-10 放大 | 数学上不可能（J(c)→0）；是 moment 合并错误 / 灾难性消减 / 实现问题 |
| 全阶段 J 曲线 | 学到未来信息 | M=344 下强制相交 85 维，独立噪声 J≈128，观测值可能主要是几何偏差 |

**结论**：目前看到的是"白化统计实现不一致 + 高维小样本 CCA 饱和"，**不是已经证实的表示分解坍缩**。崩溃之前的重启任务可继续作为诊断运行，但**结果不纳入正式实验**。

---

## 4. 修复（按审阅 P0 意见，2026-08-27 已实现）

### 4.1 P0-1：窗口协方差构造重写（不再累计 raw moments）

旧实现跨 microbatch 累计 `Σz, Σzzᵀ` 等 raw moments，窗口关闭时 `E[zzᵀ] − E[z]E[z]ᵀ`——z 带大 DC 偏移时（如 1e4 量级常数 + 1e-3 噪声）float32 灾难性消减，信号全损。

新实现（`src/rpbe/loss.py`）：

```python
Z = torch.cat(z_rows).double()               # 图连接，float64
P = torch.cat(p_rows).detach().double()
Zc = Z - Z.mean(0, keepdim=True)             # 直接中心化
Pc = P - P.mean(0, keepdim=True)
den = Z.shape[0] - 1
Czz = Zc.T @ Zc / den
Cpp = Pc.T @ Pc / den
Czp = Zc.T @ Pc / den
Czz = 0.5 * (Czz + Czz.T)                    # 显式对称化
Cpp = 0.5 * (Cpp + Cpp.T)
```

- 三矩阵来自**同一批 rows、同一个去重 mask、同一个分母**（`_close` 内有 shape assert）；
- Z/P 行数不一致会直接 assert 失败。

### 4.2 P0-2：先归一化尺度再 Cholesky（`src/rpbe/loss.py::_score_from_covs`）

```python
sz = Czz.diagonal().mean()        # 不能 detach：保持完整梯度路径
sp = Cpp.diagonal().mean()
if sz.detach() <= 0 or sp.detach() <= 0:
    return None, {"failed": "nonpositive_scale", ...}
A = Czz / sz
B = Cpp / sp
C = Czp / torch.sqrt(sz * sp)
A = 0.5*(A + A.T) + eps * eye     # ridge 现在是尺度无关的
B = 0.5*(B + B.T) + eps * eye
Lz, info_z = torch.linalg.cholesky_ex(A)
Lp, info_p = torch.linalg.cholesky_ex(B)
if info_z.any() or info_p.any():  # 不再盲目兜底
    return None, {"failed": "cholesky", ...}
W = torch.linalg.solve_triangular(Lz, C, upper=False)
K = torch.linalg.solve_triangular(Lp, W.T, upper=False).T
J = K.square().sum()
```

- 归一化使 J 对 z/p 的**任意正尺度严格不变**（比旧的 relative-ridge 更强），`J(c)` 恒等于常数，不可能再出现尺度驱动的爆炸；
- `mean(diag)` 不 detach → 梯度路径是归一化目标的全梯度（径向导数精确为 0，测试守护）。

### 4.3 P0-3：取消盲目 Cholesky 成功保证

- 删除 `_cholesky_retry` 的 escalating jitter + floor 兜底（`commit 13697e4` 的实现）；
- 失败窗口：`KFMomentWindow._close` 返回**可微 0**（该窗口不参与 backward，无梯度贡献）+ 完整诊断（`failed` 码、scale_z/scale_p、info_z/info_p、矩阵谱诊断）+ monitor `warning`（`kf_window_failed`）；
- `KFMomentWindow(strict=True)` 时失败 raise（调试用）。

### 4.4 P0-4：窗口尺寸与计数

- 计数从"unique cuts"改为**unique cut trees**（`seen_trees`）：同一棵 trace 的多个 cut 共享历史，不是独立样本；
- 配置（`config.py` + 两个脚本 CLI）：`sketch-dim`（P 维）256 → **64**，`kf-min-abs` 64 → **1024**（即 M_unique_trees ≥ 1024）；
- 按审阅口径：d_p=64 时 M_unique_trees ≥ 1024；若保留 d_p=256 则 M ≥ 2048。

### 4.5 P0-5：新增诊断（每 τ 每窗口）

`_diagnostics` 输出（写入 monitor step 行）：`scale_z/scale_p`（全局尺度漂移）、`zz_r_eff/pp_r_eff`（谱熵有效秩）、`zz_min_eig/zz_max_eig/cond`、`joint_min_eig`（联合协方差最小特征值，负值 = moment 实现错误）、`symmetry_error`（对称化前不对称度）、`J_real/J_shuffled`、`J_real_minus_shuffled`（区分小样本 CCA 饱和：两值都高且接近）、`M_unique/M_unique_trees`。

---

## 5. 回归测试（`test/test_rpbe_loss.py`，本机与云端全绿 23/23）

新增 6 项（审阅 §七）：

1. `test_scale_invariance_across_orders_of_magnitude` — Z → 10^k·Z, k=−12..12：J 不变且 ≤ min(r,m)；
2. `test_large_dc_offset_no_catastrophic_cancellation` — Z = 1e4 + 1e-3·noise：直接中心化恢复与纯 noise 相同的 J（旧 raw-moment 路径必挂）；
3. `test_window_matches_direct_centered_score` — 窗口关闭路径与手动 stack-center（同 den=M−1）逐项一致；
4. `test_z_and_p_share_one_row_mask` — dedup 后 Z/P 行数一致、rows_per_cut>1 仍一行一对、`_close` 内置 assert；
5. `test_independent_noise_saturation_regime_344_172_256` — M=344/d_z=172/d_p=256 独立噪声下 J_real ≈ J_shuffled（并记录饱和值）；
6. `test_window_radial_derivative_vanishes` — 窗口级径向导数 auto/fd 均 ≈ 0（无 half-gradient 混入）。

---

## 6. 遗留风险

1. **CCA 饱和的正式口径**：在窗口达到 M ≥ 1024（tree 计数）之前，J 曲线不能作为方法有效性的证据；报告机制列必须同时给出 `J_real − J_shuffled`。
2. **窗口内存**：M=1024 的图连接窗口驻留 ~30+ batches 的计算图，需在冒烟时实测 GPU 内存（可配合减少并行任务数）。
3. **validate_kf 保持 raise**（未改）：越界 raise 是有意的安全网，修复数值根因后不应再触发；再次触发即新的病态。
4. 正式实验配置：d_p=64、M_unique_trees ≥ 1024（本轮已设为默认）。
