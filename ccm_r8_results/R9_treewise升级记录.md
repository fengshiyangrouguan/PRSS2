# CCM RPBE → TGN 线最新 treewise 对齐（R9）

2026-09-15。把 CCM 的 RPBE 优化耦合升级为 TGN 线（feature_ablation/feature_per_tree
lineage，最新 commit b523cf3）的 final-spec treewise 实现。

## 1. 源码来源与语义

b523cf3 本身只是 κ=0 Cimmino 精修（迭代 500→2000 + 里程碑遥测，9 行改动）。
真正的 final spec 在 `src/rpbe/training/tgb_link_loop.py::_cstr_group_close_treewise`：

- **QP**：min_d ½‖d−t‖² s.t. g_jᵀd ≥ −κ‖g_j‖·‖t‖，t = scope 上的聚合 task 梯度；
  **行归一化** H = G/‖g‖（约束等价、对偶条件数大幅改善），对偶
  max_{μ≥0} −½μᵀKμ + μᵀc，K=HHᵀ，c=b−Ht，b_j=−κ·‖t‖。
- **求解**：FISTA 投影梯度（幂迭代估 λ_max 定步长），ladder 分档预算；
  TGN κ>0 路径 3 轮 active-set 扩展（每轮 +1 最违反行）。
- **全量证书**：所有行 viol ≤ 1e-6 才写回 d；失败 = **CERT_FAIL → 跳过该窗口的
  repr step**（审查要求：不执行无证书的约束更新）。
- **probe 模式**：只记录 cos 分布不写回（用于选 κ）。
- **κ=0 特化**：dense 全加 + 12 轮 + Cimmino primal 精修（b523cf3）——**CCM 不用**
  （CCM 走 κ=0.05 软约束档）。
- **2obs_aligned / per_tree Est.**：TGN 结构消融臂（use_parent 父观测配对 /
  统计估计粒度 per-tree），CCM 的 2Obs 构造（两个连续未来 turn 的 CE 目标，
  chi1/phi1 + chi2/phi2）是 CCM 自己的 production 协议，不动。

## 2. CCM 侧改动（scripts/train_ccm.py）

1. `treewise_feasibility_projection` 重写：
   - 行归一化 H + K=HHᵀ（旧版用未归一 G、Q=GGᵀ）
   - 解后**全量证书**（旧版只记 min_slack 不 gate）
   - 返回 `(ok, diag)`；证书失败**不写任何 Γ 梯度**
   - 诊断对齐 TGN：cos_mean/med/p5/min、frac_below、frac_below_grid
     （κ∈{0,0.02,0.05,0.1,0.15,0.2,0.3} 违反比例曲线）、feasible_d0、max_viol、
     corr_ratio、d_norm_ratio、fista_iter；保留旧 proj_* 字段作跨线对照
   - CCM 规模说明：窗口 N=128 方向（TGN 2000+），**active set = 全集**，
     无 active-set 扩展循环（TGN 的 3 轮扩展是大 N 求解器容量装置）；
     FISTA 步长用 eigvalsh 精确 1/λ_max（TGN 幂迭代的同值估计，N 小更优）
2. **CERT_FAIL → skip grad_step**（TGN "skipping repr step" 语义）：窗口照常
   关闭、数据流/step/boundary/checkpoint 节奏不变；boundary 记录加 `:SKIP`
   后缀（哈希如实反映该窗口未更新）；Γ 与非 Γ 梯度全部清零。
   与 TGN 的对应差异：CCM 的 task 与 repr 更新耦合在每窗口一次 grad_step 里
   （TGN 是每批 task step + 每 group repr step 分离），CERT_FAIL 跳过整个窗口
   更新是 CCM 结构下的正确对应，文档化于此。
3. 新增 `cert_skip_steps` 计数器（log row / checkpoint / summary / _SUCCESS
   全链路，resume 恢复）。
4. 投影在 fp16 GradScaler 的 scaled 梯度上运行——QP 对 t 与 G 的同一正缩放
   不变，数学不受影响（代码注释说明）。

sanity（服务器 torch CPU/GPU，随机构造）：
- 可行起点 → feasible_d0 直接 task-only 写回 ✓
- 违反方向 → 投影后 viol 3.6e-8（≤1e-6 证书通过）✓
- 强违反（cos≈−0.95 × 6 行）→ viol 1.2e-7 ✓
- 极端矛盾（60% 方向反平行）/κ=0 硬约束 → **CERT_FAIL 正确拒绝、梯度不写** ✓
- 退化路径（无方向/零 task）✓

## 3. 超参映射（TGN 表 → CCM）

| TGN 参数 | TGN 值 | CCM 采用 | 说明 |
|---|---|---|---|
| lambda_kf | 0.00668 | **0.02229659292991447** | CCM frozen R8 r_eff=0.1 校准（θ₀ 上测）；投影与 λ 无关（约束归一化），无需重扫 |
| rpbe_kappa | 0 / 0.05 | 0.05 | A7 软约束档；κ=0 的 Cimmino 路径不移植 |
| rpbe_constrain_scope | gamma | gamma ✓ | 投影只写 Γ |
| kf_min_abs | 896 | kf_min_cuts=128 | CCM frozen 窗口门禁（自己的规模） |
| kf_group_batches | 40 | — | CCM 窗口=组（每窗口一 close 一 step） |
| cuts_per_tau | 1024 | — | TGN 每接口配额，CCM 无 |
| n_observations | 2 | 2 ✓ | CCM 已有 |
| supervision_mode | 2obs_aligned | CCM production | CCM 自己的 2Obs 构造，不搬消融臂 |
| kf_variant | full_balancing | full_balancing ✓ | 已有 |
| state_dims | {τ:172} | z_dim=128 | CCM JMemLift 状态维 |
| width_D / m / d_c / d_f | 128/64/32/32 | 128/32/… | sketch_dim_m=32 为 frozen R8 口径（改 m 需重扫 λ，不动） |
| ridge_eps | 1e-3 | 1e-3 ✓ | frozen |
| delta_t_scale | 1e6 | — | TGN 连续时间 RFF，CCM 无 |
| rpbe_seed | 0 | 0 ✓ | frozen |
| trace_roots / trace_pairs_per_parent | 32 / 2 | — | TGN 树采样；CCM 每窗口全部 cut 进 |
| num_counter_bins | 4096 | — | TGN maps 的 counter sketch；CCM JMemLift 自研 sketch（z_dim=128 bins） |

## 4. R9 训练协议

- 命令 = R8 标准协议 + `--rpbe-constrain-mode treewise --rpbe-kappa 0.05
  --proj-iters 400 --max-windows 50 --checkpoint-every 10`
- 3 seeds × ours，2 并行（80GB 显存限制）；lr 3e-5、batch 2×64、
  schedule 1000（R8 不变）
- 每 checkpoint（s10/20/30/40/50）跑 EOS 排除版时间步评估
  （eval_ccm_depth.py，TIME_STEPS=[1,2,4,8,13]，turns=L+2）

### 4.1 R9 最终决策（2026-09-15 晚，用户拍板）：aggregate 等价运行

实测证据：smoke 每窗口 128 树方向、cos_min ≈ −0.02（第二窗口 −0.010），
κ=0.05 下 **viol 恒为 0** → treewise 投影在 CCM 上从不介入。且 treewise 的
pass2 per-oid backward（128 次 7B 全链 vjp/窗口）使单窗口 4.5 分钟（双进程
竞争下 16 分钟），aggregate 仅 2 分钟（R8 实测）。

决策：**R9 用 aggregate 模式**（与 R8 训练同构，`--rpbe-constrain-mode
aggregate`），保留 max-windows=50、checkpoint-every=10 细粒度评估协议。
treewise 移植代码保留在仓库（数学 sanity 已全过）；零冲突证据（cos 分布）
是论文侧"约束在 CCM 上不绑定"的机制证据。

## 5. 评估协议（同会话更新）

EOS 排除修复后最终数字与同实现对比口径见
[DailyDialog评估协议排查记录.md](DailyDialog评估协议排查记录.md) 第 4-6 节。
