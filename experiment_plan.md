# RPBE 论文实验执行索引

> 依据《RPBE论文实验设计最终交接规格》。本文件汇总：做什么实验、记录哪些数据、参数为什么这样设。正文三件套 = Table 1（跨领域）+ Table 2（TGN 消融）+ Figure 1（信息保存机制）；其余全部附录。

## 0. 全局统一公平协议（所有实验通用）

| 项 | 设置 | 理由 |
|---|---|---|
| 时间划分 | 任何树/cut 构造前完成 train/val/test（quantile 0.70/0.85） | 规格 §七：防泄漏 |
| 邻居预算 | n_degree=5、三层采样树（n_layer=2 → layer0/1/2） | TGN 与 TGN+RPBE 必须完全同树（规格 §三） |
| batch / 根查询 | bs=200、trace_roots=32、evenly_spaced | 同树同 cut；RPBE 禁 positive_first（防 label 依赖） |
| Γ 架构 | 统一 width_D=128、GELU 核心、每接口独立头 | 规格 §四：所有消融完全相同 Γ |
| 输出维度 | r_τ = 宿主接口维（172） | 宿主决定，不调 |
| checkpoint 选择 | 全部按 validation AUC 选模（TGN 协议），同时报告 AP 与 AUC | 规格 §三公平协议 |
| 优化 | Adam lr=3e-4、grad-clip 5.0、10~30 epoch（视收敛）、同早停 | 相同训练协议 |
| seed | **5 个配对 seed**（0-4），报告 mean ± std 与关键配对差置信区间 | 规格 §七 |
| KF 统计 | 窗口按独立根树数关闭（min_ratio×min(d,m)、min_abs=1024）；whitening 只限 train 窗口；报告 real-vs-shuffled | 规格 §七 |
| λ_KF | 主表用各自调优后的值；附录给同固定 λ 的配对结果 | 排除调参差异 |

## 1. Table 2：TGN 受控消融（P0 变体实现后的第一个实验）

**记录数据**：Wikipedia 与 Reddit 各 5 seed 的 val AUC（选模）、test AP、test AUC；每行 mean ± std；Δ_closure = AP(2Obs) − AP(1Obs) 配对 CI。

| 行 | 变体 | 参数 | 回答的问题 |
|---|---|---|---|
| 1 | TGN + Γ, task only | λ_KF=0（kf_variant 任意，auxiliary 关闭） | 额外参数攻击：增益是否只来自加参数/微调 |
| 2 | TGN + Γ, profiled reconstruction | kf_variant="reconstruction" | 普通压缩攻击：任务无关重建是否足够 |
| 3 | TGN + RPBE-1Obs + full balancing | n_observations=1、kf_variant="full_balancing" | 与行 5 的差值 = 第二次 pullback 约束的实际影响 |
| 4 | TGN + RPBE-2Obs + diagonal whitening | n_observations=2、kf_variant="diagonal" | balancing 攻击：只处理对角是否够 |
| 5 | TGN + RPBE-2Obs + full balancing（完整方法） | n_observations=2、kf_variant="full_balancing" | 完整方法 |

**关键约束**（规格 §四）：
- 1Obs 与 2Obs：输入历史、Γ、balancing、cut 集合完全相同，只差是否用第二次观测；1Obs 每树监督总权重 1、2Obs 两观测各 1/2 每树仍 1；不因 2Obs 增加根样本数或优化步数
- 行 2 的 J_rec = tr(Σ_UZ Σ_ZZ⁻¹ Σ_ZU)（Z 重建 U 的 profiled 方差，线性下 PCA 等价）；U = 压缩前 vanilla aggregate（共享定义，CutCandidate.u）
- 行 4 的 J_diag = ‖D_Z^{-1/2} Σ_ZP D_P^{-1/2}‖²_F；完整方法命名 J_KF（不叫 J_diag）

## 2. Figure 1：信息保存机制（P1 审计工具的数据）

**Panel A（PIR vs 历史距离）**：d=1,2,3；曲线 = TGN / task-only / RPBE-1Obs / RPBE-2Obs full。

**记录数据**（每 d、每方法、每 seed）：
- R0^(d)、RU^(d)、RZ^(d)（校准后 test NLL，三个 probe q0/qU/qZ）
- I_available = R0−RU、I_retained = R0−RZ、PIR = I_retained/I_available
- 两者置信区间（cluster bootstrap，按根查询聚类）；I_available≈0 或 CI 覆盖 0 时不报 PIR；负值不裁 0

**Panel B（局部递归稳定性）**：F_r 换成 Y(1)/Y(2)，只比 1Obs vs 2Obs。
**判读**（规格 §五）：1Obs/2Obs 在 Y(1) 接近 + 1Obs 在 Y(2) 明显更差 + 2Obs 恢复 Y(2) + Table 2 的 2Obs 根任务 AP 更高 → 支持"递归不闭合"解释；否则只能解释为监督强弱。

**PIR 防攻击协议（14 条）**：冻结主模型不回传；先按根划分 probe 数据再生成 cut 样本；每树总权重相同；q0/qU/qZ 同类型同预算；线性 ridge logistic 主结果 + 两层 MLP 附录复核；独立 calibration 校准；Z=U 恒等 ~100% PIR；随机 Z ~0；打乱目标 ~0；不裁剪负值；cluster bootstrap；报告 I_available 防虚高；根标签绝不用于构造 Y(2)。

## 3. Table 1：跨领域（P3 宿主确定后）

- Panel A（Temporal Graph）：TGN / TGAT / TCL / DyGFormer / RTRGN / CTAN / TGN+RPBE × Wikipedia+Reddit × 5 seed AP+AUC。公平协议：同时间划分/标签/预处理/评估代码；TGN 与 TGN+RPBE 完全同树同协议；其他模型用官方原生配置；有效历史预算不匹配时附录公开
- Panel B/C（Point-cloud / LLM Memory）：宿主与任务确定后设计——必须事先写清树节点/递归使用关系、cut 处历史、Y(1)/Y(2) 的两个自然同语义任务观测、为何不是根标签/教师预测。**一个任务不能提供两个合法局部观测就不能作证据**

## 4. 附录实验

| 附录 | 内容 | 参数 |
|---|---|---|
| A 深度测试 | 深度 2/3/4，仅 TGN / 1Obs / 2Obs-full，报 AP、AUC、远程 PIR | n_layer=1/2/3 |
| B 超参数 | λ_KF 完整曲线；m、ridge ε、窗口有效样本阈值紧凑表 | λ ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.05} |
| C 次要消融 | 去上下文只留任务观测 / 均匀 cut 权重 vs 推导权重 / 离线 PCA / 随机投影 / raw cross-covariance | — |
| D 效率 | 参数量、每 epoch 时间、总收敛时间、峰值显存、推理延迟、各环节耗时分解 | — |
| E 合成不闭合 | 构造 Y(1) 相同 Y(2) 不同的两段历史：1Obs 可合并、再递归后失败、2Obs 可区分 | 合成数据 |

## 5. 论文写作口径（记录在案，不在本文件展开）

- 术语：one-observation predictive relation / two-observation pullback refinement（不用 "1-step/2-step closure"）
- 主张限定："两次局部观测在固定任务测试族与允许递归上下文下，对一次递归使用后的预测稳定性提供可计算约束；结合误差递推可控制更深层根任务误差"——不声称无条件任意深度精确闭合
- 缺失 Y(2) 必须 mask/舍弃 cut，绝不根标签补齐（代码已保证）

## 6. 数据与日志落点

| 数据 | 位置 |
|---|---|
| 每实验 config.json | 记录全部 CLI 参数（复现索引） |
| metrics.jsonl | 每 epoch 的 val/test 全指标 + kf 诊断（J、J_frac、real-vs-shuffled、参考刷新次数、p 缓存命中率） |
| summary.json | best_epoch、best_val、test 全指标 |
| monitor/alerts.jsonl | 警告与不变量失败 |
| PIR 审计输出 | 每个 (方法, d, seed) 的 R0/RU/RZ、I_available/I_retained/PIR + bootstrap CI |
