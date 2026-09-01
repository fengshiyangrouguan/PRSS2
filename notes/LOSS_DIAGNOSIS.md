# RPBE loss 独立复核与诊断门槛（2026-08-30）

## 结论先行

当前精确分数

\[
J(Z,P)=\operatorname{tr}
\left[(\Sigma_{ZZ}+\varepsilon I)^{-1}\Sigma_{ZP}
(\Sigma_{PP}+\varepsilon I)^{-1}\Sigma_{PZ}\right]
\]

作为“固定有限测量下，`Z` 与联合测量 `P=Psi(C,Y)` 的正则化 CCA / predictive energy”是成立的；`full_balancing` 的协方差归一化、Cholesky 实现和完整 `Z` 侧梯度没有发现原则性错误。

但它目前不能被表述成已经推出或实现了“全 context 条件预测商的递归闭合”。问题不在 balancing 本身，而在目标对象、有限测量和生产梯度之间尚未对齐。

## 已确认的问题

### P0：实现对象与论文口径不一致

1. `records.py` 的 Y1/Y2 是 cut 节点在 `cut_time` 后的前两个真实交互，不是同一计算树中的 parent / grandparent consumption。
2. compact trace 只保留 query-node 的 SELF spine；不同 tau 的 cut 对同一 root 使用相同 node、time 和 future outcome。trace 中没有可供 CutBuilder 读取的 parent-consumption contract。
3. 因此 1Obs/2Obs 只能解释成“一个/两个时间未来事件监督”。`AP(2Obs)-AP(1Obs)` 不能命名为 closure 或 pullback 效应。
4. 真实未来 counterpart 来自后来实际发生的事件。当前目标可解释为 observed/on-policy joint continuation；除非 continuation policy 与历史独立或做了相应重加权，不能把它直接升级为任意给定 context 下的条件响应。

### P0：宏组 loss 单位错误（已修）

各 batch raw VJP 在同一参考均值下具有严格可加性；它们的和才是当前宏组的整窗线性化梯度。旧代码把 raw VJP 相加后，又在 `_close_repr_group` 中把任务和 KF 的总梯度一起除以宏组 batch 数 K。于是：

\[
g_{task}\sim O(1),\qquad
g_{KF}\sim O(1/K).
\]

名义 lambda 因而绑定宏组长度；`K=56, lambda=5` 不是可移植的 loss 系数。修复为每 batch 的零值 auxiliary 先乘当前宏组的实际 K，再和任务梯度一起做 K-batch 平均：任务仍取平均，KF 的 K 恰好抵消，留下 raw VJP 的严格和。这里不能使用 `W_ref/W_batch`，因为当各 batch 的有效 cut 权重不等时，它会偷偷改成 equal-batch reweighting。新增回归测试把宏组平均纳入了等价性检查。

独立 NumPy 有限差分反例（3 个 batch，总权重分别为 1.0/4.0/2.3）得到：raw 求和与 group-K 修复对整窗梯度的相对误差均为 0；旧的 raw 后统一除 K 相对误差为 0.667；`W_ref/W_batch` 后再平均相对误差为 0.558、cosine 为 0.871。这个修法由整窗目标的可加性决定，不是经验调尺度。

修复前的 lambda 扫描和 0.9067 数字不能与修复后结果混用。

### P0：生产梯度不是当前 J 的精确梯度

生产路径使用上一宏组的均值和 adjoint作用于当前宏组：

\[
\widehat g_t
=\left\langle \nabla_S J(S_{t-1}),
\nabla_\theta S_t\right\rangle.
\]

这是一阶滞后近似。`reference_age=1` 只证明版本年龄，不证明方向正确；时间流的分布漂移也不由该计数捕获。原测试只证明“同一点 adjoint replay 等于 direct gradient”，并明确允许 stale gradient 指向真实 batch J 的其他方向，所以还不足以验收论文中的 loss。

### P1：联合测量可能主要奖励 context

二元 future 的两个固定随机签名可写为

\[
\varphi_Y(y)=a+b_y,\qquad
a=\tfrac12(\varphi_Y(0)+\varphi_Y(1)).
\]

`[1;phi_C(C)] tensor_product a` 完全不依赖 outcome，却仍进入 P。Wikipedia 标签极稀疏，而 future counterpart / delta_t 对历史通常可预测，因此总 J 可能主要来自“预测以后会和谁、何时交互”，而不是状态变化标签。full whitening 不会自动消除这种统计混杂。

新增测量分解：

- `joint`：生产 P；
- `context_common`：只保留 outcome-independent 的 a；
- `outcome_contrast`：只保留 `b_y`；
- outcome shuffle 与整行 P shuffle 零假设。

### P1：有效样本量与删失

相邻 root 的 Y1/Y2 高度重叠；同一 future event 还会被多个 tau 重复使用。窗口 gate 目前按 tree id 计数，但 tree id 是 root query，不等于独立节点或独立 outcome。训练尾部和低活跃节点更容易缺 Y2，造成 informative censoring。诊断必须同时报告 node-cluster ESS、outcome-cluster ESS、outcome multiplicity 和时间四分位 future coverage。

### P1：现有实验数字不是确认性证据

旧脚本会为每个 lambda 直接产生 test 指标，提交信息又用 seed-0 test AUC 宣传 lambda=5。该 test 已用于方法迭代，0.9067 只能作为探索记录；正式超参数必须只由 validation 决定，锁定后再做未被查看的确认性评估。并且主比较必须是 RPBE 与 matched `TGN+Gamma, lambda=0`，不是只和冻结宿主 vanilla 比。

## 树传播算子到底解决什么

先纠正一个容易混淆的说法：从“2Obs 的有限监督”推出“结构递归闭合”并不需要 Markov 或收缩假设。只要父节点只能读取 child z 和同一个合法局部输入，确定性算子的同余可直接逐层归纳；收缩只在要控制**近似误差随深度传播**时才需要，而不是 exact equality 的前提。

宿主递归算子没有被破坏，而且 Gamma 在各 occurrence 共享。若两个孩子已经给出完全相同的 z，在相同局部输入与 sibling context 下，确定性父算子会给出相同父状态。这保证的是架构同余（syntactic congruence）。

但架构同余本身很弱：常数 z 也严格满足它，却不保留任何预测信息。loss 要承担的不是再次证明树可以递归，而是让这个已经闭合的 z 接近**预测**等价类。要从有限联合 CCA energy 走到条件预测商，至少还需要说明：

- continuation 的采样测度是否固定/外生，或如何对 observational policy 重加权；
- 固定测试族对目标条件分布是否足够分离；
- 状态预算是否覆盖全部相关 canonical directions；
- 有限样本、ridge、近似优化和滞后梯度造成的误差；
- 若主张树 pullback，监督记录是否真的来自相应 parent context。

因此“原树算子没问题”能保住递归架构，但不能替当前 Y1/Y2 自动补出 pullback 语义。

## 诊断顺序与停止门槛

先运行：

```bash
python -m pytest test/test_rpbe_loss.py test/test_eighth_review.py -q

python -m scripts.diagnose_loss \
  --run-dir outputs/s0_lambda_scan/lam0 \
  --data-dir old/processed_tgn_data \
  --audit-split val --group-batches 40 --groups 3 --permutations 20
```

再对一个 RPBE checkpoint 运行同一命令。`diagnose_loss.py` 不训练，也不读取 test。

门槛：

1. **实现门槛**：`same_point_control.cosine >= 0.99`、norm ratio 在 `[0.99,1.01]` 且 relative error 不超过 0.01。只检查 cosine 会漏掉旧代码严格同向但缩小 `1/K` 的单位错误；任一失败都停止训练。
2. **滞后门槛**：连续宏组 latent-gradient cosine 建议至少 0.90，norm ratio 在 `[0.5, 2]`。失败则改成同窗 replay / 两遍精确梯度，或缩短 lag；不能靠调 lambda 修方向。
3. **outcome 门槛**：两个 tau 的 `outcome_contrast` 均应超过 label-shuffle 的 95% 分位。失败则说明当前数据/测量没有可辨认的 outcome 信号，应先改 P（对比编码、条件残差或重新定义 query/outcome），不能做多 seed。
4. **context 门槛**：必须同时报告 `context_common`。若 joint excess 几乎全来自 context-common，只能把方法称为 joint future-event representation regularizer，不能声称保留了标签条件响应。
5. **支持门槛**：outcome/node cluster ESS、正例数和 score 对窗口长度翻倍必须稳定；只看 1024 个 tree id 不够。
6. **因果步门槛**：通过以上门槛后，再做一个冻结 checkpoint 的小步实验：精确当前窗 KF-only step 应提高独立下一窗的 exact J；shuffled-outcome step 不应提高任务指标。

只有这些门槛通过后，才开始 validation-only lambda 校准和 matched 三组小规模实验：vanilla、`Gamma+task only`、RPBE。五 seed 与 test 留到最后。

## 两条可选路线

- 若论文核心是树预测商：恢复 audit-only/full tree trace 和 parent-consumption target，先做 current-future 与 true-pullback 的同 checkpoint 对照，再决定正式训练 target。
- 若接受时间未来正则器：保留当前 FutureIndex，但全文删除 pullback/closure 推断，明确目标是 observational joint future-event energy，并通过 contrast/context/null 诊断限定主张。

balancing 可以保留；现在没有证据要求把 full balancing 换成 diagonal。先修目标语义与梯度可信度，再做 whitening 消融。
