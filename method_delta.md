可以实现，而且不需要教师模型。这个 txt 的问题不是主体不可实现，而是把两代方案粘在了一起：

* 前半段残留了“冻结宿主输出、最终根部行为”；
* 后半段换成了更完整的 \(U/B\) profiled-balancing loss；
* 伪代码最后却仍然只更新 \(\Gamma_\theta\)。

正确做法是：保留前半段的递归压缩架构，采用后半段的增量 balancing loss，但彻底替换监督来源和训练参数。

---

# 一、我们真正要做什么

对于树中的每个切口 \(v\)，有三部分：

$$
(H_v,C_v,Y_v).
$$

* \(H_v\)：切口之前已经形成的历史；
* \(C_v\)：该切口之后将怎样继续被使用；
* \(Y_v\)：数据中与这个切口对应的真实未来 continuation。

我们学习：

$$
z_v
=
\Gamma_{\omega,\theta}
\left(
o_v,\{z_{v_i},\xi_{vi}\}
\right),
$$

其中：

* \(\omega\)：宿主的特征提取、消息编码、聚合参数；
* \(\theta\)：新增递归压缩器参数；
* 二者在同一个 loss 下联合训练。

目标是：

> 宿主允许传播多少维，就输出多少维；在这个维数预算中，尽量保留该切口真实未来仍能区分的信息。

---

# 二、未来监督到底从哪里来

未来不来自模型，而来自每个切口自己的数据 continuation：

$$
Y_v^{(1)},\qquad
Y_v^{(2)},\qquad
Y_v^{(\star)}.
$$

应理解为：

* \(Y_v^{(1)}\)：切口 \(v\) 之后第一个合法未来记录；
* \(Y_v^{(2)}\)：第二个合法未来记录；
* \(Y_v^{(\star)}\)：相对于该切口预先规定的较远期记录或窗口终点。

其中 \(\star\) 不再叫“根行为”。它是 cut-relative long horizon。

具体在 TGN 中读取什么字段，由数据适配器确定，例如：

* 后续交互事件；
* 后续节点/边观察；
* 后续状态变化；
* 与该切口相联系的真实行为结果。

但绝对不能是：

* 冻结宿主 hidden；
* 父 occurrence 的 teacher embedding；
* 根标签广播；
* 另一模型生成的未来；
* 当前模型 rollout 出来的预测。

固定未来测试为：

$$
f_v
=
\bigoplus_{h\in\{1,2,\star\}}
\sqrt{\omega_h}
\left[
m_v^{(h)}\varphi_h(Y_v^{(h)});
m_v^{(h)}
\right],
$$

其中可取：

$$
\omega_1=\frac14,\qquad
\omega_2=\frac14,\qquad
\omega_\star=\frac12.
$$

然后：

$$
\boxed{
p_v
=
\operatorname{TensorSketch}
\left(
[1;\chi(C_v)]\otimes f_v
\right).
}
$$

这里：

$$
\nabla p_v=0,
$$

因为 \(\chi,\varphi_h,\operatorname{TensorSketch}\) 都是固定随机映射，且 \(C_v,Y_v\) 来自数据。

固定的是测量坐标，不是宿主。

---

# 三、宿主和压缩器如何联合训练

每个节点先由宿主提取局部特征：

$$
a_v=A_\omega(o_v).
$$

孩子状态适配为：

$$
t_{vi}
=
E_{\omega,\theta}
(z_{v_i},\xi_{vi},\ell_{vi}).
$$

使用宿主原本适合的聚合器：

$$
c_v
=
\operatorname{Agg}_{\omega,\theta}
\{t_{vi}\}.
$$

然后得到宿主接口规定维数的状态：

$$
\boxed{
z_v
=
Q_{\tau_v,\omega,\theta}
\left(
G_{\omega,\theta}(a_v,c_v)
\right)
\in\mathbb R^{r_{\tau_v}}.
}
$$

\(r_{\tau_v}\) 由宿主接口决定，不统一写死为 32。

为了计算统一 loss：

$$
b_v
=
J_{\tau_v}z_v
\in\mathbb R^{r_B}.
$$

\(J_\tau\) 固定，只在训练 loss 中使用。

最终优化器必须包含：

```python
optimizer = AdamW(
    chain(
        host_feature_encoder.parameters(),
        host_message_encoder.parameters(),
        host_aggregator.parameters(),
        recursive_compressor.parameters(),
        interface_adapters.parameters(),
    )
)
```

不能只写：

```python
optimizer = AdamW(compressor.parameters())
```

---

# 四、为什么采用 txt 后半段的增量 loss

早期 loss 是：

$$
D_P-\mathcal R_{\rm PB}(B).
$$

它衡量 \(z_v\) 总共还缺多少未来信息，但会把两类损失混在一起：

1. 孩子状态以前已经丢掉的信息；
2. 当前节点再次压缩时新丢掉的信息。

如果每层都使用这个总缺失，再沿树相加，就可能重复计数。

所以后半段引入“当前节点压缩之前真正可用的信息”。

当前节点的实际输入为：

$$
I_v
=
\left(
o_v,\{z_{v_i},\xi_{vi},\ell_{vi}\}
\right).
$$

用固定映射构造：

$$
s_v
=
\rho_{\rm fix}
\left(
o_v,
\{\operatorname{sg}(z_{v_i}),\xi_{vi},\ell_{vi}\}
\right).
$$

这里 \(\rho_{\rm fix}\) 必须包含固定编码的原始局部输入，不能只读取可训练宿主 hidden，否则宿主可能把参考侧也一起压塌。

再定义：

$$
\boxed{
u_v=
\begin{bmatrix}
\operatorname{sg}(b_v)\\
\operatorname{sg}(s_v)
\end{bmatrix}.
}
$$

\(u_v\) 不是教师表示。它只是同一次前向中，当前节点在执行本次瓶颈前实际可用信息的固定统计见证。

---

# 五、真正的 predictive-balancing loss

对任意特征 \(X\in\{U,B\}\)，计算：

$$
C_{XX},\qquad C_{PX},\qquad C_{PP}.
$$

定义：

$$
G_X=C_{XX}+\varepsilon_xI,
\qquad
G_P=C_{PP}+\varepsilon_pI.
$$

解析 observability map：

$$
\mathcal O_X
=
G_P^{-1/2}C_{PX}G_X^{-1}.
$$

从 \(X\) 可读出的未来测试为：

$$
\widehat\mu_X(v)
=
\mathcal O_X(x_v-\mu_X).
$$

对应的 balancing 能量：

$$
\boxed{
\mathcal R_{\rm PB}(X)
=
\operatorname{tr}
\left(
G_P^{-1}
C_{PX}
G_X^{-1}
C_{XP}
\right).
}
$$

它等于有限测试层 predictive Hankel operator 的 Hilbert–Schmidt 能量：

$$
\mathcal R_{\rm PB}(X)
=
\|\mathcal H_X\|_{\rm HS}^2.
$$

因此当前节点新丢失的未来信息为：

$$
\boxed{
\delta_v
=
\left\|
\widehat\mu_U(v)
-
\widehat\mu_B(v)
\right\|_2.
}
$$

总体上：

$$
\mathbb E[\delta_v^2]
=
\mathcal R_{\rm PB}(U)
-
\mathcal R_{\rm PB}(B)
$$

成立于嵌套投影的总体无正则情形；有限样本 ridge 版本附带估计误差。

这正是后半段 loss 比前半段 loss 更完整的地方：

$$
\boxed{
\delta_v
=
\text{当前节点这一次压缩新丢掉的信息},
}
$$

而不是“整棵子树到这里累计丢掉的所有信息”。

---

# 六、树结构怎样进入最终 loss

对每个连接定义误差放大系数：

$$
L_{v,i}.
$$

然后执行动态规划：

$$
E_v
=
\delta_v+
\sum_{i\in\operatorname{ch}(v)}
L_{v,i}E_{v_i}.
$$

叶节点：

$$
E_v=\delta_v.
$$

展开为：

$$
E_T
=
\sum_{v\in T}
g_v\delta_v,
\qquad
g_v
=
\prod_{e\in\operatorname{path}(v\to\text{tree output})}L_e.
$$

最终：

$$
\boxed{
\mathcal L_{\rm RPBE}
=
\frac{
\mathbb E_T[E_T^2]
}{
D_P+\eta
}.
}
$$

这里的 `tree output` 或代码中的 `root_bound[root]` 只是把整棵树的局部误差汇总成一个标量。

它不是：

* 根标签；
* 根节点提供未来监督；
* 从根向所有切口传播目标。

为了防止其他 AI 继续误解，代码变量最好改名：

```python
root_bound[root]
```

改成：

```python
tree_energy[tree_id]
```

并直接写成：

```python
tree_energy = sum(path_gain[v] * delta[v] for v in cuts(tree))
```

这样不会再让人误以为监督来自根节点。

---

# 七、完整训练流程

每个 batch 做以下事情。

## 1. 数据侧构造固定测试

```python
with torch.no_grad():
    context_feature = fixed_context_map(cut_records.context)

    future_feature = fixed_future_map(
        future_1=cut_records.observed_future_1,
        future_2=cut_records.observed_future_2,
        future_long=cut_records.observed_future_long,
        masks=cut_records.future_masks,
    )

    P = fixed_tensor_sketch(context_feature, future_feature)
```

这一步不调用任何宿主模型。

## 2. 联合递归前向

```python
z_by_node = {}

for v in bottom_up_order(trees):
    local_feature = host_feature_encoder(v.raw_local_input)

    child_messages = [
        host_child_adapter(
            z_by_node[child],
            edge.features,
            edge.role,
        )
        for child, edge in children(v)
    ]

    z_by_node[v] = recursive_compressor(
        local_feature=local_feature,
        child_messages=child_messages,
        interface=v.interface,
    )
```

这里所有宿主特征参数和 \(\Gamma_\theta\) 都可训练。

## 3. 构造压缩前后两套统计坐标

```python
B = stack([
    fixed_loss_lift[cut.interface](z_by_node[cut.node])
    for cut in cuts(trees)
])

S = fixed_prebottleneck_witness(
    raw_local_inputs=cuts(trees).raw_local_inputs,
    child_states=stopgrad(cut_child_states),
    edge_features=cuts(trees).edge_features,
)

U = concat([
    stopgrad(B),
    stopgrad(S),
], dim=-1)
```

这里没有 teacher，只是同一次前向的局部输入与输出。

## 4. 交叉拟合解析 profile

按整棵树把 batch 分成 A/B 两折：

```python
O_B, O_U = fit_profiles(
    B=fit_fold.B,
    U=fit_fold.U,
    P=fit_fold.P,
    weights=fit_fold.weights,
)
```

在另一折上：

```python
mu_post = apply_profile(O_B, eval_fold.B)
mu_pre = apply_profile(O_U, eval_fold.U).detach()

delta = l2_norm(mu_pre - mu_post, dim=-1)
```

交换两折再算一次，避免同一批样本既拟合 probe 又评估 probe。

## 5. 汇总树能量并反向传播

```python
for tree in eval_trees:
    tree_energy[tree.id] = sum(
        detached_path_gain[cut] * delta[cut]
        for cut in cuts(tree)
    )

loss = mean(tree_energy.square()) / (future_energy + eps)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

梯度更新：

$$
\boxed{
\omega_{\rm host}
\quad\text{和}\quad
\theta_\Gamma.
}
$$

---

# 八、txt 中逐项怎么改

| txt 当前内容                          | 最终处理                               |
| --------------------------------- | ---------------------------------- |
| 宿主决定 \(r_\tau\)                   | 保留                                 |
| \(J_\tau\) 统一 loss 坐标             | 保留                                 |
| 非线性共享 \(\Gamma_\theta\)           | 保留                                 |
| attention 不是理论要求                  | 保留                                 |
| 固定 \(\chi,\varphi,\) TensorSketch | 保留                                 |
| “冻结宿主 embedding”构造 context        | 删除                                 |
| “未来来自冻结未压缩宿主”                     | 删除                                 |
| \(Y^{(\star)}=\)最终根部行为            | 改为 cut-relative long future        |
| “用冻结宿主构造完整计算树”                    | 改为数据 sampler 构造树和 cut continuation |
| 离线缓存宿主 hidden                     | 删除                                 |
| 离线缓存 raw future 及其固定 sketch       | 可以                                 |
| 旧 \(D_P-\mathcal R_{\rm PB}(B)\)  | 只作为基线                              |
| 后半段 \(U/B\) 增量 balancing          | 作为主 loss                           |
| \(R_{\rm root}\)                  | 改名为 `tree_energy`，只做结构汇总           |
| 只更新 \(\Gamma_\theta\)             | 改为联合更新宿主表征参数和 \(\Gamma_\theta\)    |
| future decoder/teacher/EMA        | 全部禁止                               |

---

# 九、给其他 AI 的不可误读版本

可以把下面这段直接交给实现者：

> RPBE 不使用教师模型、冻结参考宿主、EMA target 或根标签传播。每个 cut 从数据中读取与自身对应的 observed future continuation，并通过固定的 context map、future map 和 TensorSketch 得到 \(p_v\)。这些固定映射只是训练测量工具。
>
> 宿主的特征提取、消息编码、聚合模块与非线性递归压缩器 \(\Gamma_\theta\) 在一次 bottom-up forward 和一次 backward 中联合训练。每个接口输出宿主规定的 \(r_\tau\) 维状态，随后通过固定 \(J_\tau\) 进入统一 loss 坐标。
>
> 主 loss 使用 txt 后半段的嵌套 profiled predictive balancing：\(U_v\) 是同一次前向中当前节点压缩前实际可用信息的固定统计见证，不是 teacher；\(B_v\) 是压缩后的实际部署状态。二者对固定未来测试的解析可读差异定义局部增量缺失 \(\delta_v\)。再用 detached path gain 把各 cut 的 \(\delta_v\) 汇总成整棵树的 `tree_energy`，最终最小化其平方期望。
>
> `tree_energy` 的汇总位置不是监督来源；任何 cut 都不读取根标签或根 hidden。部署时删除所有未来记录、随机测试、profile、covariance 和 path-gain，仅保留联合训练后的宿主表征模块与 \(\Gamma_\theta\)。

所以结论是：**完全可以实现。**这个 txt 不需要推翻，应该把前半段的教师残留清除，并让前半段的递归架构、后半段的增量 balancing loss、以及“宿主+\(\Gamma_\theta\) 联合训练”真正接成同一条数据流。
对，先把边界钉死：

$$
\boxed{\text{没有教师信息，没有冻结宿主输出，没有根节点标签传播。}}
$$

训练中只有两条信息流：

$$
H_v \xrightarrow{\omega,\Gamma_\theta} z_v
$$

和

$$
(C_{v,h},Y_{v,h})
\xrightarrow[\text{完全固定}]{\chi_h,\varphi_h,\mathrm{Sketch}}
p_v.
$$

其中未来信息只进入 loss 的测量端，绝不能输入 \(\Gamma_\theta\)。

---

## 1. 三个对象的严格区分

对类型为 \(\tau_v\) 的切口 \(v\)，定义：

$$
H_v=\text{切口形成之前、切口以下的全部合法历史},
$$

$$
C_{v,h}
=\text{从切口到第 }h\text{ 个测量位置的“带洞 continuation”},
$$

$$
Y_{v,h}
=\text{该 continuation 下真实观察到的结果}.
$$

最直观地说：

$$
\boxed{
H_v=\text{被压缩的历史},\qquad
C_{v,h}=\text{向历史提出的问题},\qquad
Y_{v,h}=\text{数据给出的答案}.
}
$$

这里的 \(h\) 是切口相对的 horizon，不是固定指向根节点：

$$
h\in\{1,2,\star\}.
$$

* \(1\)：第一个合法未来测量点；
* \(2\)：第二个合法未来测量点；
* \(\star\)：预先规定的较长时间窗口、较远事件或任务终点；
* \(\star\) 不等于“根节点”。

---

## 2. 上下文 \(C_{v,h}\) 的严格定义

在树语言中，把 \(H_v\) 所对应的子树拿掉，留下一个类型为 \(\tau_v\) 的洞。洞以上、直到指定 horizon \(h\) 的计算环境就是：

$$
C_{v,h}[\square_{\tau_v}].
$$

它描述的是：

> 如果向这个位置放入某个同类型历史状态，接下来会怎样使用它、对它提出什么查询。

因此理论上的预测响应是：

$$
\mu_\tau(H,C)
=
\mathbb E[\varphi(Y)\mid H,C].
$$

预测同余定义为：

$$
H\sim_\tau H'
\iff
\mu_\tau(H,c)=\mu_\tau(H',c),
\quad
\forall c\in\mathcal C_\tau.
$$

### \(C_{v,h}\) 应该包含什么

| 信息                            | 是否进入 context | 说明                                  |
| ----------------------------- | -----------: | ----------------------------------- |
| horizon 编号 \(h\)              |            是 | 一步、两步或长程                            |
| 距离切口的时间差 \(\Delta t\)         |            是 | 说明何时查询                              |
| continuation 中的 operator 类型序列 |            是 | 说明如何使用切口状态                          |
| child role、参数槽位、边关系           |            是 | 区分左右孩子、源/目标角色等                      |
| 查询类型或任务类型                     |            是 | 如 link query、node query、event query |
| 候选对象或目标槽位                     |            是 | 只要查询时已经指定                           |
| continuation 中合法的 sibling 输入  |            是 | 它们是洞以外被固定的环境                        |
| 外生变量、预定动作、已知日历信息              |            是 | 必须在结果发生前可确定                         |
| truncation/censoring 信息       |            是 | 用于区分“没观察到”                          |
| 切口下方完整历史 \(H_v\)              |            否 | 否则绕过压缩                              |
| 压缩状态 \(z_v\)                  |            否 | context 不由模型状态定义                    |
| 未来结果 \(Y_{v,h}\)              |            否 | 会造成标签泄漏                             |
| 事件是否真实发生                      |            否 | 如果这是要预测的结果，它属于 \(Y\)                |
| 未来才揭示的节点/边特征                  |            否 | 属于结果的一部分                            |
| 宿主 hidden、logits、embedding    |            否 | 没有教师模型                              |
| 根标签、根 hidden                  |            否 | 与本方法无关                              |
| 模型预测值                         |            否 | 不能用预测结果定义测量标准                       |

最严格的判定方法是：

> 在不知道这次未来结果的情况下，能否先确定这个字段？

能确定，才可以进入 \(C\)；必须看到结果后才知道，就应放进 \(Y\)。

---

## 3. 未来行为 \(Y_{v,h}\) 的严格定义

\(Y_{v,h}\) 必须是数据在 \(C_{v,h}\) 对应位置真实记录的结果：

$$
\boxed{
Y_{v,h}
=
\operatorname{ObservedOutcome}
\bigl(H_v,C_{v,h}\bigr).
}
$$

它不是：

* 教师输出；
* 冻结宿主输出；
* 当前宿主模型的 hidden；
* 未来 decoder 的预测；
* 父节点或祖父节点的模型 embedding；
* 根标签向下复制。

### 可以作为 \(Y_{v,h}\) 的内容

| 任务    | \(C_{v,h}\)      | \(Y_{v,h}\)    |
| ----- | ---------------- | -------------- |
| 链接预测  | 候选节点对、查询时间、关系类型  | 边是否存在 \(0/1\)  |
| 时序交互  | 查询实体、未来时间窗口、事件类型 | 实际事件、消息、边属性    |
| 节点分类  | 节点、查询时间、任务类型     | 该时刻真实类别        |
| 连续动力学 | 未来时间、干预或控制输入     | 实际观测状态或状态增量    |
| 树结构任务 | 后续合法操作、role、外部输入 | 对应位置真实可观测输出    |
| 序列任务  | 下一位置、时间差、查询类型    | 实际 token、事件或数值 |

如果事件结果本身是结构化的，可以定义：

$$
Y_{v,h}
=
\bigl(
a_{v,h},
x_{v,h},
m_{v,h}^{\mathrm{payload}}
\bigr),
$$

其中：

* \(a_{v,h}\)：事件是否发生；
* \(x_{v,h}\)：发生时观察到的 payload；
* \(m^{\mathrm{payload}}\)：payload 是否存在。

注意，事件是否发生属于 \(Y\)，不是 context validity mask。

---

## 4. validity mask 和负样本必须分开

这是实现中很容易写错的地方。

### 没有观察到

例如数据窗口已经结束，无法知道第二个未来事件：

$$
q_{v,h}=0.
$$

这叫 censored/missing，不能当作负例。

### 观察到结果为否

例如候选边明确不存在：

$$
q_{v,h}=1,\qquad Y_{v,h}=0.
$$

这是有效负样本。

因此：

$$
\boxed{
q_{v,h}=0\neq Y_{v,h}=0.
}
$$

固定 future block 可以写成：

$$
f_{v,h}
=
\left[
q_{v,h}\varphi_h(Y_{v,h});
q_{v,h}
\right].
$$

---

## 5. 多 horizon 不应该共享一个含糊 context

因为一步、两步和长程可能对应不同查询对象、不同时间和不同关系，所以应分别定义：

$$
(C_{v,1},Y_{v,1}),
\qquad
(C_{v,2},Y_{v,2}),
\qquad
(C_{v,\star},Y_{v,\star}).
$$

每个 horizon 独立形成联合测试：

$$
p_{v,h}
=
R_h
\left(
[1;\chi_h(C_{v,h})]
\otimes
[q_{v,h}\varphi_h(Y_{v,h});q_{v,h}]
\right).
$$

再组合：

$$
\boxed{
p_v
=
\bigoplus_{h\in\{1,2,\star\}}
\sqrt{\omega_h}\,p_{v,h}.
}
$$

可以暂定：

$$
\omega_1=\frac14,\qquad
\omega_2=\frac14,\qquad
\omega_\star=\frac12,
$$

但权重属于超参数，不是理论必须值。

这里使用张量积，是因为我们要测试的是：

$$
\text{“在什么 context 下，出现了什么 future”},
$$

而不是分别记录 context 和 future 的边际统计。

---

## 6. 在 TGN 中的具体定义

假设切口 \(v\) 表示节点 \(i\) 在时间 \(t_v\) 之前的历史状态。

对于未来查询 \(h\)：

$$
C_{v,h}
=
\left(
\Delta t_h,\,
j_h,\,
\text{relation}_h,\,
\text{query type}_h,\,
\text{source/target role},\,
\text{已知静态特征}
\right).
$$

如果是链接预测，则：

$$
Y_{v,h}
=
\mathbf 1\{
(i,j_h)\text{ 在指定时间/窗口发生交互}
\}.
$$

若发生交互，还可以加入真实 payload：

$$
Y_{v,h}
=
\left(
\text{edge-existence},
\text{message},
\text{edge attributes}
\right).
$$

特别注意：

* 候选节点 \(j_h\) 属于 context；
* 是否与 \(j_h\) 发生连接属于 future；
* 不允许把“这是正样本还是负样本”的标志放进 context；
* TGN 自己生成的 parent/ancestor hidden 不能作为 future；
* 如果某个内部树 occurrence 没有对应真实数据结果，就不给它制造 \(Y\)，而是换到真实事件 horizon 或令该 block 无效。

---

## 7. 固定特征映射怎样处理原始数据

只冻结测量坐标，不冻结宿主模型：

$$
\chi_h(C_{v,h})
\quad\text{和}\quad
\varphi_h(Y_{v,h})
$$

必须是固定映射。

推荐规则：

* 类别：one-hot 或固定 hashing；
* 连续值：用训练集统计量标准化，再用固定 RFF；
* 高维向量：固定 JL 投影；
* 集合或序列：固定分桶、计数统计和固定 sketch；
* 联合交互：精确张量积或 TensorSketch；
* 随机矩阵和种子训练前生成，训练时无梯度。

禁止使用一个可训练 encoder 把 \(C,Y\) 编成 \(p\)，因为它可能主动丢掉难预测的信息，让 loss 虚假下降。

---

## 8. \(\Gamma_\theta\) 到底能看到什么

压缩端只能看到：

$$
I_v
=
\left(
o_v,\,
\{z_{v_i},\xi_{vi},\ell_{vi}\}_{i\in\operatorname{ch}(v)},\,
\sigma_v
\right).
$$

然后：

$$
z_v
=
\Gamma_\theta\bigl(\omega(I_v)\bigr).
$$

它不能看到：

$$
C_{v,h},\quad
Y_{v,h},\quad
p_v.
$$

训练时只是 loss 把两边对齐：

$$
\underbrace{H_v\rightarrow z_v}_{\text{要学习的压缩}}
\qquad\longleftrightarrow\qquad
\underbrace{(C_{v,h},Y_{v,h})\rightarrow p_v}_{\text{固定测量标尺}}.
$$

梯度只更新：

$$
\omega,\Gamma_\theta
$$

以及宿主中本来允许联合训练的模块；不更新：

$$
\chi_h,\varphi_h,R_h.
$$

---

## 9. 数据记录可以直接这样定义

```text
CutRecord
    tree_id
    cut_id
    interface_type

    history_branch:
        local_input
        child_cut_ids
        child_roles
        edge_features
        operator_type

    horizons[h]:
        context:
            horizon_id
            time_delta
            query_type
            candidate_or_target
            relation_type
            role_or_path
            legal_exogenous_features

        future:
            observed_outcome
            observed_payload

        validity:
            is_observed
            censoring_reason

        sampling:
            sampling_probability
            context_bucket
```

明确禁止出现：

```text
teacher_hidden
teacher_logits
frozen_host_embedding
ema_target
root_label_for_all_cuts
parent_model_embedding
ancestor_model_embedding
predicted_future
```

---

最终一句话定死：

$$
\boxed{
C_{v,h}\text{ 描述“未来怎样询问这个切口”；
Y_{v,h}\text{ 是该询问在数据中的真实答案；
二者只形成固定 loss 测量，不进入压缩器。}
}
$$

这样定义后，算法学到的才是“历史在所有被观测 continuation 下的条件响应剖面”，而不是对教师 hidden 或根标签的模仿。
