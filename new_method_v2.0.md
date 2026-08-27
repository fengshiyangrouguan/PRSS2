对，这一版在方法论上确实和 ISK 很像，但不是照搬 ISK：

$$
\boxed{
\text{理论负责推出“什么信息必须保留”，
一个 loss 负责把它变成可训练目标，
普通非线性网络负责真正拟合。}
}
$$

我们不把 Koopman 放进算法，也不要求树递归线性。真正借鉴的是：

> 不必发明复杂求解器；从预测商与树同余推出一个结构化 loss，再用普通非线性网络优化。

下面把最终方案重新定死。

---

# 一、我们究竟解决什么问题

宿主模型在树上递归传播信息：

$$
H_v
\longrightarrow
x_v
\longrightarrow
\text{父节点继续使用}.
$$

随着子树越来越大，宿主必须把完整历史 \(H_v\) 压进一个有限状态。但普通聚合器只根据最终任务 loss 训练，并不知道：

> 哪些历史差异在后续 continuation 中仍会改变未来行为，哪些差异可以安全丢弃。

我们学习一个递归压缩器：

$$
\boxed{
z_v
=
\Gamma_\theta
\left(
o_v,\{z_{v_i},\xi_{vi}\}_{i=1}^{k_v}
\right),
}
$$

使 \(z_v\) 在宿主允许的维数预算下，尽量保留完整历史对未来行为的影响。

训练时使用未来行为作为监督；部署时完全不读取未来。

---

# 二、最终的维数规则

这次必须明确区分三类维数。

## 1. 宿主传播维数：可变

节点接口类型记为 \(\tau_v\)。宿主要求该接口传播：

$$
z_v\in\mathbb R^{r_{\tau_v}}.
$$

不同接口可以有不同维数：

$$
r_{\tau_1}\neq r_{\tau_2}\neq r_{\tau_3}.
$$

例如：

$$
r_{\text{memory}}=64,\qquad
r_{\text{message}}=32,\qquad
r_{\text{summary}}=128.
$$

因此：

$$
\boxed{
r_\tau\text{ 由宿主接口和目标压缩率决定，不由 RPBE 理论统一规定。}
}
$$

32 维只能是某个具体实验的预算，不能再写成所有层统一为 32。

## 2. 生成器内部工作维数：固定

统一生成器需要一个公共计算空间：

$$
h_v\in\mathbb R^D.
$$

\(D\) 只是网络内部 hidden width，例如 128 或 256，不是实际传播的信息预算。不同维数的孩子先被映射到这个公共工作空间。

## 3. B 方案的统一 loss 维数：固定

由于不同节点的 \(z_v\) 维数不同，不能直接堆成一个矩阵计算 covariance。

因此，在实际宿主瓶颈之后增加一个仅训练使用的统一映射：

$$
\boxed{
b_v=J_{\tau_v}(z_v)\in\mathbb R^{r_B}.
}
$$

这里：

* \(z_v\) 是真正传播和部署的宿主状态；
* \(r_{\tau_v}\) 可以变化；
* \(b_v\) 是统一 loss 坐标；
* \(r_B\) 在一次训练中固定；
* \(J_\tau\) 位于实际瓶颈之后，不能放在之前。

所以最终维数关系是：

$$
\boxed{
\underbrace{r_\tau}_{\text{宿主决定，可变}}
\quad\longrightarrow\quad
\underbrace{r_B}_{\text{统一 loss 坐标，固定}}.
}
$$

这正是你说的：固定维数应该放在 B 方案的统一映射和 loss 中，而不是强迫宿主每层都输出同一个 \(r\)。

---

# 三、为什么 loss 必须放在宿主瓶颈之后

错误做法是：

$$
h_v\in\mathbb R^{r_B}
\overset{\text{计算 loss}}{\longrightarrow}
\mathcal L,
\qquad
z_v=Q_\tau(h_v)\in\mathbb R^{r_\tau}.
$$

这会产生漏洞：\(h_v\) 可以保留很多未来信息，但经过 \(Q_\tau\) 后，真正部署的 \(z_v\) 已经把信息丢掉了。

正确顺序必须是：

$$
\boxed{
H_v
\xrightarrow{\Gamma_\theta}
z_v\in\mathbb R^{r_\tau}
\xrightarrow{J_\tau}
b_v\in\mathbb R^{r_B}
\xrightarrow{\mathcal L_{\mathrm{RPBE}}}
\text{监督}.
}
$$

由于 \(b_v\) 只能读取已经压缩后的 \(z_v\)，loss 不可能监督到部署状态中不存在的信息。

---

# 四、B 方案的统一非线性生成器

统一生成器不表示所有接口输出维数相同，而表示：

> 不同接口共享同一个主要非线性递归核心，只保留轻量的输入适配器和输出头。

具体为：

## 1. 当前节点输入适配

$$
u_v=A_{\tau_v}(o_v)\in\mathbb R^D.
$$

不同宿主输入可以具有不同原始维数：

$$
o_v\in\mathbb R^{d_{\tau_v}},
$$

由轻量适配器 \(A_\tau\) 映射到公共工作空间。

## 2. 孩子状态适配

孩子 \(v_i\) 的状态为：

$$
z_{v_i}\in\mathbb R^{r_{\tau_{v_i}}}.
$$

先转换为统一消息：

$$
t_{vi}
=
E_{\tau_{v_i}},\ell_{vi}}
(z_{v_i},\xi_{vi})
\in\mathbb R^D.
$$

其中：

* \(E_\tau\)：接口适配器；
* \(\ell_{vi}\)：必要时表示孩子角色；
* \(\xi_{vi}\)：边、时间或关系特征。

## 3. 聚合孩子

$$
c_v
=
\operatorname{Agg}_\theta
\left(
u_v,\{t_{vi}\}_{i=1}^{k_v}
\right)
\in\mathbb R^D.
$$

## 4. 共享非线性递归核心

$$
h_v
=
G_\theta
\left(
u_v,c_v,e_{\sigma_v}
\right)
\in\mathbb R^D.
$$

## 5. 输出宿主规定的维数

$$
\boxed{
z_v
=
Q_{\tau_v}(h_v)
\in\mathbb R^{r_{\tau_v}}.
}
$$

所以完整的 \(\Gamma_\theta\) 是：

$$
\boxed{
\Gamma_\theta
=
\{A_\tau,E_\tau,\operatorname{Agg}_\theta,G_\theta,Q_\tau\}.
}
$$

其中 \(G_\theta\) 和主体聚合器共享，\(A_\tau,E_\tau,Q_\tau\) 只是薄适配层。这仍然是一个统一生成器，而不是逐层训练多个独立压缩器。

---

# 五、\(\Gamma_\theta\) 不规定必须是 attention

这次不再把 attention 写成方法要求。

$$
\boxed{
\Gamma_\theta\text{ 是非线性递归生成器；
attention 只是其中一种实现。}
}
$$

聚合器根据宿主选择：

| 宿主结构              | 默认聚合器                |
| ----------------- | -------------------- |
| 固定二叉、有左右语义        | 拼接 + MLP             |
| 无序、可变孩子数          | DeepSets / sum + MLP |
| 高分支且需要条件化选择孩子     | 轻量 cross-attention   |
| 宿主已有成熟 aggregator | 直接沿用宿主 aggregator    |

因此最推荐的原则是：

> 沿用宿主原本合适的聚合结构，只加入预测压缩瓶颈和 RPBE loss。

这样最符合 ISK 式设计：创新集中在理论推出的学习目标，而不是包装一个复杂 attention 架构。

对于 TGN，同质邻居且只有一种递归运算时，可以直接删掉：

* 接口类型 embedding；
* child role embedding；
* operator embedding。

整个实现可以退化为：

$$
z_v
=
Q\,
G_\theta
\left(
A(o_v),
\operatorname{Agg}_{\mathrm{TGN}}
\{E(z_i,\xi_i)\}
\right).
$$

---

# 六、B 方案怎样统一不同维数进入一个 loss

## 单一宿主接口

如果当前实验只有一个递归接口，例如 TGN memory：

$$
r_{\tau_v}=r,
$$

那么直接使用：

$$
r_B=r,\qquad J_\tau=I.
$$

不需要任何额外映射。

## 多个不同维数接口

如果同一宿主确实存在多个接口，可使用固定的直和嵌入：

$$
J_\tau:
\mathbb R^{r_\tau}
\rightarrow
\mathbb R^{r_B},
\qquad
r_B=\sum_{\tau\in\mathcal T}r_\tau.
$$

例如：

$$
J_1(z)=[z,0,0],\qquad
J_2(z)=[0,z,0],\qquad
J_3(z)=[0,0,z].
$$

这样：

* 不同维状态可以堆叠；
* 映射是无损的；
* 不需要分类型计算多个 covariance；
* 仍然只有一个全局 loss；
* 不会错误地假设不同接口坐标具有相同语义。

如果 \(r_B\) 太大，可以对直和结果做固定 CountSketch/JL：

$$
b_v=S_BJ_{\tau_v}z_v\in\mathbb R^{q_B}.
$$

但这是计算近似，第一版接口数量不多时优先使用精确直和。

\(J_\tau\) 默认固定、不训练。这样统一映射本身不会通过学习改变考试标准。

---

# 七、训练时究竟用什么监督

对节点输出处做切口：

$$
e=(H_e,C_e,Y_e).
$$

其中：

* \(H_e\)：切口以下的完整历史；
* \(C_e\)：切口以上如何继续使用该状态；
* \(Y_e\)：该 continuation 下实际产生的未来行为。

压缩状态只能由历史产生：

$$
z_e=q_\theta(H_e).
$$

它不能读取 \(C_e\) 或 \(Y_e\)。

## 1. 固定 context 特征

$$
s_e=\chi(C_e)\in\mathbb R^{d_c}.
$$

可以包含：

* 剩余深度；
* 后续 operator/path；
* 时间与关系信息；
* 任务/query；
* 合法的 sibling/environment 信息；
* validity mask。

\(\chi\) 使用固定 one-hot、标准化、RFF、JL 或冻结宿主 embedding，不参与训练。

## 2. 固定 future 特征

使用：

$$
\mathcal H=\{1,2,\star\}.
$$

即：

* 经过一次父构造后的行为；
* 经过两次构造后的行为；
* 最终根部行为。

构造：

$$
f_e
=
\bigoplus_{h\in\{1,2,\star\}}
\sqrt{\omega_h}\,
[\varphi_h(Y_e^{(h)}),\operatorname{mask}_e^{(h)}].
$$

第一版可取：

$$
\omega_1=\frac14,\qquad
\omega_2=\frac14,\qquad
\omega_\star=\frac12.
$$

未来行为来自：

* 冻结的未压缩宿主模型；
* 或数据中真实可观察的未来结果。

它们可以离线缓存。没有 future decoder，也不在部署时运行。

---

# 八、为什么需要 context–future 联合测试

我们不能只问：

$$
H\text{ 能否预测平均未来？}
$$

而必须问：

$$
H\text{ 在不同 continuation }C\text{ 下会产生什么未来？}
$$

因此定义：

$$
\Psi(C_e,Y_e)
=
[1;\chi(C_e)]\otimes\varphi(Y_e).
$$

这对应：

$$
\chi_j(C_e)\varphi_k(Y_e),
$$

能够表示“某种 continuation 与某种未来行为如何配对”。

如果维数不大，直接使用精确张量积。若维数过大，则使用固定 TensorSketch：

$$
\boxed{
p_e
=
\operatorname{TensorSketch}
\left(
[1;s_e]\otimes f_e
\right)
\in\mathbb R^m.
}
$$

这里：

* \(m\) 是固定的 loss 测试维数；
* 它独立于宿主的 \(r_\tau\)；
* TensorSketch 只是降低张量积计算量；
* RFF 只是扩大固定的非线性测试族；
* 它们都不是模型主体。

因此：

$$
\boxed{
\Psi\text{ 定义要保留什么，}
\Gamma_\theta\text{ 负责真正保留，}
\mathcal L_{\mathrm{RPBE}}\text{ 衡量保留了多少。}
}
$$

---

# 九、唯一的训练 loss

对一个 batch 的所有切口，得到：

$$
B=
\begin{bmatrix}
b_1^\top\\
\vdots\\
b_M^\top
\end{bmatrix}
\in\mathbb R^{M\times r_B},
\qquad
P=
\begin{bmatrix}
p_1^\top\\
\vdots\\
p_M^\top
\end{bmatrix}
\in\mathbb R^{M\times m}.
$$

使用树级等权和必要的 context balancing 权重组成 \(W\)。

中心化后计算：

$$
C_{BB}=B_c^\top WB_c,
$$

$$
C_{PP}=P_c^\top WP_c,
$$

$$
C_{BP}=B_c^\top WP_c.
$$

Cholesky 分解：

$$
C_{BB}+\varepsilon_bI=L_BL_B^\top,
$$

$$
C_{PP}+\varepsilon_pI=L_PL_P^\top.
$$

得到 predictive-balancing operator：

$$
\boxed{
\mathcal B_\theta
=
L_B^{-1}
C_{BP}
L_P^{-\top}.
}
$$

以及未来测试的有效总能量：

$$
D_P
=
\operatorname{tr}
\left(
L_P^{-1}C_{PP}L_P^{-\top}
\right).
$$

最终只优化一个 loss：

$$
\boxed{
\mathcal L_{\mathrm{RPBE}}
=
\frac{
D_P-\|\mathcal B_\theta\|_F^2
}{
D_P+\eta
}.
}
$$

不再添加：

* future reconstruction；
* future decoder；
* InfoNCE；
* KL/ELBO；
* Koopman transition；
* 独立 closure loss；
* 后处理 SVD；
* 单独训练的线性 probe；
* predictive-spectrum entropy。

奇异值谱、有效秩和能量曲线只作为诊断指标，不进入主 loss。

---

# 十、为什么一个 loss 就够了

这个 loss 实际上解析消去了最优线性 probe：

$$
A^\star
=
(C_{BB}+\lambda I)^{-1}C_{BP}.
$$

所以它衡量的是：

> 从实际部署状态 \(z\) 经固定统一映射得到的 \(b\)，最多能线性读取出多少 continuation-conditioned future tests。

如果 \(\Gamma_\theta\) 把重要信息丢掉：

$$
C_{BP}\downarrow,
\qquad
\|\mathcal B_\theta\|_F^2\downarrow,
\qquad
\mathcal L_{\mathrm{RPBE}}\uparrow.
$$

如果状态坍缩成常数：

$$
C_{BP}=0,
$$

loss 反而很大，所以不需要额外 reconstruction 或防坍缩网络。

虽然最后的 probe 是线性的，但：

$$
H
\overset{\text{整树非线性 }\Gamma_\theta}{\longrightarrow}
z
\longrightarrow
b
\longrightarrow
A^{\star\top}b
$$

对原始历史 \(H\) 仍然是高度非线性的。

---

# 十一、递归闭合从哪里来

闭合不是额外靠一个 loss 猜出来，而是由网络接口直接保证：

$$
z_v
=
\Gamma_\theta
(o_v,\{z_{v_i}\}).
$$

父节点只能读取：

* 孩子的压缩状态；
* 当前合法局部输入；
* 边/时间/关系信息。

不能重新读取孩子的完整历史。

因此如果两个孩子历史产生相同状态：

$$
q_\theta(H_i)=q_\theta(H_i'),
$$

则放入相同父构造器时：

$$
\Gamma_\theta(q_\theta(H_i),o_v)
=
\Gamma_\theta(q_\theta(H_i'),o_v).
$$

这是架构上的精确递归闭合。

RPBE loss 负责的则是另一件事：

> 让这种递归闭合状态尽量接近真正的未来预测等价类。

所以不需要再增加独立 closure loss。

---

# 十二、完整训练算法

## 阶段 0：由宿主确定接口

为每个被压缩的接口确定：

$$
r_\tau.
$$

它由宿主状态形状和压缩目标决定，不由论文写死。

同时确定：

* 共享生成器内部宽度 \(D\)；
* 统一 loss 坐标维数 \(r_B\)；
* 固定未来测试维数 \(m\)。

## 阶段 1：生成训练记录

用冻结宿主模型构造完整计算树。

对每个切口保存：

$$
(H_e,C_e,Y_e^{(1)},Y_e^{(2)},Y_e^{(\star)}).
$$

数据必须按 tree ID 和时间划分 train/calibration/audit/test，避免同一棵树泄漏。

## 阶段 2：固定测试映射

固定：

$$
\chi,\quad
\varphi_h,\quad
\operatorname{TensorSketch},\quad
J_\tau.
$$

全部：

```python
requires_grad = False
```

标准化统计量只从训练集估计。

## 阶段 3：端到端训练统一生成器

每个 batch：

1. 读取完整树；
2. bottom-up 递归计算所有 \(z_v\)；
3. 不同接口输出各自的 \(r_{\tau_v}\) 维状态；
4. 映射成统一 loss 坐标 \(b_v=J_{\tau_v}z_v\)；
5. 读取固定的 \(p_v=\Psi(C_v,Y_v)\)；
6. 计算一次全局 covariance；
7. 得到唯一的 \(\mathcal L_{\mathrm{RPBE}}\)；
8. 一次反向传播更新整个 \(\Gamma_\theta\)。

内部节点从一开始就读取孩子实际产生的压缩状态，不另外训练 rich-state teacher。

## 阶段 4：冻结与部署

训练完成后只保留：

* 输入适配器 \(A_\tau\)；
* 孩子适配器 \(E_\tau\)；
* 宿主聚合器；
* 共享核心 \(G_\theta\)；
* 输出头 \(Q_\tau\)。

部署时删除：

* \(J_\tau\)；
* context/future 特征；
* TensorSketch/RFF；
* covariance；
* Cholesky；
* \(\mathcal B_\theta\)；
* 解析 probe。

实际推理只有：

$$
\boxed{
o_v,\{z_{v_i}\}
\longrightarrow
z_v\in\mathbb R^{r_{\tau_v}}.
}
$$

---

# 十三、最终伪代码

```python
for trees in loader:
    z = {}

    for v in bottom_up_order(trees):
        local = input_adapter[v.type](v.local_input)

        child_tokens = []
        for child, edge in children(v):
            child_tokens.append(
                child_adapter[child.type](
                    z[child],
                    edge.features,
                    edge.role,
                )
            )

        hidden = shared_generator(
            local=local,
            children=child_tokens,
            operator=v.operator,
        )

        # Actual host-specified bottleneck
        z[v] = output_head[v.interface](hidden)   # [r_tau(v)]

    # Fixed-dimensional B-scheme loss coordinates
    B = stack([
        fixed_loss_lift[cut.interface](z[cut.node])
        for cut in cuts(trees)
    ])                                            # [M, r_B]

    with no_grad():
        P = fixed_context_future_tests(cuts(trees))  # [M, m]
        W = tree_and_context_weights(cuts(trees))

    loss = rpbe_loss(B, P, W)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# 十四、和 ISK 的准确关系

相似之处是：

| ISK 式思路           | 我们                          |
| ----------------- | --------------------------- |
| 理论分析应该保留什么信息      | 定义 all-context 预测等价         |
| 不直接求理想理论对象        | 不直接求完整预测商                   |
| 把理论要求转成 loss      | 推出 RPBE loss                |
| 用普通非线性 encoder 拟合 | 用非线性递归 \(\Gamma_\theta\) 拟合 |
| 简单线性对象用于监督结构      | 解析 probe、白化只用于 loss         |
| 部署保留 encoder      | 部署只保留递归生成器                  |

区别是：

$$
\boxed{
\text{ISK 研究 Koopman/时间动力学结构；
我们研究树 continuation 下的预测商与递归同余。}
}
$$

我们没有：

$$
z_{t+1}=Kz_t,
$$

而是：

$$
z_v
=
\Gamma_\theta(o_v,\{z_{v_i}\}),
$$

并允许 \(\Gamma_\theta\) 完全非线性。

---

最终对外可以压缩成一句话：

$$
\boxed{
\text{宿主决定每个接口传播多少维；
统一非线性生成器负责递归压缩；
固定映射把不同宿主维数送入同一个 loss；
RPBE loss 要求这些有限状态保留 continuation-conditioned future information。}
}
$$

所以这套方法真正像 ISK 的地方，就是算法主体可以很简单：不靠复杂解算流程，主要靠一个由理论推出来的 loss，把普通非线性递归模型训练成近似预测商。
可以。把上一轮最后的缺口补上后，最终 loss 应该定为：

$$
\boxed{
\text{嵌套 profiled predictive balancing 局部缺失}
\;+\;
\text{树路径增益递推}
\;\Longrightarrow\;
\text{一个根部 loss}.
}
$$

关键修正是：局部缺失不能再用

$$
D_P-\mathcal R_{\rm PB}(z_v)
$$

直接定义，因为它包含不可预测噪声和孩子已经造成的损失，会在递归时重复计数。

最终要比较的是：

> 压缩前，本节点当前实际可用的信息能够解释多少未来；经过本节点瓶颈后，还能解释多少未来。

两者之差才是节点 \(v\) 新引入的增量缺失。

---

# 一、节点递归状态

节点 \(v\) 的孩子已经产生宿主规定维数的状态：

$$
z_{v_i}\in\mathbb R^{r_{\tau_{v_i}}}.
$$

统一非线性生成器计算：

$$
z_v
=
\Gamma_\theta
\left(
o_v,\{z_{v_i},\xi_{vi}\}_{i=1}^{k_v}
\right)
\in\mathbb R^{r_{\tau_v}}.
$$

为进入统一 loss 坐标，使用实际瓶颈之后的固定映射：

$$
\boxed{
b_v=J_{\tau_v}z_v\in\mathbb R^{r_B}.
}
$$

\(r_{\tau_v}\) 由宿主决定，可以变化；只有 loss 坐标 \(r_B\) 固定。

---

# 二、固定 continuation–future 测试

对切口 \(v\)，固定构造：

$$
p_v
=
\Psi(C_v,Y_v)
\in\mathbb R^m,
$$

其中可使用：

$$
\Psi(C_v,Y_v)
=
\operatorname{TensorSketch}
\left(
[1;\chi(C_v)]\otimes\varphi(Y_v)
\right).
$$

\(\chi,\varphi,\Psi\) 全部固定，不训练。

未来可以包含一步、两步和最终行为：

$$
Y_v=(Y_v^{(1)},Y_v^{(2)},Y_v^{(\star)}).
$$

所有 pullback/shift 测试都堆入同一个 \(p_v\)，不再形成第二个 closure loss。

---

# 三、构造“压缩前局部信息” \(u_v\)

这是解决增量缺失的关键。

令节点 \(v\) 在输出瓶颈之前实际能够读取的信息为：

$$
I_v
=
\left(
o_v,\{z_{v_i},\xi_{vi},\ell_{vi}\}_{i=1}^{k_v}
\right).
$$

使用固定、训练期专用的特征映射：

$$
s_v
=
\rho_{\mathrm{fix}}(I_v)
\in\mathbb R^{d_s}.
$$

然后构造：

$$
\boxed{
u_v=
\begin{bmatrix}
b_v\\
s_v
\end{bmatrix}
\in\mathbb R^{r_B+d_s}.
}
$$

这里必须包含 \(b_v\)，因此线性函数空间满足：

$$
\operatorname{span}(b)
\subseteq
\operatorname{span}(u).
$$

这保证“压缩前”的最优预测能力不会弱于“压缩后”。

注意：

* \(\rho_{\rm fix}\) 不是第二个 rich encoder；
* 它不训练、不传播、不部署；
* 固定二叉输入可直接拼接；
* 无序孩子使用固定的一阶、二阶矩或 TensorSketch；
* 在作为参考分支时，对 \(s_v\) 使用 `stop_gradient`，防止网络主动破坏压缩前基准。

因此：

$$
u_v
=
[b_v;\operatorname{sg}(s_v)].
$$

---

# 四、未来测试统一白化

在 profile-fit 数据上，根据权重 \(w_v\) 计算：

$$
\mu_P=\sum_vw_vp_v,
$$

$$
C_{PP}
=
\sum_vw_v(p_v-\mu_P)(p_v-\mu_P)^\top.
$$

定义：

$$
G_P=C_{PP}+\varepsilon_pI,
$$

以及白化未来测试：

$$
\widetilde p_v
=
G_P^{-1/2}(p_v-\mu_P).
$$

实际实现只用 Cholesky triangular solve，不显式求逆或矩阵平方根。

权重 \(w_v\) 应同时包含：

* 每棵树等权；
* 切口采样概率校正；
* 必要的 context overlap/density-ratio 权重。

---

# 五、分别 profile 压缩后状态和压缩前信息

对任意特征 \(x_v\)，定义加权协方差：

$$
C_{XX}
=
\sum_vw_v(x_v-\mu_X)(x_v-\mu_X)^\top,
$$

$$
C_{PX}
=
\sum_vw_v(p_v-\mu_P)(x_v-\mu_X)^\top.
$$

定义 profiled observability map：

$$
\boxed{
\mathcal O_X
=
G_P^{-1/2}
C_{PX}
(C_{XX}+\varepsilon_xI)^{-1}.
}
$$

它给出从 \(x\) 到白化未来测试的解析最优线性预测：

$$
\widehat\mu_X(v)
=
\mathcal O_X(x_v-\mu_X).
$$

分别取：

$$
X=B=\{b_v\},
\qquad
X=U=\{u_v\},
$$

得到：

$$
\widehat\mu_B(v)
=
\mathcal O_B(b_v-\mu_B),
$$

$$
\widehat\mu_U(v)
=
\mathcal O_U(u_v-\mu_U).
$$

\(\mathcal O_B,\mathcal O_U\) 都是解析 profile，不是训练网络，也不部署。

---

# 六、balancing 在哪里真正出现

对任意状态特征 \(X\)，定义 profiled predictive-balancing energy：

$$
\boxed{
\mathcal R_{\rm PB}(X)
=
\operatorname{tr}
\left[
G_P^{-1}
C_{PX}
(C_{XX}+\varepsilon_xI)^{-1}
C_{XP}
\right].
}
$$

等价地，定义有限测试层上的 predictive Hankel operator：

$$
\mathcal H_X
=
G_P^{-1/2}
C_{PX}
(C_{XX}+\varepsilon_xI)^{-1/2},
$$

则：

$$
\boxed{
\mathcal R_{\rm PB}(X)
=
\|\mathcal H_X\|_{\mathrm{HS}}^2.
}
$$

所以本节点压缩丢失的 predictive Hankel 能量为：

$$
\boxed{
\Delta_v^{\rm PB}
=
\mathcal R_{\rm PB}(U)
-
\mathcal R_{\rm PB}(B).
}
$$

在总体、无正则的嵌套线性投影条件下：

$$
\mathcal R_{\rm PB}(U)
\ge
\mathcal R_{\rm PB}(B).
$$

这一步意味着 balancing 已经真正进入 loss，而不是额外加一个“类似 balancing”的正则。

---

# 七、每个节点的增量局部缺失

不能只使用整个 batch 的标量能量差，因为后面要沿树递推。定义节点级预测差：

$$
d_v
=
\widehat\mu_U(v)-\widehat\mu_B(v).
$$

局部增量缺失为：

$$
\boxed{
\delta_v
=
\left(
\|d_v\|_2^2+\varepsilon_\delta
\right)^{1/2}.
}
$$

在无正则、总体投影的理想情况下，因为

$$
\operatorname{span}(B)\subseteq\operatorname{span}(U),
$$

有正交投影恒等式：

$$
\boxed{
\mathbb E_w[\delta_v^2]
=
\mathcal R_{\rm PB}(U)
-
\mathcal R_{\rm PB}(B).
}
$$

因此 \(\delta_v\) 的含义非常准确：

> 在孩子当前压缩状态已经固定的前提下，本节点把其可用输入进一步压成 \(z_v\) 时，新丢失的未来可观察信息。

孩子以前丢失的信息已经不在 \(I_v\) 中，所以不会再次进入 \(\delta_v\)。这就解决了重复计数问题。

有限样本中使用 ridge 后，上式会增加一个正则化扰动项，理论应写为：

$$
\left|
\mathbb E[\delta_v^2]
-
\left(
\mathcal R_{\rm PB}(U)-\mathcal R_{\rm PB}(B)
\right)
\right|
\le
\operatorname{Err}_{\rm ridge}
+
\operatorname{Err}_{\rm sample}.
$$

---

# 八、路径增益

设孩子 \(i\) 的预测测试误差经过父构造器后最多被放大 \(L_{v,i}\) 倍：

$$
\|\Delta_v^{(i)}\|_{\Psi}
\le
L_{v,i}\|\Delta_{v_i}\|_{\Psi}.
$$

理论上，\(L_{v,i}\) 是 continuation pullback 在预测测试度量中的局部 Lipschitz 常数。

工程上可以使用：

* 自动微分 JVP/VJP 估计局部算子范数；
* calibration split 上的高分位数；
* 再乘一个大于 1 的安全系数。

训练时：

$$
\boxed{
L_{v,i}\text{ 必须 stop-gradient}.
}
$$

否则网络可能仅通过人为缩小估计增益来降低 loss，而不是改善压缩。

---

# 九、树上动态规划

从叶到根递归：

$$
\boxed{
R_v
=
\delta_v
+
\sum_{i\in\operatorname{ch}(v)}
L_{v,i}R_{v_i}.
}
$$

叶节点：

$$
R_v=\delta_v.
$$

对普通树，每个节点到根只有一条路径，因此可以展开为：

$$
\boxed{
R_{\rm root}
=
\sum_{v\in T}
g_v\delta_v,
}
$$

其中：

$$
g_v
=
\prod_{(a\leftarrow b)\in\operatorname{path}(v\to\rm root)}
L_{a,b},
\qquad
g_{\rm root}=1.
$$

这说明深层节点是否重要，不仅由它自己的 \(\delta_v\) 决定，还取决于其误差在后续路径上被放大还是衰减。

---

# 十、最终唯一 loss

理论版本直接定义：

$$
\boxed{
\mathcal L_{\rm TreePB}(\theta)
=
\mathbb E_{T}
\left[
R_{\operatorname{root}(T)}^2
\right].
}
$$

展开后：

$$
\boxed{
\mathcal L_{\rm TreePB}(\theta)
=
\mathbb E_T
\left[
\left(
\sum_{v\in T}
g_v\,
\left\|
\widehat\mu_U(v)-\widehat\mu_B(v)
\right\|_2
\right)^2
\right].
}
$$

为了不同数据集尺度可比，工程上可除以固定未来测试能量：

$$
D_P
=
\operatorname{tr}
\left[
G_P^{-1}C_{PP}
\right].
$$

最终实现为：

$$
\boxed{
\mathcal L_{\rm RPBE}
=
\frac{
\frac1N\sum_{n=1}^{N}
R_{\operatorname{root}(T_n)}^2
}{
D_P+\eta
}.
}
$$

这就是唯一需要反向传播的 loss。

---

# 十一、推荐的实际计算流程

为了避免解析 probe 在同一批样本上拟合和评估造成乐观偏差，使用树级交叉拟合。

每个大 batch 按整棵树分成两折 \(A,B\)。

1. 用折 \(A\) 计算 \(\mathcal O_B,\mathcal O_U\)。

2. 在折 \(B\) 上计算：

$$
\delta_v
=
\|
\widehat\mu_U(v)-\widehat\mu_B(v)
\|.
$$

3. 在折 \(B\) 的每棵树上执行路径动态规划，得到 \(R_{\rm root}\)。

4. 交换两折再计算一次。

5. 两折根部 loss 取平均，一次反向传播。

伪代码为：

```python
z = recursive_compressor(tree_batch)        # actual host-width states

B = fixed_host_lift(z)                      # [M, r_B]
S = fixed_prebottleneck_witness(
    local_inputs,
    stopgrad(child_states),
)
U = concat([stopgrad(B), stopgrad(S)], dim=-1)

P = fixed_context_future_tests(records)     # [M, m]

O_B, O_U = cross_fitted_profile(B, U, P, weights)

mu_B = apply_profile(O_B, B)
mu_U = apply_profile(O_U, U).detach()

delta = l2_norm(mu_U - mu_B)                # one value per node
gain = calibrated_path_gain(...).detach()

for v in postorder(tree):
    root_bound[v] = delta[v] + sum(
        gain[v, child] * root_bound[child]
        for child in children(v)
    )

loss = mean(root_bound[root] ** 2) / (future_energy + eps)
loss.backward()
```

这里没有第二个学习网络。`mu_U` 是由固定 pre-bottleneck witness 和解析 profile 得到的训练目标，不是 future decoder。

---

# 十二、这版和旧 loss 的准确关系

旧 loss：

$$
D_P-\mathcal R_{\rm PB}(B)
$$

衡量的是“压缩状态对全部未来测试还缺多少”，其中混有：

* 不可预测噪声；
* 孩子已经丢失的信息；
* 当前节点新增的损失。

新局部缺失：

$$
\boxed{
\mathcal R_{\rm PB}(U)-\mathcal R_{\rm PB}(B)
}
$$

只衡量当前瓶颈新丢掉的 predictive Hankel 能量。

再通过：

$$
R_v=\delta_v+\sum_iL_{v,i}R_i
$$

把孩子损失传播上来，因而不重复计数。

---

# 十三、接下来推导应严格证明四件事

你接下来可以直接按照下面四个定理推进：

1. **嵌套 profile 恒等式**

$$
\operatorname{span}(B)\subseteq\operatorname{span}(U)
\Rightarrow
\mathbb E\|\widehat\mu_U-\widehat\mu_B\|^2
=
\mathcal R_{\rm PB}(U)-\mathcal R_{\rm PB}(B).
$$

2. **balancing 恒等式**

$$
\mathcal R_{\rm PB}(X)
=
\|\mathcal H_X\|_{\mathrm{HS}}^2,
$$

并由 \(G_R,G_O\) 说明这是生成性—可观察性的联合能量。

3. **树递归误差定理**

$$
e_v
\le
\delta_v+\sum_iL_{v,i}e_i
\Rightarrow
e_{\rm root}\le R_{\rm root}.
$$

4. **有限样本定理**

把总误差拆为：

$$
\operatorname{Err}_{\rm profile}
+
\operatorname{Err}_{\rm ridge}
+
\operatorname{Err}_{\rm RFF/sketch}
+
\operatorname{Err}_{\rm overlap}
+
\operatorname{Err}_{\rm gain}.
$$

最终方向就定为：

$$
\boxed{
\mathcal L_{\rm RPBE}
=
\mathbb E_T
\left[
\left(
\sum_{v\in T}
g_v
\left\|
\widehat\mu_U(v)-\widehat\mu_B(v)
\right\|
\right)^2
\right].
}
$$

它是一个 loss；balancing 位于每个 \(\delta_v\) 的严格定义中；树结构通过路径增益进入最终目标；整个训练仍然只优化非线性递归压缩器 \(\Gamma_\theta\)。

