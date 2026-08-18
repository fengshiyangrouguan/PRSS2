# Predictive Relation-State Sheaf（PRSS）方法规格 v1.0
## —— 面向递归/树式深度模型的宿主接口预算预测商压缩

> **用途**：本文件是直接交给代码 Agent 实现的算法规格。  
> **当前载体**：先以 TGN / temporal computation tree 做验证，但方法本身不绑定 TGN。  
> **核心目标**：对每个递归接口类型 \(	au\)，PRSS 读取宿主模型要求的接口宽度 \(k_\tau\)，并在该预算内显式学习 **未来 continuation 如何读取当前历史**，再通过 SVD/特征分解选出最有预测价值的 \(k_\tau\) 维子空间。  
> **关键原则**：从训练第一步起，父节点只接收符合宿主模型原接口宽度 \(k_\tau\) 的 quotient state；**不存在“先训练普通 TGN，再逐层压缩”**。

---

# 0. 最终决定（代码实现不得偏离）

本版本采用：

\[
\boxed{
\text{Nonlinear Candidate Lift}
+
\text{Conditional Future-Reading Matrix}
+
\text{Spectral Rank-}k\text{ Quotient}
+
\text{Original Recursive Aggregator}
}
\]

具体分工：

1. **原模型 \(F\)**：仍负责聚合。TGN 就用 TGN 的 aggregation / memory update；LSTM 就用 LSTM；其他模型同理。
2. **候选表示 \(\Phi_\omega\)**：产生较宽、较丰富的 pre-quotient representation，默认 `d=128`。
3. **未来读取网络**：训练期估计上部 continuation 如何读取当前 pre-quotient representation，输出矩阵 \(B(C)\)。
4. **SVD / eigendecomposition**：给定这些读取矩阵，在宿主接口预算 \(k_\tau\) 下求最优公共预测子空间 \(R_\tau\)。
5. **真正传入父聚合器的状态**：
   \[
   z_v=R_{\tau(v)}h_v\in\mathbb R^k.
   \]
6. **深度学习不是替代 SVD**：深度学习负责非线性 lift 和难估的条件 response；SVD 决定固定宽度里真正保留哪些方向。
7. **不使用** Information Bottleneck、dimension gate、L0 gate 作为主方法。
8. **不使用** “root-to-leaf 后处理式逐层压缩”。
9. **不允许** inference-time compression 读取未来 label、未来事件或 aggregation 后才出现的状态。
10. **不要求** forward 每聚合一次某个能量数值单调下降；我们的目标是“全 context 定义、局部接口谱最优”。

---


## 0.1 维度规则：\(k_\tau\) 完全由宿主模型决定

PRSS **不规定统一 quotient width**。对每个递归接口类型 \(\tau\)，定义：

\[
\boxed{
k_\tau=\operatorname{dim}_{\text{host}}(\tau)
}
\]

即 \(k_\tau\) 是宿主模型原本要求该接口输出/输入的维数。

例子：

- 例如，TGN 某个 message 接口原本要求 32 维，则该接口才取 \(k_\tau=32\)；
- LSTM hidden/interface 为 128 维，则 \(k_\tau=128\)；
- Transformer / attention / heterogeneous modules 若不同接口宽度不同，则分别使用各自的 \(k_\tau\)；
- 若宿主模型本来就存在 `wide candidate -> narrow message` 的 projection，PRSS 应替换或结构化这一步，而不是另造一个固定宽度；
- 若某接口 \(d_\tau=k_\tau\)，PRSS 只能学习预测相关的坐标/子空间重组，不能宣称该处发生了 dimensional compression。

统一矩阵形式：

\[
\boxed{
R_\tau\in\mathbb R^{k_\tau\times d_\tau},
\qquad
R_\tau R_\tau^\top=I_{k_\tau}.
}
\]

所有 `top-k`、SVD/eigh、spectral tail、projector 和 tensor shape 都必须使用当前接口自己的 \(k_\tau\)。

代码层面：

```python
k_tau = host_model.get_interface_dim(tau)
d_tau = config.get_candidate_dim(tau, default=k_tau)

assert d_tau >= k_tau
R_tau.shape == (k_tau, d_tau)
```

**禁止在 PRSS 核心模块里硬编码 `32`。**

---

# 1. 我们到底解决什么问题

考虑递归计算树：

\[
D,E\rightarrow A,\qquad
F,G\rightarrow B,\qquad
A,B\rightarrow C,\qquad
C\rightarrow X.
\]

每个节点/子树 \(v\) 有完整历史对象 \(H_v\)。实际深度模型必须把它编码成有限宽度状态，再交给父节点聚合。

如果接口类型 \(\tau\) 只能接收宿主模型规定的 \(k_\tau\) 维，普通深度学习通常直接学：

\[
H_v\rightarrow z_v\in\mathbb R^{k_\tau},
\]

然后只依靠最终任务 loss 决定这 \(k_\tau\) 维装什么。

问题在于：一个历史方向可能在当前节点单独看不重要，却会在与 sibling 聚合、经过更高层 continuation 后影响最终预测。仅靠普通根部梯度容易发生：

- 历史混叠；
- 提前遗忘；
- 聚合前丢掉后续交互需要的信息；
- 固定接口宽度内的状态缺乏明确预测语义。

我们的目标不是“完全无损压缩”，而是：

> **在宿主模型规定的接口宽度 \(k_\tau\) 下，选择对所有数据支持的未来 continuation 最有价值的 \(k_\tau\) 个预测方向。**

---

# 2. 理论对象：contextual predictive quotient

对于节点 \(v\) 的历史 \(H_v\)，令 \(C_v[\square]\) 表示把该子树挖掉以后剩余的合法上部 continuation（parent constructor、siblings、更高层路径、最终预测目标等）。

定义：

\[
H_v\sim H_v'
\iff
P(Y\mid C[H_v])
=
P(Y\mid C[H_v'])
\qquad
\forall C\in\mathcal C_v.
\]

也可用 characteristic / sufficiently rich output feature \(\phi(Y)\)：

\[
\mu(H,C)
=
\mathbb E[\phi(Y)\mid H,C],
\]

则

\[
H\sim H'
\iff
\mu(H,C)=\mu(H',C),\quad \forall C.
\]

## 2.1 为什么这是递归同余

若：

\[
H_D\sim H_D',
\]

则对任意 sibling \(H_E\) 及 parent constructor \(F_A\)，任意 A 上部 context \(C_A\) 都诱导一个 D-context：

\[
C_D[\square]
=
C_A[F_A(\square,H_E)].
\]

因此：

\[
P(Y\mid C_A[F_A(H_D,H_E)])
=
P(Y\mid C_A[F_A(H_D',H_E)]).
\]

于是：

\[
F_A(H_D,H_E)\sim F_A(H_D',H_E).
\]

所以：

\[
\boxed{
\text{“对所有合法 context 的预测等价”天然是树 constructor 的 congruence。}
}
\]

这说明：

- **全局 context equivalence** 定义“正确的商是什么”；
- **递归闭合** 是同一个对象的局部结构性质；
- 不需要另外发明一个新 aggregator 来实现闭合。

---

# 3. 从不可计算的全 context 商到可学习对象

完整 \(H\) 与 \(C\) 是高维、连续、自然数据只给 matched triples：

\[
(H_i,C_i,Y_i).
\]

不能构造巨大 sample-indexed Hankel table，更不能依赖任意 counterfactual \(C_j[H_i]\)。

因此引入较丰富的 pre-quotient representation：

\[
h_v=\Phi_\omega(H_v)\in\mathbb R^d,
\qquad d>k.
\]

默认：

```text
d = 128
k = 32
```

关键假设不是“真实世界线性”，而是：

> 经过一个小型 nonlinear lift \(\Phi_\omega\) 后，未来 continuation 对历史表示的读取可在该 feature space 中近似成线性算子。

即：

\[
\boxed{
\mu(h,C)
\approx
D_\psi\!\left(b_\eta(C)+B_\eta(C)h\right).
}
\]

其中：

\[
B_\eta(C)\in\mathbb R^{p\times d}
\]

就是 **future-reading matrix**。

它的语义：

> 在 continuation \(C\) 下，哪些 history directions 会改变未来 response。

---

# 4. 为什么最终必须做 SVD

对所有 supported continuations，理论上的最小线性预测商满足：

\[
\ker R^*
=
\bigcap_C\ker B(C),
\]

等价于：

\[
\operatorname{row}(R^*)
=
\operatorname{span}_C\operatorname{row}(B(C)).
\]

如果允许无限维，保留整个 span 即可。

但宿主接口 \(	au\) 只允许 \(k_\tau\) 维。因此我们求该接口预算下的最佳 rank-\(k_\tau\) 近似。

令：

\[
R\in\mathbb R^{k\times d},
\qquad
RR^\top=I_k,
\]

以及 projector：

\[
P_R=R^\top R.
\]

定义 **predictive operator residual energy**：

\[
\boxed{
\mathcal E_{\rm PQ}(R)
=
\mathbb E_C
\left[
\|B(C)(I-P_R)\|_F^2
\right].
}
\]

其意义非常直接：

> 被投影删除的 history directions 中，还剩多少 continuation-visible predictive operator energy。

定义：

\[
G
=
\mathbb E_C[B(C)^\top B(C)]
\in\mathbb R^{d\times d}.
\]

则：

\[
\mathcal E_{\rm PQ}(R)
=
\operatorname{tr}[(I-P_R)G].
\]

因此：

\[
\min_{RR^\top=I_k}
\mathcal E_{\rm PQ}(R)
\]

等价于：

\[
\max_{RR^\top=I_k}
\operatorname{tr}(RGR^\top).
\]

其最优解就是 \(G\) 前 \(k\) 个特征向量：

\[
G=V\Lambda V^\top,
\]

\[
\boxed{
R^*
=
V_{1:k}^{\top}.
}
\]

若将多个 \(B(C)\) 纵向 stack：

\[
\mathcal B=
\begin{bmatrix}
B(C_1)\\
B(C_2)\\
\vdots
\end{bmatrix},
\]

则这等价于对 \(\mathcal B\) 做 truncated SVD，取前 \(k\) 个 right singular vectors。

因此：

\[
\boxed{
\text{SVD 不是启发式，也不是仅用于初始化；
它是固定 rank-}k\text{ predictive-operator approximation 的解析最优子问题。}
}
\]

尾谱：

\[
\boxed{
E_k^*
=
\sum_{j>k}\lambda_j
}
\]

直接衡量当前宿主接口预算 \(k_\tau\) 无法容纳的剩余 predictive operator energy。

---

# 5. 深度学习到底负责什么

深度学习只负责 SVD 无法自己解决的部分：

## 5.1 Nonlinear candidate lift

\[
h_v=\Phi_\omega(x_v^{raw})\in\mathbb R^d.
\]

目标是把复杂非线性历史依赖展开到一个较丰富 feature space，使未来 response 对 \(h_v\) 更容易由矩阵 \(B(C)\) 读取。

第一版建议：

```text
Linear/MLP:
raw_dim -> 128 -> 128
activation: GELU
LayerNorm
residual if dimensions allow
```

不要上大网络。

## 5.2 Conditional future-response estimation

训练：

\[
B_\eta(C),\quad b_\eta(C)
\]

使：

\[
\hat P(Y\mid h,C)
\]

尽量逼近真实：

\[
P(Y\mid H,C).
\]

例如分类任务：

\[
\operatorname{logits}(Y\mid h,C)
=
b_\eta(C)+B_\eta(C)h.
\]

然后：

\[
\mathcal L_{\rm resp}
=
\operatorname{CE}(Y,\hat Y).
\]

连续输出可用：

- Gaussian NLL；
- Huber / MSE（若只关心条件均值）；
- distributional head（若任务要求完整分布）。

**重要**：第一版不要直接上 unrestricted `MLP(h, C)` 作为主 reader，因为那样 \(B(C)\) 的全局矩阵语义消失，SVD 只能退成 Jacobian 局部近似。

---

# 6. context \(C_v\) 怎么表示：推荐 training-only outside encoder

这是实现中的关键点。

`future reader` 训练时需要知道：

> 在挖掉当前子树 \(v\) 后，剩余上部 continuation 是什么。

但 \(C_v\) **不能包含当前 \(h_v\) 本身**，否则 reader 可以绕过“读取矩阵”的语义。

推荐使用 computation tree 的 **inside/outside 式辅助编码**。

## 6.1 Main inside states

主模型 bottom-up 得到每个子树的 candidate：

\[
h_v
\]

以及真正传播状态：

\[
z_v=R_{\tau(v)}h_v.
\]

## 6.2 Training-only outside state

定义：

\[
c_v=O_\eta(C_v).
\]

对 root：

\[
c_{\rm root}
=
E_{\rm root}(\text{query/root metadata}).
\]

对子节点 \(v\)（parent 为 \(p\)）：

\[
\boxed{
c_v
=
O_\eta
\left(
c_p,\;
o_p,\;
r_{vp},\;
\Delta t,\;
\operatorname{Agg}_{s\in\operatorname{siblings}(v)}
\tilde h_s
\right).
}
\]

其中：

- `c_p`：父节点外部 context；
- `o_p`：父节点当前可用 local/event metadata；
- `r_vp`：关系/constructor 类型；
- `Δt`：时间差；
- `siblings(v)`：同一个 parent 的其他分支；
- `tilde h_s`：训练期 sibling candidate，可以使用 pre-quotient candidate 的 `detach()` 版本，以避免当前 quotient 错删导致 teacher 一起失明。

**严格禁止**把 \(h_v\) 自己放入 \(c_v\)。

这样只需一次 bottom-up + 一次 top-down outside pass，复杂度与树大小线性相关。

---

# 7. Future-reading matrix generator

对每个接口类型 \(\tau\)：

\[
\boxed{
B_v
=
M_{\eta,\tau}(c_v)
\in\mathbb R^{p\times d}.
}
\]

以及：

\[
b_v=b_{\eta,\tau}(c_v).
\]

推荐共享方式：

```text
one shared context encoder
+ type/relation embedding
+ small type-conditioned matrix head
```

不要每个节点单独参数化矩阵。

第一版 `B` 是 **training-only reader operator**，不是 inference-time dynamic compression matrix。

---

# 8. 真正部署的 quotient matrix

对接口类型 \(\tau\)，维护 predictive Gram：

\[
G_\tau
=
\mathbb E_{v:\tau(v)=\tau}
[B_v^\top W_v B_v].
\]

V1：

\[
W_v=I.
\]

使用 EMA：

\[
\boxed{
G_\tau^{EMA}
\leftarrow
(1-\rho)G_\tau^{EMA}
+
\rho\,G_\tau^{batch}.
}
\]

然后周期性 eigendecomposition：

\[
G_\tau^{EMA}
=
V_\tau\Lambda_\tau V_\tau^\top,
\]

\[
\boxed{
R_\tau
=
V_{\tau,1:k}^\top.
}
\]

真正 main forward：

\[
\boxed{
z_v
=
R_{\tau(v)}h_v
\in\mathbb R^k.
}
\]

父节点 **只接收 \(z_v\)**。

`B_v`、`c_v`、outside encoder 都不参与 inference。

---

# 9. 关键：不是“response NN → SVD”两个互不相干模块

如果只优化 response loss：

\[
\mathcal L_{\rm resp},
\]

reader 可能需要大于宿主接口预算 \(k_\tau\) 的 predictive span。

我们真正希望 deep lift / reader **把预测信息组织成可由当前宿主接口预算 \(k_\tau\) 承载的形态**。

因此加入 spectral-tail loss：

\[
\boxed{
\mathcal L_{\rm spec}
=
\mathbb E_v
\frac{
\|B_v(I-P_{R_{\tau(v)}})\|_F^2
}{
\|B_v\|_F^2+\epsilon
}.
}
\]

训练时：

- \(R_\tau\) 来自最近一次 SVD；
- 对 \(R_\tau\) `stop_gradient / detach`；
- 梯度通过 \(B_v\)、context reader、必要时 \(\Phi_\omega\) 回传。

作用：

- `L_resp` 防止 \(B\to0\) 的 trivial collapse；
- `L_spec` 逼迫预测读取依赖尽量集中到当前 \(k_\tau\) 维商子空间；
- SVD 再对当前 operator bank 给出 rank-\(k_\tau\) 最优 \(R\)。

这是一个 block-coordinate / alternating spectral-neural optimization，而不是后处理。

---

# 10. Main task loss

主模型从训练第一步就使用 quotient state：

\[
h_D
\rightarrow
R_Dh_D=z_D
\rightarrow
F_A
\rightarrow
h_A
\rightarrow
R_Ah_A=z_A
\rightarrow\cdots
\]

最终：

\[
\hat Y_{\rm task}.
\]

主任务 loss：

\[
\mathcal L_{\rm task}
=
\ell(Y,\hat Y_{\rm task}).
\]

总目标：

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm task}
+
\lambda_{\rm resp}\mathcal L_{\rm resp}
+
\lambda_{\rm spec}\mathcal L_{\rm spec}.
}
\]

V1 **不要**再加入：

- Information Bottleneck KL；
- gate sparsity；
- latent reconstruction；
- 新的 learned quotient aggregator；
- root-to-leaf closure distillation。

先验证这三项足够。

---

# 11. 为什么这不是“先训练普通 TGN 再压缩”

从 step 0 起，main branch 就是：

\[
h_v
\rightarrow
R_\tau h_v
\rightarrow
F_{\rm parent}.
\]

父节点永远没看到完整 \(h_v\)。

训练循环：

1. 用当前 \(R_\tau\) 做 **compressed bottom-up forward**；
2. 得到主任务 loss；
3. 保存每个接口压缩前的 `h_v`；
4. training-only outside pass 得到 `c_v`；
5. reader 输出 \(B_v\)，计算 `L_resp` 与 `L_spec`；
6. 反向更新 base model / lift / context reader；
7. 累积 \(B_v^\top B_v\)；
8. 每隔若干步对所有 \(\tau\) **并行**更新 SVD；
9. 下一轮仍然直接跑 compressed model。

不存在：

```text
vanilla TGN fully trained
→ root layer compression
→ parent layer compression
→ leaf layer compression
```

---

# 12. 训练算法伪代码

```python
initialize base_model
initialize lift_phi
initialize outside_encoder
initialize reader_heads
initialize R_tau as orthonormal kxd matrices
initialize G_ema_tau = eps * I

for step, batch in enumerate(train_loader):

    # ---------------------------------------------------------
    # A. MAIN BOTTOM-UP FORWARD: compressed from the first step
    # ---------------------------------------------------------
    tree = build_computation_tree(batch)

    for v in postorder(tree):
        raw_candidate_v = base_model.make_candidate(
            compressed_children=[z[c] for c in children(v)],
            local_features=local_features(v),
        )

        h[v] = lift_phi(raw_candidate_v)      # d-dimensional
        z[v] = R[type(v)] @ h[v]             # k-dimensional

    y_hat = base_model.readout(z[root])
    L_task = task_loss(y_hat, y_true)

    # ---------------------------------------------------------
    # B. TRAINING-ONLY OUTSIDE CONTEXT PASS
    # ---------------------------------------------------------
    c[root] = outside_encoder.root_context(root_metadata)

    for p in preorder(tree):
        for v in children(p):
            sibling_summary = aggregate([
                stopgrad(h[s]) for s in siblings(v)
            ])

            c[v] = outside_encoder(
                parent_outside=c[p],
                parent_local=local_features(p),
                relation=relation(v, p),
                delta_t=delta_t(v, p),
                sibling_summary=sibling_summary,
            )

    # ---------------------------------------------------------
    # C. FUTURE READER
    # ---------------------------------------------------------
    L_resp = 0
    L_spec = 0
    batch_gram = {tau: 0 for tau in interface_types}

    for v in nodes(tree):
        tau = type(v)

        B_v, bias_v = reader_head[tau](c[v])
        logits_v = bias_v + B_v @ h[v]

        L_resp += proper_response_loss(logits_v, y_true)

        P = R[tau].T @ R[tau]        # d x d, R detached buffer
        residual_B = B_v @ (I - P)

        L_spec += (
            frob_sq(residual_B)
            / (frob_sq(B_v) + eps)
        )

        batch_gram[tau] += stopgrad(B_v.T @ B_v)

    # ---------------------------------------------------------
    # D. NEURAL UPDATE
    # ---------------------------------------------------------
    L = L_task + lambda_resp * L_resp + lambda_spec * L_spec

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    # ---------------------------------------------------------
    # E. SPECTRAL STATISTICS UPDATE
    # ---------------------------------------------------------
    for tau in interface_types:
        G_ema[tau] = (
            (1-rho) * G_ema[tau]
            + rho * normalize(batch_gram[tau])
        )

    # ---------------------------------------------------------
    # F. PERIODIC SPECTRAL QUOTIENT UPDATE
    # ---------------------------------------------------------
    if step % spectral_update_interval == 0:
        for tau in interface_types:
            eigvals, eigvecs = eigh(G_ema[tau])

            V_top = eigvecs[:, -k:]          # d x k
            R_new = V_top.T                  # k x d

            # Align basis to old R to avoid sign/rotation jumps
            R_new = procrustes_align(R_new, R[tau])

            R[tau] = stopgrad(R_new)
```

---

# 13. 为什么需要 Procrustes alignment

SVD / eigendecomposition 的子空间是唯一对象，但 basis 有：

- sign ambiguity；
- permutation ambiguity；
- repeated/close eigenvalues 下的 orthogonal rotation ambiguity。

父聚合器实际消费的是坐标：

\[
z=Rh.
\]

如果每次 eigendecomposition 让 basis 任意翻转/旋转，会造成训练震荡。

所以每次得到 \(R_{\rm new}\) 后，对齐旧 basis：

\[
Q^*
=
\arg\min_{Q^\top Q=I}
\|QR_{\rm new}-R_{\rm old}\|_F.
\]

用 SVD 求 orthogonal Procrustes，然后：

\[
R_{\rm new}\leftarrow Q^*R_{\rm new}.
\]

**不要直接 EMA 两个 R 矩阵**；优先对 Gram 做 EMA，再 eigendecompose。

---

# 14. TGN 载体如何插入

不要把整个 TGN memory 当成一个先训练好的历史向量。

建议接口位置：

```text
historical/local/event information
        ↓
candidate message / candidate state h (d=128)
        ↓
PRSS projection R_tau
        ↓
z (k_tau = host interface width)
        ↓
TGN aggregation / memory update
        ↓
next candidate h
        ↓
PRSS projection
        ↓
...
```

即：

\[
\boxed{
\text{compress before each recursive aggregation interface.}
}
\]

父节点/TGN update 只接收压缩消息。

如果某个 TGN 接口原版的 hidden/message width 是 \(k_\tau\)：

- 保持该接口对外宽度 \(k_\tau\) 不变；
- 若原 message encoder 存在更宽的中间 candidate，则直接把该中间宽度作为 \(d_\tau\)；
- 若没有，则可增加轻量 candidate lift 得到 \(d_\tau\ge k_\tau\)；
- PRSS 只负责 \(d_\tau
ightarrow k_\tau\)；
- 原 aggregation / GRU memory update 的接口 contract 不改变。

例如宿主 TGN 恰好要求 32 维时才有 \(k_\tau=32\)；这只是一个实例，不是 PRSS 的方法常数。

---

# 15. interface type \(\tau\) 怎么定义

V1 不要每个 occurrence 一个 \(R\)。

推荐：

\[
\tau
=
(\text{relation type},\text{message/update role})
\]

例如 TGN：

```text
src_to_dst / relation_r
dst_to_src / relation_r
optional self-update
```

若数据 relation 单一，则先全局共享一个 \(R\)。

后续只有在实验证明不同 relation 的 predictive subspace principal angles 很大时，再做：

```text
shared core + relation-specific subspace
```

不要一开始过度参数化。

---

# 16. 时间与信息泄漏约束

## Main inference path 可以看

仅允许压缩发生时已经可用的信息：

- 当前节点 memory/history；
- 当前 event；
- 当前 edge/relation type；
- 当前时间与 past time gap；
- parent 在此次 update 前已有的状态（若原模型本身可访问）。

## Main inference path 禁止看

- 当前待预测 label；
- 未来事件；
- aggregation 后的新 parent state；
- 完整 upper continuation；
- test future labels。

## Training-only reader 可以用

- 当前训练样本真实 computation tree 的 upper context；
- sibling branches；
- root/query metadata；
- training label \(Y\) 只通过 loss 使用。

注意：

\[
Y
\]

**不能作为 reader/context encoder 输入**，只能作为监督目标。

## 验证/测试

- reader/outside encoder 不参与 inference；
- \(R_\tau\) 用训练期统计得到并冻结；
- 不得用 validation/test label 更新 \(G_\tau\) 或重新 SVD；
- 若以后做在线版本，只能在真实 label 已揭晓后用于更晚时刻。

---

# 17. 理论上“闭合”在哪里

本方法有两层闭合含义，必须区分。

## 17.1 理论精确闭合

若等价关系由 **所有合法 context** 定义：

\[
H\sim H'
\iff
P(Y\mid C[H])=P(Y\mid C[H']),
\quad\forall C,
\]

则它天然对 parent constructors 构成 congruence。

这是理论定义层面的 exact result。

## 17.2 学习后的近似闭合

我们只从自然数据中估计：

\[
B(C)
\]

并且固定 rank \(k\)。

因此得到的是：

\[
\boxed{
\text{supported-context + fixed-rank approximate quotient}.
}
\]

此外，main architecture 从一开始只让 parent 看：

\[
z=Rh,
\]

所以模型计算本身不会在 parent 再偷偷访问被删掉的 child dimensions。

因此：

- **结构上**：递归只能通过 quotient state；
- **语义上**：`L_resp + SVD + L_spec` 让 quotient 尽量逼近 contextual predictive equivalence。

不要声称有限样本下已经得到 exact minimal quotient。

---

# 18. 和 NSD / Sheaf 的关系

可借鉴但不要混同。

## NSD

学习局部 matrix-valued restriction maps，用 restriction maps 构造 sheaf diffusion / Laplacian。其理论中心是 sheaf geometry、harmonic spaces、diffusion 等。

## PRSS

学习 continuation-dependent future-reading operators：

\[
B(C),
\]

再构造：

\[
G_{\rm pred}
=
\mathbb E[B(C)^\top B(C)],
\]

其谱给出固定 rank 的 predictive quotient：

\[
R_k.
\]

结构类比：

\[
\mathcal F_{v,e}
\leftrightarrow
B(C),
\]

\[
\delta_{\mathcal F}^\top\delta_{\mathcal F}
\leftrightarrow
G_{\rm pred},
\]

但**不要在论文里声称 \(G_{\rm pred}\) 就是 sheaf Laplacian**。

语义不同：

```text
NSD: relation-dependent consistency / transport geometry
PRSS: continuation-dependent future observability
```

---

# 19. 和 spectral PSR / Hankel / tree automata 的关系

理论上更直接的血统是：

\[
\boxed{
\text{context-history response operator}
\rightarrow
\text{low-rank predictive state}
}
\]

这与：

- Hankel operator / weighted automata；
- weighted tree automata；
- spectral predictive-state learning；

的思想一致。

我们的区别在于自然连续数据下不显式补全 sample Hankel table，而是：

\[
C
\xrightarrow{\text{neural reader}}
B(C)
\]

形成一个 **functionalized response operator**，再对其 history mode 做谱截断。

---

# 20. V1 为什么不用 Jacobian-SVD

可选 fallback：

\[
\hat\mu_\psi(h,C)
\]

使用 unrestricted NN，然后：

\[
J(h,C)
=
\partial_h\hat\mu_\psi(h,C),
\]

\[
G=E[J^\top J],
\]

再 top-k eigendecomposition。

这个更灵活，但有两个缺点：

1. Jacobian 是局部一阶对象；
2. variance 和计算开销更大；
3. “统一 future-reading matrix”解释变弱。

因此 V1 优先：

\[
\boxed{
\text{nonlinear lift }h=\Phi(x)
+
\text{linear-in-}h\text{ matrix reader}.
}
\]

只有当下面诊断失败才切换 Jacobian 版本。

---

# 21. 最重要的诊断

代码必须记录以下指标。

## 21.1 Response quality

比较：

```text
structured matrix reader:
    b(C) + B(C) h

vs

unrestricted MLP reader:
    MLP(h, C)
```

若 unrestricted MLP 明显更好，说明当前 nonlinear lift 不足，不能盲信 SVD。

优先处理顺序：

1. 增加 lift 表达力；
2. 增大 `d`；
3. 再考虑 Jacobian-SVD fallback。

## 21.2 Spectral concentration

记录：

\[
\lambda_1,\ldots,\lambda_d
\]

以及：

\[
\boxed{
\text{energy@k}
=
\frac{\sum_{j=1}^{k}\lambda_j}
{\sum_{j=1}^{d}\lambda_j}.
}
\]

应围绕当前接口预算 \(k_\tau\) 记录谱覆盖。例如可报告：

```text
energy@floor(k_tau/4)
energy@floor(k_tau/2)
energy@k_tau
energy@min(2*k_tau, d_tau)
```

若为了跨模型展示，也可以额外报告统一的绝对维数，但不能用这些绝对维数决定 PRSS 的接口宽度。

若 `energy@k_tau` 很低，则宿主接口给定的 \(k_\tau\) 对当前 predictive operator 来说本身过于窄。

## 21.3 Spectral tail

\[
\boxed{
\text{tail@k}
=
1-\text{energy@k}.
}
\]

这就是当前 function class 中 rank-k quotient 的不可避免 operator residual。

## 21.4 Subspace stability

连续 SVD 更新之间计算：

- principal angles；
- projector distance：

\[
\|P_t-P_{t-1}\|_F.
\]

若长期剧烈变化，reader 尚未收敛或 context 分布非平稳。

## 21.5 Task performance

和以下 baseline 对比：

1. vanilla original model；
2. random orthogonal \(d_\tau\to k_\tau\)；
3. PCA \(d_\tau\to k_\tau\)；
4. end-to-end learned Linear/MLP \(d_\tau\to k_\tau\)；
5. ridge / linear future reader + SVD；
6. **PRSS neural reader + SVD**；
7. optional Jacobian-SVD。

## 21.6 History-mixing diagnostic

构造/筛选：

- 当前局部预测相近；
- 但不同 sibling/context 下未来结果不同；

的 pairs。

检查 PRSS 是否比普通 \(k_\tau\)-dim bottleneck 更能区分这些 histories。

这是验证理论主张最重要的机制实验之一。

---

# 22. 必做消融

至少：

```text
A. no spectral update: fixed random R
B. PCA R
C. direct trainable projection R (no SVD)
D. B(C)+SVD, but no nonlinear lift
E. nonlinear lift + B(C)+SVD
F. E but remove L_spec
G. full PRSS: task + response + spectral-tail
H. optional unrestricted response + Jacobian-SVD
```

若 full PRSS 的提升只来自更大的 lift，而不是 spectral mechanism，则论文主张不成立。

---

# 23. 初始超参数建议（先跑小实验）

```yaml
# REQUIRED: derive per interface/type from the host model.
quotient_width:
  source: host_model_interface
  # e.g. k_tau may be 32, 64, 128, ... and may differ across tau

candidate_width:
  source: config_or_existing_preprojection
  # d_tau >= k_tau; usually d_tau > k_tau for nontrivial compression

lift:
  hidden_dim: auto_or_configured  # per interface; typically d_tau
  layers: 2
  activation: gelu
  layer_norm: true

outside_encoder:
  dim: 64
  layers: 2

reader:
  context_dim: 64
  # output dimension p follows predictive target parameterization
  matrix_head_rank: null  # V1 general matrix; do not low-rank B itself initially

loss:
  lambda_task: 1.0
  lambda_resp: 1.0
  lambda_spec: 0.1

spectral:
  gram_ema_rho: 0.05
  update_interval_steps: 200
  ridge_eps: 1.0e-5
  procrustes_align: true

training:
  no_vanilla_pretrain: true
  detach_sibling_in_outside_teacher: true
  detach_R_between_spectral_updates: true
```

`lambda_spec` 做：

```text
0
0.01
0.1
0.5
1.0
```

的小范围敏感性即可，不要大规模调参。

---

# 24. 初始化

禁止：

```text
先训练完整普通 TGN
→ 再提取 h
→ 再训练 PRSS
```

V1 初始化：

1. `lift / base model / reader` 正常随机初始化；
2. \(R_\tau\) 用随机 semi-orthogonal 或 identity-like projection；
3. 从第一批开始模型就使用宿主接口规定的 \(k_\tau\)-dim quotient；
4. 先累计若干 batch 的 \(G_\tau\)；
5. 到第一次 `spectral_update_interval` 再进行第一次 eigendecomposition；
6. 以后同步更新。

可以有 “spectral statistics warm-up”，但**不能有 uncompressed model warm-up**。

---

# 25. 数值实现注意

## Gram

确保：

\[
G_\tau
=
\frac{1}{N_\tau}
\sum B_v^\top B_v.
\]

然后：

```python
G = 0.5 * (G + G.T)
G += eps * I
```

再 `torch.linalg.eigh`。

## SVD 不参与普通 backward

V1：

```python
with torch.no_grad():
    eigvals, eigvecs = torch.linalg.eigh(G_ema)
    R = eigvecs[:, -k:].T
```

训练通过 block-coordinate 更新，而不是 differentiation-through-SVD。

## 防止 reader collapse

必须：

- 保留 `L_resp`；
- 监控 `||B||_F`；
- 监控 unrestricted reader 对照；
- 不能只优化 `L_spec`。

## 防止 reader 无限放大

`L_spec` 使用归一化形式：

\[
\frac{\|B(I-P)\|_F^2}{\|B\|_F^2+\epsilon}.
\]

必要时对 reader 加轻微 weight decay。

---

# 26. 当前方案“闭环”到什么程度

## 已经闭环的部分

### A. 数学目标

全 context predictive equivalence：

\[
P(Y|H,C)
\]

定义正确商。

### B. 递归性质

全 context equivalence 对 tree constructors 是 congruence。

### C. 可计算近似

用 nonlinear lift + conditional matrix reader 近似 context-history response operator。

### D. 固定宽度最优性

给定 reader operator bank，rank-\(k\) 最优公共 history subspace 由 SVD / top eigenspace 给出。

### E. 深度学习作用

`L_resp` 估计复杂 response；`L_spec` 让 feature/reader 把 predictive dependence组织到固定 rank 里。

### F. 工程递归

从训练第一步起，parent 只看 \(z=Rh\)，无事后压缩。

### G. 因果/泄漏边界

future/context 仅用于训练期 reader supervision；部署 quotient 不读取未来。

---

# 27. 还没有完成严格证明、论文里暂时不能过度声称的部分

## 27.1 有限样本一致性

还需要证明或至少给出条件：

\[
\hat B(C)
\rightarrow
B^*(C)
\]

时：

\[
\hat G
\rightarrow
G^*
\]

以及 top-\(k\) 子空间恢复误差。

可后续用 Davis–Kahan 类型谱扰动界处理。

## 27.2 nonlinear lift 的 identifiability

\(\Phi_\omega\) 与 \(B(C)\) 存在 gauge / reparameterization 自由度。

论文需要强调我们关心的是：

\[
\operatorname{rowspace}(R)
\]

及预测行为，而不是特定 latent 坐标。

## 27.3 observed contexts 不等于所有 legal contexts

自然数据只覆盖 supported continuations。

所以理论 exact quotient 与实际 learned quotient 之间存在 support gap。

必须通过：

- rolling split；
- unseen context/relation combinations；
- longer-depth trees；
- OOD temporal windows；

验证泛化。

## 27.4 rank-k operator error 到最终 task risk 的全树上界

当前最自然的是逐接口 operator residual：

\[
\sum_\tau \operatorname{tail}_k(G_\tau).
\]

但它如何经非线性递归 \(F\) 累积成最终 prediction-risk bound，还需要 Lipschitz / stability 条件。

这应作为理论下一阶段，而不是现在编码的 blocker。

---

# 28. 论文可主张与不可主张

## 可以主张（若实验支持）

- learning a fixed-budget predictive quotient before recursive aggregation；
- contextual future-reading operators；
- spectral extraction of the optimal rank-\(k\) subspace for the learned operator family；
- end-to-end recursive use from the beginning of training；
- training-only future supervision without inference leakage；
- improved retention of context-dependent historical information.

## 暂时不要主张

- exact minimal sufficient statistic in finite data；
- exact quotient over all possible future contexts；
- globally optimal nonlinear compression；
- every aggregation monotonically decreases an energy；
- universal quotient shared by arbitrary TGN/LSTM/LLM without retraining；
- NSD/sheaf theory directly proves our method.

---

# 29. 代码 Agent 的第一阶段任务

实现时不要一口气改完整大项目。

建议拆为：

## Phase 1 — 独立 PRSS 模块与单元测试

实现：

```text
rpq/
  lift.py
  outside_context.py
  reader.py
  spectral.py
  losses.py
  state.py
```

测试：

1. `B.T @ B` shape；
2. `eigh` top-k；
3. `R @ R.T ≈ I`；
4. Procrustes alignment；
5. spectral tail calculation；
6. no gradient through R；
7. gradient does reach reader/lift through L_resp/L_spec；
8. toy low-rank operator 能恢复正确 subspace。

## Phase 2 — synthetic tree sanity test

人工构造：

\[
D,E\rightarrow A\rightarrow Y
\]

其中：

- 一部分 D directions 单独与 Y 无相关；
- 与 E 交互后决定 Y；
- true predictive subspace 已知。

验证：

- direct supervised \(k_\tau\)-dim bottleneck 容易丢交互方向；
- PRSS reader + SVD 能恢复 true subspace；
- top-k eigenvectors 与真 subspace principal angle 小。

**这个 synthetic test 通过以前，不要直接跑 TGN 大实验。**

## Phase 3 — TGN integration

将 PRSS 插入 message→aggregation 接口。

确认：

- 父节点只收到 k 维；
- training outside branch 不进入 inference；
- validation/test 不更新 Gram/SVD。

## Phase 4 — UCI / Enron 初步实验

先一个 seed 验证：

- training health；
- spectrum；
- response quality；
- subspace stability；
- main task。

再跑多 seed。

---

## 29.1 宿主接口维度兼容性

实现必须验证：

- `k_tau` 来自宿主模型接口，不来自 PRSS 默认常数；
- `R_tau.shape == (k_tau, d_tau)`；
- `z_v.shape[-1] == k_tau`；
- 不同接口允许不同 `k_tau`；
- SVD/eigh 始终截断到当前接口自己的 `k_tau`；
- PRSS 与 baseline 的接口输出宽度完全一致，避免因 hidden width 改变造成不公平容量收益；
- 当 `d_tau == k_tau` 时，记录为“无维数压缩兼容模式”，不能报告 compression ratio > 1。

---

# 30. 最关键的验收标准

算法不是“能跑”就算成功。

至少满足：

### 1. Reader 真能读未来

structured reader 与 unrestricted MLP reader 差距不能过大。

### 2. 宿主接口预算 \(k_\tau\) 有谱依据

\[
\text{energy@}k_\tau
\]

必须明显高于 random/PCA 的 predictive energy coverage。

### 3. SVD 有用

full PRSS 必须优于：

```text
same lift + direct learned d_tau -> k_tau projection
```

否则 spectral quotient 没有额外贡献。

### 4. 递归机制有用

在 interaction/history-mixing 样本上，PRSS 必须明显减少错误混叠。

### 5. 不是参数量收益

baseline 要做 parameter-matched。

### 6. 没有未来泄漏

严格 temporal split；test label 不参与 Gram 或 R 更新。

---

# 31. 一句话方法定义

\[
\boxed{
\textbf{
PRSS learns how observed future continuations linearly read a nonlinear history feature space,
then uses the top-}k\textbf{ spectral subspace of these reading operators
as the only state allowed to cross each recursive aggregation interface.
}
}
\]

中文：

> **PRSS 先用神经网络估计不同真实 continuation 如何读取当前历史候选空间，再按宿主模型每个接口给定的 \(k_\tau\)，取读取算子的公共 top-\(k_\tau\) 谱子空间作为该接口唯一允许传播的状态；原模型聚合器和接口宽度保持不变。**

---

# 32. 与现有理论工作的参考关系

实现和写论文时可参考以下原始工作，但不要把 PRSS 写成它们的直接实例。

1. **Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs**, NeurIPS 2022.  
   Matrix-valued restriction maps / learned sheaf structure：  
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html

2. **Sheaf Hypergraph Networks**, NeurIPS 2023.  
   General / diagonal / low-rank restriction parameterizations：  
   https://proceedings.neurips.cc/paper_files/paper/2023/file/27f243af2887d7f248f518d9b967a882-Paper-Conference.pdf

3. **Sufficient Dimension Reduction via Direct Estimation of the Gradients of Logarithmic Conditional Densities**, AISTATS 2015.  
   Conditional-response gradient / OPG 与 sufficient dimension reduction：  
   https://proceedings.mlr.press/v45/Sasaki15.html

4. **Low-Rank Approximation of Weighted Tree Automata**, 2015.  
   Tree/Hankel predictive structure 与 SVD low-rank approximation：  
   https://arxiv.org/abs/1511.01442

5. **Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning**, AISTATS 2019.  
   Spectral predictive representations 与 recurrent neural models：  
   https://proceedings.mlr.press/v89/rabusseau19a.html

---

# 33. 最后再次强调：禁止代码 Agent 自动“简化”成以下方案

如果代码 Agent 提议以下任一项，先拒绝：

```text
1. 先训练 vanilla TGN，再离线提取 hidden 做 SVD
2. 先从 root 算好，再一层层向 leaf 压缩
3. 直接 MLP \(d_\tau\to k_\tau\)，删除 SVD
4. 用 PCA 替代 predictive operator SVD
5. 用 Information Bottleneck / VAE / KL 作为主压缩目标
6. 用 gate 决定保留多少维
7. inference 时把未来 context / label 输入 matrix generator
8. 每条样本动态旋转一个完全不同的 \(k_\tau\)-dim quotient basis（V1 不做）
9. 新训练一个 aggregator 替换原 TGN 聚合器
10. 只靠 root task loss 学矩阵，不训练 future-response reader
11. 让 SVD 只作为初始化然后失去作用
12. 在 test label 上更新 G 或重新求 R
```

当前 V1 的核心必须保持：

\[
\boxed{
\text{neural response estimation}
\rightarrow
\text{predictive operator bank}
\rightarrow
\text{SVD top-}k
\rightarrow
\text{compressed recursive forward}
}
\]

而且四者从训练早期开始同步迭代，而不是串行后处理。
