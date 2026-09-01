# LLM Recursive Memory 实验最终规格（CCM-merge × DailyDialog）

> **版本**：Final Draft v1  
> **日期**：2026-08-31  
> **用途**：作为 LLM 跨域实验的最终实现与实验协议；可直接交给 coding AI / 实验执行者。  
> **核心原则**：不在 LLM 上重复 TGN 已完成的完整消融；只验证同一“未来充分的递归压缩”原则能否迁移到 LLM memory。  
> **资源上限**：单张 A100 80GB；不允许增加第二次 LLaMA backbone rollout；保持原 CCM 的单次 Transformer forward 主路径。

---

# 0. 最终决定

LLM 跨域实验固定为：

- **宿主**：CCM-merge，*Compressed Context Memory for Online Language Model Interaction*，ICLR 2024。
- **代码**：作者官方 `snu-mllab/Context-Memory`。
- **Backbone**：沿用论文与代码的 LLaMA-7B。
- **数据集**：DailyDialog，仅一个数据集。
- **数据划分**：DailyDialog 官方 `train / validation / test`，不自造 split。
- **宿主压缩预算**：沿用 CCM-merge 的 **2 个 COMP tokens**。
- **训练长度**：沿用官方 DailyDialog 配置 `max_steps=1000`。
- **LoRA**：沿用官方 `rank=8`、`q_proj/k_proj/v_proj/o_proj`。
- **主指标**：Perplexity（PPL，越低越好）。
- **机制验证**：同一 checkpoint 直接测试递归深度 `1 / 2 / 4 / 8 / 12`。
- **随机种子**：5 seeds，三种需要训练的配置使用 paired seeds。
- **LLM 侧不再做**：1Obs vs 2Obs、J_diag vs full、PCA、不同 λ、不同 Gamma rank、第二数据集等完整消融；这些由 TGN 主实验承担。

真正需要重新训练的只有三组：

1. `CCM-merge`
2. `CCM-merge + Γ, task-only`
3. `CCM-merge + Γ + Ours (2Obs)`

其余原论文方法只作为 reference 或 evaluation-only baseline。

---

# 1. 这个实验到底证明什么

不要把 LLM 实验写成“我们解决了 LLM 长上下文”。

它只承担一个跨域主张：

\[
\boxed{
\text{在相同固定 memory budget 下，}
\;
\text{显式学习 future-sufficient recursive state}
\;
\text{能否比普通递归压缩更好地保留深层历史信息？}
}
\]

CCM 的递归过程可抽象为：

\[
h_t=g_{\rm comp}(M_{t-1},c_t),
\]

\[
M_t=g_{\rm update}(M_{t-1},h_t).
\]

其中：

- \(h_t\)：当前 dialogue turn 经 CCM conditional LoRA 产生的 compressed K/V；
- \(M_{t-1}\)：此前已经压缩的固定容量 memory；
- \(M_t\)：更新后的固定容量 memory，后面继续参与下一次压缩。

原 CCM-merge 的更新是历史 COMP K/V 的 prefix arithmetic mean。代码中 `SUM` token 的 K/V 由 `sum_attn_mask @ key_states/value_states` 直接得到。

因此它天然存在：

\[
M_1\rightarrow M_2\rightarrow\cdots\rightarrow M_t,
\]

而我们研究的问题就是：

> 当前压出来的 \(M_t\) 是否不仅能支持当前 next-turn prediction，而且在下一次真实 memory update 之后仍保留后续任务真正需要的信息。

---

# 2. 为什么选 CCM-merge，而不是再找一个“有树”的 LLM

CCM-merge 的优势不是名字里有 memory，而是它同时满足：

1. **正式发表**：ICLR 2024。
2. **官方代码完整**。
3. **明确存在 recurrent compression**。
4. **固定 memory budget**，不是简单扩大 context。
5. **原 update 极简单**：arithmetic merge，因此替换位置干净。
6. **官方本身使用 conditional LoRA**，不需要大规模 full fine-tuning。
7. **DailyDialog 很轻**：作者官方 README 明确说明该任务可在单 RTX 3090 24GB 上训练；作者总体实验为单 A100 80GB。
8. **深度协议已存在**：官方代码直接构造 `1/2/4/8/12` context-step evaluation。

因此这里不是“为了跨领域硬找宿主”，而是：

\[
\boxed{
\text{CCM-merge 的固定容量递归 K/V update}
\;
\text{就是一个真实的 recursive compression bottleneck。}
}
\]

---

# 3. 原 CCM 官方代码行为：必须保持不变的部分

## 3.1 DailyDialog 训练样本如何构造

官方 `src/data/dialogue/data.py::sample_dialog()`：

```python
if random_k and instance['is_train']:
    k = np.random.randint(3, len(dialog) + 1)
    dialog = dialog[:k]

context = self._concat_dialog(dialog[:-2], ...)
context += dialog[-2] + self.sep_token
output = dialog[-1]
```

对随机前缀

\[
u_1,u_2,\ldots,u_k,
\]

官方语义是：

- 被压缩历史：\(u_1,\ldots,u_{k-2}\)；
- 当前输入：\(u_{k-1}\)；
- next-turn target：\(u_k\)。

因此原任务为：

\[
P(u_k\mid M_{k-2},u_{k-1}).
\]

**注意：DailyDialog 在这里不要写成“用户问题 / 助手回答”。**

数据文件只是连续 dialogue utterances。相邻 utterance 是 next-turn prediction 关系，但不应给它们擅自添加 user/assistant 语义。

---

## 3.2 每个历史 utterance 都已经产生 COMP state

`_concat_dialog(dialog[:-2])` 在 `online=True` 时对每个历史 utterance 添加 COMP token。

因此一次长度为 \(k\) 的官方训练 forward 内，已经存在：

\[
h_1,h_2,\ldots,h_{k-2},
\]

以及相应的 merged states：

\[
M_1,M_2,\ldots,M_{k-2}.
\]

这非常重要：

\[
\boxed{
\text{我们的 2Obs 不需要额外再跑一次 LLaMA。}
}
\]

同一个 forward 里已经同时有 \(M_{k-3}\) 和 \(M_{k-2}\)。

---

# 4. 我们到底替换哪里：只替换 `g_update`

**不要替换 conditional LoRA compressor。**

保留原：

\[
h_t=g_{\rm comp}^{\rm CCM}(M_{t-1},u_t),
\]

也就是：

- COMP token；
- conditional LoRA；
- 原 attention；
- 原 FFN；
- 原 positional logic；
- 原 causal / compression mask；

全部保持。

只替换：

\[
M_t=g_{\rm update}(M_{t-1},h_t).
\]

---

## 4.1 原 CCM-merge

代码 `src/arch/ccm_llama.py` 中，RoPE 之后、attention 之前：

```python
key_comp_avg   = torch.matmul(sum_attn_mask.unsqueeze(1), key_states)
value_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1), value_states)

key_states   = no_sum_mask * key_states   + key_comp_avg
value_states = no_sum_mask * value_states + value_comp_avg
```

`sum_attn_mask`：

- 只连接同 slot 的 COMP → SUM；
- 使用 lower-triangular mask；
- 按可见 COMP 数目归一化。

所以对第 \(t\) 个 memory step，本质是：

\[
M_t^{\rm CCM}
=
\frac{1}{t}\sum_{j=1}^t h_j,
\]

等价递归写成：

\[
M_t^{\rm CCM}
=
\frac{t-1}{t}M_{t-1}
+
\frac1t h_t.
\]

---

## 4.2 我们的 Γ

替换为：

\[
\boxed{
M_t=\Gamma_\theta(M_{t-1},h_t,t).
}
\]

第一版必须采用 **host-preserving zero-init residual**：

\[
\bar M_t
=
\frac{t-1}{t}M_{t-1}
+
\frac1t h_t,
\]

\[
\boxed{
M_t
=
\bar M_t
+
R_\theta(M_{t-1},h_t,t).
}
\]

并初始化：

\[
R_{\theta_0}=0.
\]

因此 step 0 必须满足：

\[
\boxed{
\Gamma_{\theta_0}
=
g_{\rm update}^{\rm CCM}
}
\]

到数值误差范围内。

这样不会因为随机初始化把宿主 memory 先破坏掉。

---

# 5. Γ 的具体参数规模

目标是 **tiny learned update**，不是给 LLM 再加一个大 memory network。

LLaMA-7B 每个 attention head 的 `head_dim≈128`。

建议：

- K / V 分开；
- 每个 Transformer layer 各一个小 residual module；
- 同一 layer 内 **所有 heads 和两个 COMP slots 共享**；
- low-rank hidden rank 第一版固定为 TGN / 总方案已经冻结的值；若必须指定工程默认，使用小 rank（如 8），但 **不在 LLM 上扫 rank**。

每个 head 的输入：

\[
[m_{t-1};h_t]\in\mathbb R^{256}.
\]

轻量 residual：

\[
R_\theta
=
U_\ell\,
\sigma\!\left(
V_\ell[
{\rm LN}(m_{t-1});
{\rm LN}(h_t);
e(t)
]
\right),
\]

最后一层 / 输出 scale zero-init。

参数目标：

\[
\boxed{
\text{Γ 参数量远小于 1M，且相对 7B backbone 可忽略。}
}
\]

不要：

- flatten 全部 KV 后接大 MLP；
- 每 head 独立网络；
- 每 COMP slot 独立网络；
- 再加 retrieval / long-term store；
- 再加额外 Transformer。

---

# 6. 最关键部分：2Obs 的严格定义

这是本方案最容易写错的地方。

## 6.1 选择 cut

对官方随机前缀：

\[
u_1,\ldots,u_k,
\]

仅当：

\[
k\ge4
\]

时启用我们的 2Obs 监督。

固定选择：

\[
\boxed{
v=k-3.
}
\]

也就是监督状态：

\[
M_v=M_{k-3},
\]

它只由：

\[
u_1,\ldots,u_{k-3}
\]

产生。

---

## 6.2 第一次 observation：一步 future test

从 \(M_{k-3}\) 出发，真实下一 continuation turn 是：

\[
u_{k-2}.
\]

其下一真实 utterance 是：

\[
u_{k-1}.
\]

定义：

\[
\boxed{
C_v^{(1)}=u_{k-2},
\qquad
Y_v^{(1)}=u_{k-1}.
}
\]

它对应：

\[
M_{k-3}
+
u_{k-2}
\longrightarrow
u_{k-1}.
\]

注意这里的 \(C^{(1)}\) 是 **future continuation context**，只在训练统计监督里出现；它绝不能作为 \(M_{k-3}\) 的生成输入。

---

## 6.3 第二次 observation：一次真实 memory update 后的 shifted test

在同一个官方 forward 中，\(u_{k-2}\) 本来就是历史的一部分并带 COMP token，因此产生：

\[
M_{k-2}
=
\Gamma_\theta(M_{k-3},h_{k-2},k-2).
\]

随后原 CCM 任务使用：

\[
u_{k-1}
\]

作为当前输入，预测：

\[
u_k.
\]

因此第二级定义为：

\[
\boxed{
C_v^{(2)}
=
(u_{k-2},u_{k-1};\text{one-update marker}),
\qquad
Y_v^{(2)}=u_k.
}
\]

其含义是：

\[
M_{k-3}
\xrightarrow[\text{真实 }u_{k-2}]{\Gamma}
M_{k-2}
+
u_{k-1}
\longrightarrow
u_k.
\]

第二级不是“随便再看远一点”，而是：

\[
\boxed{
\text{恰好经过一次真实递归 memory update 后，状态仍应对下一任务充分。}
}
\]

---

## 6.4 为什么 \(Y^{(1)}=u_{k-1}\) 同时出现在第二级 continuation 中不是泄漏

确实有：

\[
Y_v^{(1)}=u_{k-1},
\]

同时 \(u_{k-1}\) 是第二级预测 \(u_k\) 时的当前 input。

这是正常的序列结构：

\[
u_{k-2}
\rightarrow
u_{k-1}
\rightarrow
u_k.
\]

前一步真实发生的 utterance，在下一步自然变成已观察 context。

关键约束不是“不准重复出现”，而是：

\[
\boxed{
M_{k-3}\text{ 的生成路径绝不能读取 }u_{k-1}\text{ 或 }u_k.
}
\]

\(u_{k-1},u_k\) 只能进入训练期的 fixed future tests / task target。

---

# 7. 2Obs 不是两条 loss：监督统计量必须这样构造

最终方法里，future tests 应该堆入 **一个** 固定 \(p_v\)。

不要实现：

```text
loss_obs1 + loss_obs2 + closure_loss
```

也不要给两个 future 各自训练一个 predictor。

正确做法如下。

---

## 7.1 固定 utterance embedding

不额外跑 sentence encoder。

直接使用当前 LLaMA **冻结的 input embedding table** \(E_{\rm emb}\)。

对 utterance \(u\)：

\[
e(u)
=
\frac1{|u|}
\sum_{w\in u}
E_{\rm emb}(w).
\]

要求：

- embedding table 本身冻结；
- 不运行额外 LLaMA forward；
- 不使用未来生成 hidden state；
- 不训练 future encoder。

若维数太高，用固定 JL / CountSketch：

\[
\bar e(u)=S_{\rm emb}e(u),
\]

其中 \(S_{\rm emb}\) 全程固定、`requires_grad=False`。

---

## 7.2 第一级固定 test

定义固定 context feature：

\[
\chi_1
=
\chi\!\left(
e(u_{k-2}),\,
\text{horizon}=1
\right).
\]

固定 future feature：

\[
\phi_1
=
\varphi\!\left(e(u_{k-1})\right).
\]

然后：

\[
\boxed{
q_v^{(1)}
=
\operatorname{TensorSketch}
\left(
[1;\chi_1]\otimes\phi_1
\right).
}
\]

---

## 7.3 第二级固定 shifted test

第二级 continuation 必须显式包含：

- 先发生的真实 update turn \(u_{k-2}\)；
- update 后的当前 input \(u_{k-1}\)；
- `horizon=2 / one-update` 固定标识。

定义：

\[
\chi_2
=
\chi\!\left(
e(u_{k-2}),
e(u_{k-1}),
\text{horizon}=2,
\text{one-update}
\right).
\]

第二级 future：

\[
\phi_2
=
\varphi\!\left(e(u_k)\right).
\]

于是：

\[
\boxed{
q_v^{(2)}
=
\operatorname{TensorSketch}
\left(
[1;\chi_2]\otimes\phi_2
\right).
}
\]

---

## 7.4 最终只有一个 \(p_v\)

把两个 test 作为同一个 future-test bank 的两个 block：

\[
\boxed{
p_v
=
S_P
\begin{bmatrix}
\sqrt{\omega_1}\,q_v^{(1)}
\\
\sqrt{\omega_2}\,q_v^{(2)}
\end{bmatrix}.
}
\]

其中：

- \(S_P\) 为固定 identity / CountSketch/JL，只负责把最终维数整理到现有方法的 \(m\)；
- \(\omega_1,\omega_2\) **不在 LLM 上重新调**，直接继承 TGN 最终 2Obs 版本已经冻结的 horizon weighting；
- \(\chi,\varphi,\operatorname{TensorSketch},S_P\) 全部固定；
- \(p_v\) 不参与 inference。

因此：

\[
\boxed{
\text{2Obs = 一个 }p_v\text{ 里的“一步 test + 一次 update 后 shifted test”，}
}
\]

而不是第二个 closure loss。

如果现有通用代码仍保留 `Y_star` 槽位：

\[
\boxed{
\text{LLM 版本将 star mask 置 0；不要把 }Y^{(2)}\text{ 复制成 }Y^\star.
}
\]

本实验只做两次观测。

---

# 8. 实际压缩状态如何进入现有监督 loss

CCM 的真实 deployed memory \(M_t\) 很大：

- 多 Transformer layers；
- K / V；
- 多 heads；
- 2 个 COMP slots；
- 每 head 128 维。

不能为了 loss 再训练一个大 encoder。

沿用最终方法的 **post-bottleneck fixed loss lift**：

\[
\boxed{
b_t=J_{\rm mem}(M_t)\in\mathbb R^{r_B}.
}
\]

其中：

- \(J_{\rm mem}\) 必须放在真实 Γ update **之后**；
- 固定、不训练、不部署；
- 使用 structured CountSketch / JL；
- K / V、layer、head、COMP-slot 的索引进入固定 hash，使不同坐标有稳定身份；
- \(r_B\) 直接继承 TGN 最终方法配置，不在 LLM 重新扫。

这样 loss 只能读取已经实际进入递归的 CCM memory，不能在 bottleneck 前作弊。

---

# 9. 压缩前 witness：只允许看当前 update 时真实可用的信息

若当前最终 TGN loss 实现包含 pre-bottleneck witness \(u_v=[b_v;\mathrm{sg}(s_v)]\)，LLM 必须严格按同一接口实现。

对 memory update \(t\)：

\[
I_t
=
(M_{t-1},h_t,t).
\]

定义固定 witness：

\[
s_t
=
\rho_{\rm fix}(I_t),
\]

其中 \(\rho_{\rm fix}\)：

- 固定 CountSketch / JL；
- 不训练；
- 不传播；
- 不部署。

然后：

\[
\boxed{
u_t
=
[b_t;\operatorname{sg}(s_t)].
}
\]

**严禁**将以下内容放入 \(I_t\)：

- \(u_{t+1}\)；
- 未来 target；
- \(p_v\)；
- future embedding；
- 未来生成 hidden state。

这样“压缩前可用信息 vs 压缩后 memory”的比较才没有泄漏。

> 具体 Ky-Fan / whitening / covariance / profile 实现不在 LLM 中重新推导，直接复用已经在 TGN 最终代码里冻结的同一 loss module。LLM 只提供 `B/U/P/weights` 所需的宿主适配输入。

---

# 10. \(k=3\) 样本如何处理

官方训练允许：

\[
k=3.
\]

此时：

\[
u_1,u_2,u_3
\]

只有：

- 历史 \(u_1\)；
- 当前 \(u_2\)；
- target \(u_3\)。

没有合法的：

\[
M_{k-3}=M_0
\]

对应的非空、两级监督 cut。

因此：

\[
\boxed{
k=3\text{ 样本保留原 CCM task CE，但 }w_{\rm RPBE}=0.
}
\]

不要：

- 丢掉该训练样本；
- 改官方 `random_k` 分布；
- 用空 memory 人造一个 2Obs；
- 把同一个 future 重复两次凑 2Obs。

---

# 11. 主任务 loss

原 CCM next-turn language modeling loss完全保持：

\[
\mathcal L_{\rm task}
=
-\log P(u_k\mid M_{k-2},u_{k-1}).
\]

LLM 最终训练：

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm task}
+
\lambda\,
\mathcal L_{\rm Ours}^{\rm 2Obs}.
}
\]

其中：

- `CCM-merge`：只有 \(\mathcal L_{\rm task}\)，无 Γ；
- `CCM-merge + Γ, task-only`：有 Γ，但 \(\lambda=0\)；
- `CCM-merge + Γ + Ours`：相同 Γ + 最终 2Obs loss。

\(\lambda\)、ridge、whitening / Ky-Fan 版本、\(m,r_B\) 等：

\[
\boxed{
\text{直接读取 TGN 最终冻结配置；LLM 不单独 sweep。}
}
\]

这样 LLM 实验不是第二套算法调参。

---

# 12. 为什么 `Γ task-only` 这个 control 不能省

如果只比较：

\[
\text{CCM-merge}
\quad\text{vs}\quad
\text{CCM-merge + Γ + Ours},
\]

评审可以合理质疑：

> 提升只是把 arithmetic mean 换成了可学习网络。

因此必须比较：

\[
\boxed{
\text{CCM-merge + Γ, task-only}
}
\]

和：

\[
\boxed{
\text{CCM-merge + Γ + Ours}.
}
\]

这两者：

- Γ 参数完全相同；
- conditional LoRA 参数完全相同；
- initialization 完全相同；
- training steps 完全相同；
- batch 顺序 paired；
- 唯一差别就是我们的未来充分性 loss。

所以：

\[
\boxed{
\text{Ours}>\text{Γ task-only}
}
\]

才能说明收益不是单纯的容量 / learned merge。

这是 LLM 唯一必须保留的 architecture control；其余完整消融由 TGN 做。

---

# 13. 哪些参数一起训练

沿用 CCM Step 2。

统一使用同一个官方 Step-1 foundation / `llama-7b-no`。

三组 Step-2：

## A. CCM-merge

训练：

- conditional compression LoRA；

不训练 Γ。

## B. CCM-merge + Γ, task-only

训练：

- 与 A 完全相同的 conditional compression LoRA；
- Γ。

loss：

\[
\mathcal L_{\rm task}.
\]

## C. CCM-merge + Γ + Ours

训练：

- 与 B 完全相同的 conditional compression LoRA；
- Γ。

loss：

\[
\mathcal L_{\rm task}
+
\lambda\mathcal L_{\rm Ours}.
\]

LLaMA base weights保持冻结。

这样不是多阶段 teacher-fitting，也不是冻结一个未来教师；只是沿用原 CCM LoRA compression training，并在同一阶段联合训练我们的 Γ。

---

# 14. 必须保持“一个 LLaMA forward”

原 CCM 的关键效率来源是把 recursive compression 放入一次 Transformer forward。

我们的 nonlinear Γ 不能再用一条矩阵乘法算 prefix mean，但：

\[
T\le12.
\]

所以正确实现为：

1. 每个 Transformer layer 正常一次计算所有 Q/K/V；
2. 在原 `key_comp_avg / value_comp_avg` 位置，取当前 layer 各 COMP K/V；
3. 对最多 12 个 memory steps 执行 tiny recurrent scan：

\[
M_1=\Gamma(M_0,h_1),
\]

\[
M_2=\Gamma(M_1,h_2),
\ldots
\]

4. 将得到的 \(M_t\) 写回对应 SUM-token K/V；
5. 后续 attention / FFN 正常继续。

**不允许：**

```text
for memory_step:
    llama(...)
```

也不允许为了第二级 future 再跑一个 shifted LLaMA forward。

验收标准：

\[
\boxed{
\text{每个训练 batch 的 backbone forward count = 1.
}
}
\]

---

# 15. 数据 split：必须修正官方代码的一个不严谨点

官方 `DialogueDataset` 当前代码：

```python
self.valset['dialog'] = validation
self.valset['dialog'] += test
```

即将 DailyDialog validation 与 test 合并后构造 evaluation pool。

为了正式论文：

\[
\boxed{
\text{primary protocol 必须恢复官方 train / validation / test 三个 split。}
}
\]

规定：

- **train**：只用于训练；
- **validation**：只用于运行健康检查，不用于 LLM-specific hyperparameter sweep；
- **test**：只在最终 checkpoint 上报告主结果。

训练仍固定：

\[
1000\text{ steps}.
\]

不使用 test 选择 checkpoint。

建议直接报告 step-1000 final checkpoint，而不是为了 LLM 再做一轮 early-stop tuning。

---

# 16. 与原 CCM 论文如何对比

因为原官方代码将 validation+test 合并，原论文 / 官方现成结果不能直接与我们的 clean test 数字混在同一统计比较里。

最终报告分两层。

---

## 16.1 Primary controlled table：我们真正比较的表

全部使用 clean official test split：

| Method | Γ | Ours | Fixed memory | PPL ↓ |
|---|---:|---:|---:|---:|
| CCM-merge | ✗ | ✗ | ✓ | 5-seed |
| CCM-merge + Γ, task-only | ✓ | ✗ | ✓ | 5-seed |
| **CCM-merge + Γ + Ours** | ✓ | ✓ | ✓ | **5-seed** |

这是论文真正用于支持我们方法的 LLM 对比。

---

## 16.2 Evaluation-only reference：几乎不增加训练成本

在 clean test split 上，用官方 released checkpoint / inference path额外评估：

- No Context；
- Full Context；
- CCM-concat；
- released CCM-merge。

这些只是帮助读者理解：

- 不用历史的下界；
- 完整历史的参考上界；
- growing-memory CCM-concat；
- 官方 released CCM-merge。

不需要重新做 5-seed training。

---

## 16.3 原论文其他 baseline

例如：

- Gisting-online；
- Compressive；

若时间紧，不重新训练。

可以在单独的 **“Reported in CCM (paper protocol)”** 表中引用原论文数字，但必须注明：

> paper protocol / evaluation pool 与我们的 clean official-test primary protocol 不完全相同，因此不能把这些数字当成 paired controlled comparison。

不要把 published pooled number 和我们的 clean-test number放在一列直接宣称胜负。

---

# 17. 递归深度实验：LLM 唯一额外机制验证

同一个最终 checkpoint，直接评估：

\[
T\in\{1,2,4,8,12\}.
\]

不重新训练 5 个模型。

画：

\[
\text{PPL}(T).
\]

最关键比较：

\[
\Delta(T)
=
{\rm PPL}_{\Gamma\text{-task}}
-
{\rm PPL}_{\rm Ours}.
\]

如果方法真的缓解 repeated compression 的信息损失，最有解释力的现象应该是：

\[
\boxed{
\Delta(1)\text{ 小，}
\qquad
\Delta(T)\text{ 随递归深度增加而总体变大。}
}
\]

不要求严格每一个点完全单调，但整体趋势必须与“深层递归信息保持”一致。

如果只在 \(T=1\) 提升，而 \(T=8/12\) 没有优势，就不能用该结果强声称 recursive preservation。

---

# 18. 统计报告：完整但不冗余

三组训练配置均：

\[
5\text{ paired seeds}.
\]

报告：

- test PPL：mean ± std；
- 每个 seed 的 paired PPL；
- `Ours - Γ task-only` 的 paired difference；
- depth curve 同样按 5 seeds 汇总。

LLM 是跨域辅助实验，不需要再堆很多正式统计检验。

不要仅用一条 best seed。

---

# 19. 效率报告：只保留 4 个量

LLM 表里额外报告：

1. trainable parameters；
2. peak GPU memory；
3. training wall-clock / step 或总 training wall-clock；
4. inference memory budget（COMP tokens / KV memory 不变）。

关键 claim：

\[
\boxed{
\text{Ours 不增加 COMP-token memory budget。}
}
\]

也就是说我们不是靠“存更多历史”获胜。

---

# 20. 算力与时间预算

官方 README 给出的事实：

- 作者总体实验基本在单 A100 80GB 上运行；
- 总体任务约 5–24h；
- DailyDialog 因 context 更短，官方明确说可以在单 RTX 3090 24GB 上训练。

因此本实验按 **单 A100** 设计，不需要多卡。

真正的大训练数：

\[
3\text{ methods}\times5\text{ seeds}=15\text{ runs}.
\]

每个：

\[
1000\text{ steps}.
\]

工程上不预先承诺一个未经实测的绝对小时数，但设定两个硬标准：

\[
\boxed{
\text{Ours step time 目标 }\le1.5\times\text{ CCM-merge};
}
\]

如果：

\[
\boxed{
\text{Ours 明显}>2\times\text{ CCM-merge},
}
\]

优先检查是否：

- 意外做了第二次 LLaMA forward；
- Gamma scan 写成逐 step backbone recurrence；
- future feature 又跑了 encoder；
- RPBE 统计跨 batch 保留了大计算图。

不能直接接受这种开销。

建议执行顺序：

1. official checkpoint eval；
2. 1 seed 三配置；
3. 检查速度 / 显存 / loss；
4. 正常后并行或顺序补齐剩余 4 seeds。

---

# 21. 严格禁止的实现

## 数据与监督

禁止：

- 把 DailyDialog utterance擅自标成固定“user / assistant”角色；
- 让 \(M_{k-3}\) 读取 \(u_{k-1}\) 或 \(u_k\)；
- future decoder；
- 生成假 future；
- 第二个 LLM rollout；
- 用 validation/test future 更新训练统计；
- k=3 时伪造 2Obs。

## 方法

禁止：

- 两条独立 Obs loss；
- 额外 closure loss；
- LLM-specific λ sweep；
- LLM-specific Gamma-rank sweep；
- 重做 TGN 已经承担的完整 ablation；
- 增加 retrieval / long-term store 来帮我们；
- 增加 COMP token 数。

## 性能

禁止：

- 每个 memory step 重新跑一次 LLaMA；
- 跨大量 batch 保存带图的 future/Ky-Fan moment；
- 为 future text 运行第二个 encoder。

---

# 22. 必须通过的实现验收测试

在跑 5 seeds 前必须全部通过。

## Test A — Host equivalence

Gamma residual 全 zero 时：

\[
\Gamma_{\theta_0}
=
\text{CCM arithmetic merge}.
\]

固定输入下比较：

- SUM K；
- SUM V；
- logits / loss；

应与原 CCM-merge 在合理浮点误差内一致。

---

## Test B — One-forward guarantee

一个 training batch：

```text
backbone_forward_count == 1
```

2Obs 不得导致第二次 forward。

---

## Test C — Future isolation

固定过去：

\[
u_1,\ldots,u_{k-3},
\]

人为替换：

\[
u_{k-1},u_k.
\]

要求：

- \(M_{k-3}\) 完全不变；
- pre-bottleneck witness 完全不变；
- 只有 \(p_v\) / task future target 改变。

这是一条非常重要的 no-leakage unit test。

---

## Test D — 2Obs indexing

对构造样本逐项 assert：

```text
cut_state     = M[k-3]
obs1_context  = u[k-2]
obs1_future   = u[k-1]
obs2_context  = (u[k-2], u[k-1], one_update)
obs2_future   = u[k]
```

严禁旧版本：

```text
obs2_future = u[k+1] / u[k+2]
```

也严禁把两个 observation 分成两条 loss。

---

## Test E — k=3 mask

当 `k == 3`：

```text
task_loss_active = True
rpbe_weight = 0
```

---

## Test F — Fixed statistic determinism

相同 utterances 下，多次构造：

\[
p_v
\]

必须 bitwise / 数值确定一致。

\(\chi,\varphi,J,S_P\) 不应出现在 optimizer 参数中。

---

## Test G — Paired initialization

同 seed 的：

- `Γ task-only`
- `Γ + Ours`

在训练开始前：

- conditional LoRA hash 相同；
- Γ hash 相同；
- batch order 相同。

唯一差异是 `lambda_ours=0` vs `lambda_ours>0`。

---

## Test H — Inference isolation

validation/test 推理中：

- 不构造 future \(p_v\)；
- 不读取 future utterance 来更新 memory；
- 不更新 whitening / covariance / Ky-Fan statistics；
- 只保留 LLaMA + conditional LoRA + Γ。

---

# 23. 最终实验表格模板

## 23.1 Primary clean-test table

| Method | Fixed memory | Trainable merge | 2Obs supervision | Test PPL ↓ | Params | Peak VRAM | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCM-merge | ✓ | ✗ | ✗ | mean±std | — | — | — |
| CCM-merge + Γ | ✓ | ✓ | ✗ | mean±std | — | — | — |
| **CCM-merge + Γ + Ours** | **✓** | **✓** | **✓** | **mean±std** | — | — | — |

## 23.2 Depth curve

| Depth | CCM-merge | Γ task-only | Ours |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 12 | — | — | — |

真正关注：

\[
{\rm PPL}_{\Gamma\text{-task}}(T)
-
{\rm PPL}_{\rm Ours}(T).
\]

---

# 24. 这组实验最终可以支持什么

如果结果成立：

1. Ours 显著优于 CCM-merge；
2. Ours 稳定优于 **同参数量 Γ task-only**；
3. memory budget 完全相同；
4. 优势在深层 recursion 更明显；

那么 LLM 实验可以支持：

\[
\boxed{
\text{收益不是来自更大的 memory，也不是仅来自 learned merge；
而是来自对 fixed-budget recursive state 的 future-sufficiency supervision。}
}
\]

这和 TGN 的角色分工非常清楚：

- **TGN**：负责完整方法学消融、loss 组件、1Obs/2Obs、balancing 等证明；
- **LLM/CCM**：只负责证明同一最终插件在完全不同的 recursive memory host 上仍成立。

---

# 25. 这组实验不能声称什么

即使结果很好，也不要直接声称：

- “解决所有 LLM long-term memory”；
- “优于所有 retrieval memory”；
- “证明 dialogue 会一直集中到同一 topic”；
- “完整保存了历史信息”；
- “学到了无损记忆”。

准确说法只能是：

> 在 CCM 的固定容量递归 K/V memory 中，我们的 future-conditioned supervision 使 learned update 在相同 memory budget 下更好地保持后续 next-turn prediction 所需的信息，并且该优势随递归深度增加而更明显。

---

# 26. 给 coding AI 的最终任务顺序

## Task L0 — Fork 与 protocol 清理

- 基于官方 CCM repo；
- 保留原代码可复现分支；
- DailyDialog 拆回 official train / validation / test；
- 不修改 tokenizer / preprocessing / random-prefix sampling；
- 固定 Step-1 foundation。

## Task L1 — Official host reproduction

确认：

- official CCM-merge 可运行；
- clean test PPL 可得到；
- depth `1/2/4/8/12` evaluation 正常；
- released CCM-concat / full-context reference 可评估。

## Task L2 — Γ 接入

只改 `src/arch/ccm_llama.py` 原 `key_comp_avg/value_comp_avg` 的 merge 位置。

要求：

- zero-init 时数值复现原 arithmetic merge；
- K / V 都更新；
- shape 不变；
- COMP / SUM 数不变；
- 单 backbone forward。

## Task L3 — Host state adapter

实现：

\[
b_t=J_{\rm mem}(M_t)
\]

以及若现有最终 loss 需要：

\[
u_t=[b_t;\mathrm{sg}(\rho_{\rm fix}(M_{t-1},h_t,t))].
\]

所有固定映射放独立模块，明确：

```python
requires_grad = False
```

## Task L4 — 2Obs record

仅对 `k>=4`：

```text
v             = k - 3
C1            = u[k-2]
Y1            = u[k-1]
C2            = (u[k-2], u[k-1], one_update)
Y2            = u[k]
```

生成：

```text
q1
q2
p = stack_and_fixed_sketch(q1, q2)
```

只产生 **一个** supervision row。

## Task L5 — Loss integration

复用 TGN 已冻结最终 loss module。

不要在 CCM repo 内再写一套“近似版本”的 loss。

总目标：

```text
task_only: CE
ours:      CE + lambda * FINAL_TGN_LOSS
```

## Task L6 — Unit tests

必须通过第 22 节 A–H。

## Task L7 — 1-seed pilot

只跑：

1. CCM
2. Γ task-only
3. Ours

检查：

- PPL；
- Gamma 是否离开 zero-init；
- auxiliary loss 是否真的生效；
- peak VRAM；
- step time；
- backbone forward count。

不做超参数搜索。

## Task L8 — 5-seed 正式实验

pilot 健康后补齐 paired 5 seeds。

## Task L9 — Final evaluation

- clean official test；
- depth curve；
- efficiency；
- 输出 per-seed JSON/CSV；
- 不再根据 test 修改超参数。

---

# 27. 最终一句话

\[
\boxed{
\text{CCM 负责把每个 dialogue turn 编成 compressed K/V；
Γ 只负责固定容量 memory 的递归合并；
我们的 2Obs test 在同一个 cut 上同时检查“下一步可用”和“再经历一次真实 update 后仍可用”，
并把两者堆入一个固定 future statistic }p_v\text{ 来训练，而不是增加第二个 future loss。}
}
\]

这是本 LLM 实验需要保持不变的核心定义。

---

# 28. 已核对的官方来源

1. **Kim et al., “Compressed Context Memory for Online Language Model Interaction”, ICLR 2024**
   - 论文：ICLR 2024 proceedings。
2. **官方实现**
   - Repository: `snu-mllab/Context-Memory`
3. **DailyDialog 样本索引**
   - `src/data/dialogue/data.py`
   - `sample_dialog()`：随机 `k>=3`；`dialog[:-2]` 为压缩历史，`dialog[-2]` 为当前输入，`dialog[-1]` 为 target。
4. **CCM-merge arithmetic K/V merge**
   - `src/arch/ccm_llama.py`
   - `sum_attn_mask @ key_states/value_states`；
   - `sum_attn_mask` 使用 lower triangular mask 并按历史 COMP 数量归一化。
5. **DailyDialog LLaMA-7B 官方配置**
   - `src/config/dialog/llama-7b.yaml`
   - FP16；
   - train batch 1；
   - gradient accumulation 128；
   - LR \(3\times10^{-4}\)；
   - LoRA rank 8；
   - q/k/v/o projection；
   - max steps 1000。
6. **官方算力说明**
   - README：总体实验单 A100 80GB；DailyDialog 可在单 RTX 3090 24GB 上训练。
7. **需要修正的官方 evaluation split 代码**
   - `src/data/dialogue/data.py` 当前将 validation 与 test 合并为 `valset`；正式主实验必须拆回官方 split。

