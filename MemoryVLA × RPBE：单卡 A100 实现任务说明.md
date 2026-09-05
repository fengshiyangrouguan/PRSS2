# MemoryVLA × RPBE：单卡 A100 实现任务说明

## 1. 已确定的实验口径

- 宿主：**MemoryVLA，`openvla-codebase` 分支**
- 数据集：**LIBERO-Mem 一个数据集，包含 10 个 manipulation tasks**
- 数据划分：**完全使用 LIBERO-Mem 官方 train / validation 协议，不自行切分**
- 重复实验：**5 seeds**
- 资源：**1×A100 80GB**
- 大模型微调：**LoRA**
- MemoryVLA 的普通宿主设置尽量保持官方：
  - cognitive width = 4096
  - `mem_length=16`
  - `future_action_window_size=15`
  - DiT-L
  - cosine-adjacent ToMe selector
- 外部方法尽量直接引用 LIBERO-Mem 已发表实验结果；我们自己必须重跑的是同宿主内部对照。
- 第一版插件只修改 **cognitive memory consolidation**。Perceptual bank 暂时保持官方实现，不同时改两个 memory branch。

---

# 2. 为什么单张 A100 + LoRA 可行

MemoryVLA 官方完整训练脚本是 8×A100 full fine-tuning，但它建立在 OpenVLA/Prismatic 7B backbone 上。

OpenVLA 官方已经提供单张 A100-80GB LoRA 微调方案，默认 rank 32、`all-linear`，并明确把 LoRA 推荐给不足以 full-finetune 7B 模型的场景。

但 **MemoryVLA 仓库本身没有 LoRA 接口**。

因此不能直接运行 OpenVLA 的 `vla-scripts/finetune.py`。

MemoryVLA 代码里：

```text
vla.vlm.llm_backbone.llm
```

本身就是 HuggingFace `PreTrainedModel`，因此正确做法是在这里挂 PEFT：

```text
MemoryVLA
  └── vlm
       └── llm_backbone
            └── llm   ← 在这里挂 LoRA
```

建议：

- vision backbone：冻结
- multimodal base weights：冻结
- LLM：LoRA rank 32
- MemoryVLA 自己的：
  - cognitive retrieval
  - cognitive gate
  - perceptual branch
  - action DiT
  保持正常 trainable
- 新增 \(\Gamma_\theta\)：trainable

不要量化作为第一方案。OpenVLA 官方也提示 quantized LoRA 可能降低表现。

### 重要实现注意

不要：

```text
挂完 LoRA
→ 再调用原来的 freeze_backbones()
```

否则有可能把 LoRA 参数一起冻结。

正确顺序应是：

```text
load MemoryVLA checkpoint
→ freeze base vision / base LLM
→ attach LoRA
→ 显式检查 requires_grad
→ 加入 MemoryVLA modules + Gamma optimizer
```

单卡不需要硬套当前 `train.py` 的 8-GPU FSDP 逻辑。

建议新增：

```text
train_libero_mem_rpbe.py
```

复用 MemoryVLA 的 model、dataset、collator、checkpoint loader，但使用普通单 GPU BF16 + AdamW + gradient accumulation。

不要大改官方 `train.py`。

---

# 3. 真正要修改的宿主位置

核心文件：

```text
vla/memory_vla.py
```

真正的 consolidation 在：

```python
CogMemBank._consolidate_with_token_merge()
```

官方现在：

```python
@torch.no_grad()
def _consolidate_with_token_merge(...):
    ...
    idx_max = argmax(adjacent cosine similarities)

    timestep_i, feat_i = ...
    timestep_j, feat_j = ...

    fused_feat = 0.5 * (feat_i + feat_j)

    bank[idx_max] = (
        timestep_i,
        fused_feat.detach().clone()
    )
```

我们的 selector **完全不改**：

\[
(i^\*,i^\*+1)
=
\arg\max_i
\cos(m_i,m_{i+1}).
\]

只替换：

\[
\boxed{
\frac{m_i+m_j}{2}
\longrightarrow
\Gamma_\theta(m_i,m_j)
}
\]

即：

```text
官方：
fused_feat = 0.5 * (feat_i + feat_j)

我们：
fused_feat = gamma(feat_i, feat_j)
```

仍然写回同样的 bank slot。

---

# 4. 第一版只修改 CogMemBank

MemoryVLA 实际有：

```text
cog_mem_bank
per_mem_bank
```

而 `PerMemBank` 直接继承 `CogMemBank`。

因此不能粗暴修改父类后让两个 bank 自动同时使用 \(\Gamma\)。

必须显式区分：

```text
cognitive consolidation:
    RPBE Gamma

perceptual consolidation:
    official average
```

第一阶段只研究：

\[
\boxed{
m_a,m_b\in\mathbb R^{4096}
\rightarrow
z_v\in\mathbb R^{4096}
}
\]

这样最能隔离我们的插件贡献。

---

# 5. \(\Gamma_\theta\) 不要做巨大 8192→4096 MLP

采用 residual low-rank merger：

\[
m_{\rm avg}
=
\frac12(m_a+m_b),
\]

\[
h
=
f_\theta(Pm_a,Pm_b),
\]

\[
\Delta m=Uh,
\]

最终：

\[
\boxed{
\Gamma_\theta(m_a,m_b)
=
m_{\rm avg}
+
\alpha Uh
}
\]

其中：

\[
r\ll4096.
\]

初始：

\[
\alpha\approx0.
\]

因此训练开始时：

\[
\Gamma_\theta\approx\text{official AvgMerge}.
\]

这保证插件不会一上来破坏 MemoryVLA 已训练好的 latent geometry。

---

# 6. 不再像 TGN 一样“查询树”

这是 embodied 版本和当前 PRSS2 最大的结构变化。

TGN 当前：

```text
query
→ recursive tree
→ trace internal nodes
→ CutCandidate
→ future lookup
```

MemoryVLA 不需要这套。

它的树是在时间运行过程中**自然长出来的**。

例如：

\[
m_1,m_2\rightarrow z_{12},
\]

之后：

\[
z_{12},m_3\rightarrow z_{123}.
\]

这已经天然形成：

```text
m1   m2   m3
 \   /
  z12
    \   /
     z123
```

所以：

\[
\boxed{
\text{一次真实 consolidation = 一个 RPBE cut}
}
\]

绝对不要：

- 再遍历整棵树；
- 再查询 memory；
- 为了 RPBE 重建完整 recursive tree。

---

# 7. 每个 memory entry 只增加轻量 provenance

官方 bank 当前是：

```python
(timestep, feat)
```

需要改成轻量结构，至少保存：

```text
state
node_id
start_step
end_step
depth
```

merge：

\[
a+b\rightarrow v
\]

时：

\[
start_v=start_a,
\]

\[
end_v=end_b,
\]

\[
depth_v=1+\max(depth_a,depth_b).
\]

另记录：

```text
left_id
right_id
```

即可事后恢复 merge tree。

**训练时不 materialize tree。**

这些字段主要用于：

- 区分 merge identity；
- 递归深度分析；
- provenance audit；
- 防止重复统计。

---

# 8. 一个非常重要的训练时序问题：不能直接用现有 StreamRLDSDataset

MemoryVLA 已经实现：

```text
vla/datasets/datasets.py
    StreamRLDSDataset
```

它确实会按照 episode 顺序：

\[
1,2,3,\ldots,T
\]

逐 transition 输出。

但是这里存在一个关键的不一致：

### 训练 Stream

会对每个 RLDS transition 更新一次 memory。

### 实际 MemoryVLA inference

`predict_action()` 一次输出：

\[
16\times7
\]

action chunk。

官方 LIBERO evaluator 会把这一整段 action 执行完以后，才再次调用 policy。

也就是说 inference memory timestep 是：

\[
\boxed{\text{policy decision}}
\]

而不是：

\[
\text{raw simulator frame}.
\]

如果直接用现成 `StreamRLDSDataset`：

> 训练时每帧 merge，推理时每 16 步才 merge。

这会严重改变 consolidation distribution。

---

# 9. 因此必须新增 DecisionStream

基于官方：

```text
StreamRLDSDataset
```

增加：

```text
DecisionStreamRLDSDataset
```

不是重新写 RLDS pipeline。

一条 trajectory：

\[
0,1,2,\ldots,T
\]

按官方 action horizon：

\[
K
=
future\_action\_window\_size+1
=
16
\]

输出：

\[
0,16,32,48,\ldots
\]

每个 decision \(k\) 的监督本身已经是：

\[
A_k^\*
=
[a_k,\ldots,a_{k+15}].
\]

因此：

```text
decision 0:
observation_0 → actions 0:15

decision 1:
observation_16 → actions 16:31

decision 2:
observation_32 → actions 32:47
```

这才和真实 evaluation 一致。

不要硬编码 16，代码写成：

```text
decision_stride =
future_action_window_size + 1
```

---

# 10. Chronological Merge Sampler：真正的统计样本怎么产生

这是第一个核心新模块。

正常运行：

\[
D_1\rightarrow D_2\rightarrow\cdots
\]

其中一个 \(D_k\) 就是一次 policy decision。

Memory bank 前 16 个 entries 正常写入。

当：

\[
|M|=17
\]

宿主 selector 选：

\[
m_a,m_b.
\]

执行：

\[
z_v=\Gamma(m_a,m_b).
\]

此时立即创建：

```text
MergeRecord
```

至少记录：

```text
episode_id
merge_id
merge_decision_time

left_state_detached
right_state_detached
merged_state_detached

left_id
right_id
depth
start_step
end_step
```

这就是一个 cut。

---

# 11. 但 merge 刚发生时不能立即计算 RPBE

假设：

\[
\tau_v=20.
\]

此时未来还没有发生。

建立：

```text
PendingMerge(v)
```

然后继续：

\[
21\rightarrow22\rightarrow23...
\]

下一次 policy decision：

\[
s_1=21
\]

给它填：

\[
F_{v,1}.
\]

再下一次：

\[
s_2=22
\]

填：

\[
F_{v,2}.
\]

两级 future 都成熟后，这个 merge 才成为完整 RPBE sample。

因此：

\[
\boxed{
\text{future_time}>\text{merge_time}
}
\]

由代码结构天然保证。

不需要 TGN 当前的 `JodieFutureIndex`。

---

# 12. 两级 RPBE rows

每个 merge \(v\) 最终产生：

\[
(z_v,P_{v,1})
\]

和：

\[
(z_v,P_{v,2}).
\]

即仍然完全保持 PRSS2 当前：

\[
\boxed{2\text{-Observation}}
\]

结构。

不能平均：

\[
P_{v,1},P_{v,2}.
\]

两个 horizon 是两个 row，但共享同一个：

\[
cut\_id=v.
\]

这点继续沿用当前 `CutRecord` / cluster-weight 语义。

---

# 13. Future outcome \(Y\)

定义为真实 demonstration 的未来 action chunk：

\[
\boxed{
Y_{v,h}
=
A_{s_h}^{*}
}
\]

其中：

\[
A_{s_h}^{*}
\in
\mathbb R^{16\times7}.
\]

flatten：

\[
112D.
\]

绝对不用：

- subgoal label；
- object ID；
- mask；
- success label。

这些 benchmark annotation 都不能进入训练。

---

# 14. Future context \(C\)：这里必须特别注意 LoRA

之前容易犯一个错误：

> 直接拿未来 cognitive token 当 fixed context。

现在不能这么干。

因为我们要训练 LoRA：

\[
c_s^{cog}
\]

会随 LoRA 参数变化。

那：

\[
P=\psi(C,Y)
\]

就不再是固定测量，违反当前 RPBE 的 fixed-map 设定。

因此未来 context 必须只来自**固定信息**。

推荐：

\[
C_{v,h}
=
(
\chi_{\rm vision}(O_{s_h}),
\chi_{\rm instruction}(L),
\Delta s,
h
).
\]

其中：

### Future observation

使用**冻结 vision backbone**的 raw DINO/SigLIP feature。

不要使用：

- LoRA 后 cognitive token；
- memory-fused token；
- trainable `per_compr` 输出。

raw frozen vision feature 再经过固定 projection。

### Instruction

LIBERO-Mem instruction 本来就是 policy 可见输入。

使用：

- fixed categorical/hash signature；
- 或冻结 text representation。

不要用 trainable LoRA 输出。

### Temporal metadata

固定：

\[
\Delta s=s_h-\tau_v,
\]

以及：

\[
h\in\{1,2\}.
\]

这样：

\[
\boxed{
P_{v,h}
\text{ 在整个训练过程中不随模型参数变化}
}
\]

---

# 15. `maps.py` 需要改什么

当前 PRSS2：

```text
future outcome = binary 0/1
```

MemoryVLA：

\[
Y\in\mathbb R^{112}.
\]

因此保留现有 fixed-map 哲学，但换 feature map。

例如：

\[
\phi_Y(Y)
=
\text{fixed RFF}(Y)
\]

或固定 random feature。

Context：

\[
\phi_C(C).
\]

最终继续沿用当前：

\[
\boxed{
P
=
\operatorname{CountSketch}
(
[1;\phi_C]\otimes\phi_Y
)
}
\]

不要改成简单 concat。

---

# 16. 第二个核心新模块：4096D High-Dimensional RPBE Engine

宿主真正的 state：

\[
z_v\in\mathbb R^{4096}
\]

绝对不能为了统计方便降成 128 维。

问题只在于当前 `loss.py` 的：

```text
WeightedWelford
```

会建立：

\[
M2_{ZZ}\in\mathbb R^{4096\times4096}.
\]

这个实现不能直接用。

---

# 17. Full balancing：改成 exact sample-space dual

一个 RPBE window 有 \(N\) 个 rows：

\[
Z\in\mathbb R^{N\times4096},
\]

\[
P\in\mathbb R^{N\times m}.
\]

按照当前 PRSS2 的：

- row weights；
- cut clustering；
- centering；
- degrees of freedom \(D\)；

先构造：

\[
X
=
W^{1/2}(Z-\mu_Z),
\]

\[
Q
=
W^{1/2}(P-\mu_P).
\]

当前 feature-space covariance 是：

\[
C_{ZZ}
=
X^\top X/D.
\]

不要构造它。

---

# 18. 必须保持和当前 scale-normalized objective 完全等价

当前 full-balancing 先计算：

\[
s_Z
=
\operatorname{mean}
\operatorname{diag}(C_{ZZ}),
\]

\[
s_P
=
\operatorname{mean}
\operatorname{diag}(C_{PP}).
\]

定义：

\[
\tilde X
=
\frac{X}{\sqrt{Ds_Z}},
\]

\[
\tilde Q
=
\frac{Q}{\sqrt{Ds_P}}.
\]

那么原来的：

\[
A
=
C_{ZZ}/s_Z+\epsilon I
\]

等价于：

\[
A
=
\tilde X^\top\tilde X+\epsilon I.
\]

所以直接构造 sample Gram：

\[
\boxed{
K_Z
=
\tilde X\tilde X^\top
}
\]

\[
\boxed{
K_P
=
\tilde Q\tilde Q^\top.
}
\]

都是：

\[
N\times N.
\]

---

# 19. Exact dual score

计算：

\[
H_Z
=
K_Z(K_Z+\epsilon I)^{-1},
\]

\[
H_P
=
K_P(K_P+\epsilon I)^{-1}.
\]

然后：

\[
\boxed{
J_{\rm dual}
=
\operatorname{tr}(H_ZH_P)
}
\]

它对应原来的：

\[
\left\|
A^{-1/2}
C
B^{-1/2}
\right\|_F^2.
\]

实现时不要显式 inverse：

```text
Cholesky / solve
```

即可。

这不是近似降维：

\[
\boxed{
4096D representation 完整参与统计，
只是把计算从 feature space 换到 sample space。
}
\]

---

# 20. 必须做一个小维 exact-equivalence 单元测试

这是高维实现最重要的 correctness test。

随机小矩阵，例如：

\[
N=20,\quad d_Z=32,\quad m=16.
\]

同时运行：

```text
现有 feature-space full balancing
```

和：

```text
新的 sample-space dual balancing
```

必须检查：

\[
J_{\rm feature}
\approx
J_{\rm dual}
\]

以及：

\[
\nabla_ZJ_{\rm feature}
\approx
\nabla_ZJ_{\rm dual}.
\]

这不是实验 benchmark，而是数学实现单元测试。

没有通过以前，不允许跑 LIBERO-Mem。

---

# 21. Full-dual window 不再用当前完整 Welford M2

新的 full-dual accumulator 应该保存：

```text
z_detached
p_fixed
weight
cut_id
episode_id
```

即薄矩阵 rows。

不要保存：

\[
4096\times4096
\]

moment matrix。

window close 时再一次性组成：

\[
Z,P,W.
\]

这在 4096D 下反而更省。

---

# 22. Diagonal variant 单独实现

`RPBE-Diag` 仍保留。

但不能复用当前会构造 full `M2_zz` 的 `WeightedWelford`。

新增：

```text
DiagonalMomentAccumulator
```

只保存：

\[
\operatorname{diag}(C_{ZZ})
\]

以及：

\[
C_{ZP}.
\]

复杂度：

\[
O(4096m)
\]

而不是：

\[
O(4096^2).
\]

所以最终有：

```text
RPBE-Diag-4096
RPBE-Full-Dual-4096
```

两个版本。

---

# 23. RPBE gradient 如何回到 \(\Gamma\)

window close 时把 detached：

\[
Z
\]

临时变成 leaf：

```python
Z.requires_grad_(True)
```

计算：

\[
J_{\rm dual}(Z,P)
\]

得到每个 row：

\[
\frac{\partial J}{\partial z}.
\]

同一个 merge 的两个 horizon 梯度相加：

\[
g_v^{RPBE}
=
\sum_h
\frac{\partial J}{\partial z_{v,h}}.
\]

然后只 replay：

\[
\hat z_v
=
\Gamma_\theta(
m_a^{detach},
m_b^{detach}
).
\]

构造：

\[
\boxed{
\tilde L_{\rm RPBE}
=
-\sum_v
\langle
sg(g_v^{RPBE}),
\hat z_v
\rangle.
}
\]

所以 autograd graph 只有：

```text
m_a,m_b
→ Gamma
→ z
```

没有整个机器人 episode BPTT。

---

# 24. 不要删除 MemoryVLA 的 detach / no_grad

这是硬约束。

官方：

```python
@torch.no_grad()
_memory_consolidate(...)
```

以及：

```python
feat.detach().clone()
```

本来就是为了防止 memory 跨几百步保留计算图。

**不要为了训练 \(\Gamma\) 把这些全部删掉。**

否则：

\[
200\sim700
\]

step episode 很容易变成长 BPTT。

我们训练 \(\Gamma\) 必须走：

\[
\boxed{\text{record → latent adjoint → local replay}}
\]

而不是展开长图。

---

# 25. Task gradient 如何给 \(\Gamma\)

因为 bank 是 detach 的，所以正常 action loss 不会自动回到旧 merge 的 \(\Gamma\)。

这里使用一个非常局部的做法。

merge：

\[
m_a,m_b\rightarrow z_v
\]

发生以后，把：

\[
z_v
\]

作为一个 **detached leaf requiring grad** 写入训练用 bank：

\[
z_v^{leaf}
=
z_v.detach().requires\_grad_(True).
\]

下一次 policy decision retrieval 时，这个 state 会真实参与 action prediction。

正常：

\[
L_{\rm action}
\]

backward 后可以直接读取：

\[
\boxed{
g_v^{task}
=
\frac{\partial L_{\rm action}}
{\partial z_v^{leaf}}.
}
\]

然后 bank 再 detach。

因此没有：

\[
\Gamma\rightarrow20\text{ steps}\rightarrow action
\]

的长图。

最后 \(\Gamma\) replay：

\[
\hat z_v=\Gamma(m_a,m_b)
\]

使用：

\[
\boxed{
g_v
=
g_v^{task}
+
\lambda g_v^{RPBE}.
}
\]

这样：

```text
Gamma-Task
```

只使用：

\[
g_v^{task}
\]

而：

```text
Gamma-RPBE
```

使用：

\[
g_v^{task}
+\lambda g_v^{RPBE}.
\]

两个模型参数量、架构、宿主完全相同。

---

# 26. 参数版本一致性必须保留

这是从 TGN 代码必须继承过来的原则。

Cognitive memory state 来自：

\[
\text{LLM LoRA}
\]

然后经过：

\[
\Gamma.
\]

所以一个 RPBE statistics window 内：

\[
\boxed{
\text{LoRA 和 }\Gamma
\text{ 不能中途更新。}
}
\]

否则同一个 window 的：

\[
z_1,z_2,\ldots
\]

来自不同 representation parameter versions。

做法：

```text
开始 macro-window
    ↓
固定 LoRA + Gamma 参数版本
    ↓
顺序跑若干完整 episode
    ↓
积累真实 merge samples
    ↓
window close
    ↓
计算 RPBE latent adjoint
    ↓
replay Gamma
    ↓
统一 optimizer step
    ↓
下一 parameter version
```

Action DiT、retrieval/fusion 等不决定写入 cognitive bank 的 raw state，仍可按 task optimizer 正常更新。

特别注意官方默认：

```text
update_fused=False
```

因此 bank 保存的是当前 raw cognitive token，不是 gate-fused token。

这使这个参数版本控制容易很多。

---

# 27. episode 内不要更新 LoRA / Gamma

至少：

\[
\boxed{
\text{一个 episode 内 representation version 必须固定。}
}
\]

最简单实现：

- LoRA/Gamma optimizer 在 macro-window close 才 step；
- 普通 task modules 可正常 step。

这和当前 PRSS2 的 macro-group 思路一致。

---

# 28. 第一阶段不要碰 perceptual RPBE

保持：

```text
PerMemBank:
    adjacent cosine selector
    + official 0.5 average merge
```

只修改：

```text
CogMemBank:
    adjacent cosine selector
    + Gamma merge
```

原因是：

- cognitive 是 4096D；
- 正好验证高维 statistical RPBE；
- 语义上最接近长期 task state；
- 不引入第二种完全不同的 256×256 tensor compression 问题。

如果 cognitive-only 没有效果，再讨论 perceptual。

不要一开始同时实现两套。

---

# 29. 需要新增/修改的代码模块

### A. `vla/memory_vla.py`

修改：

```text
CogMemBank
```

新增：

```text
Gamma merger
MergeRecord hook
provenance
pending task-gradient leaf
```

不要修改：

```text
selector
retrieval mathematics
gate mathematics
action model interface
```

---

### B. `vla/datasets/datasets.py`

新增：

```text
DecisionStreamRLDSDataset
```

基于：

```text
StreamRLDSDataset
```

只改变 episode 内采样 index：

\[
0,K,2K,\ldots
\]

其中：

\[
K=future\_action\_window\_size+1.
\]

---

### C. `vla/materialize.py`

增加：

```text
dataloader_type="decision_stream"
```

只负责实例化上面的 dataset。

---

### D. 新增 `rpbe_embodied/records.py`

实现：

```text
MergeRecord
PendingMergeQueue
EmbodiedCutRecord
```

完全替代 TGN：

```text
CompactCutTrace
JodieFutureIndex
JodieCutBuilder
```

不要为了复用而强行套 graph node 语义。

---

### E. 新增 `rpbe_embodied/maps.py`

实现：

```text
fixed observation context map
fixed instruction map
fixed continuous action map
tensor-product CountSketch
```

保持：

\[
P=\psi(C,Y)
\]

完全 fixed。

---

### F. 新增 `rpbe_embodied/loss.py`

复用 PRSS2 数学逻辑，但新增：

```text
dual_full_score()
dual_latent_z_adjoint()
DiagonalMomentAccumulator
```

不要改坏现有 TGN `loss.py`。

最好：

\[
\boxed{\text{embodied 新文件，TGN 原实现保持不动}}
\]

等新实现稳定后再抽象公共代码。

---

### G. 新增 `train_libero_mem_rpbe.py`

负责：

```text
MemoryVLA checkpoint load
LoRA attach
single-A100 BF16
decision-stream training
pending future maturation
RPBE window lifecycle
task-gradient capture
Gamma replay
checkpoint
```

不要强行改官方 8-GPU `train.py`。

---

# 30. 实现顺序

## Task 1 — Host 跑通

只做：

```text
MemoryVLA
+ LIBERO-Mem
+ single A100
+ LoRA
```

不加 RPBE。

确认：

- official split；
- action shape；
- checkpoint；
- 评测 pipeline。

---

## Task 2 — Decision Stream

实现：

```text
DecisionStreamRLDSDataset
```

确认 timestep 是：

\[
\text{policy decision index}
\]

而不是 raw frame index。

---

## Task 3 — Merge tracing

仍使用官方 Average Merge。

只记录：

```text
MergeRecord
depth
left/right
start/end
merge time
```

确认树可以由 records 还原。

这里仍然不训练 \(\Gamma\)。

---

## Task 4 — \(\Gamma\) 插入

把 cognitive：

\[
Avg
\rightarrow
\Gamma.
\]

zero-init residual，保证初始输出和 Avg 接近。

Perceptual 不动。

---

## Task 5 — Future queue

每个 merge 等待：

\[
Y_1,Y_2.
\]

检查严格：

\[
future\_decision > merge\_decision.
\]

---

## Task 6 — Fixed embodied maps

构造：

\[
P_{v,h}=\psi(C_{v,h},Y_{v,h})
\]

并确认：

- LoRA 更新前后，给定同一 raw sample 的 \(P\) 完全不变；
- 不使用任何 LIBERO-Mem oracle annotation。

---

## Task 7 — High-dimensional statistics

先实现：

```text
RPBE-Diag-4096
```

然后：

```text
RPBE-Full-Dual-4096
```

小矩阵上严格验证：

\[
J,\nabla_Z
\]

和原 feature-space 实现一致。

---

## Task 8 — Local gradient replay

实现：

\[
g^{task}
\]

和：

\[
g^{RPBE}
\]

最终训练 \(\Gamma\)。

整个 episode 不允许建立跨时间 autograd graph。

---

## Task 9 — 正式实验

同一训练协议下至少跑：

```text
MemoryVLA-Avg
MemoryVLA-Gamma-Task
MemoryVLA-Gamma-RPBE
```

5 seeds。

其它 published LIBERO-Mem competitors 尽量直接使用论文结果；但 **MemoryVLA-Avg 必须在我们的单卡 LoRA protocol 下自己跑**，因为这是严格 host-matched control，不能拿别人不同训练设置的数字冒充。

---

# 31. 编码过程中绝对不能做的事情

1. 不改 MemoryVLA 的 adjacent-cosine selector。
2. 不删除整个 bank 的 `detach/no_grad`。
3. 不重新查询或展开完整 merge tree。
4. 不把 4096D state 为了 RPBE 直接降到 128D。
5. 不构造 4096×4096 covariance。
6. 不把两个 future horizons 平均。
7. 不使用 subgoal/object/mask 等 oracle labels 训练。
8. 不用 trainable LoRA cognitive token 构造 fixed future \(P\)。
9. 不让一个 RPBE window 混合不同 LoRA/\(\Gamma\) parameter versions。
10. 不同时改 cognitive 和 perceptual branch。
11. 不为了 LoRA 大幅重写 MemoryVLA model；只在 HF LLM 子模块挂 PEFT。
12. 不破坏现有 PRSS2/TGN 代码；embodied 先独立实现。

---

# 最核心的两项改动

如果下一位 AI 只记住两件事，就是：

### ① Chronological Merge Sampler

不是像 TGN 一样：

\[
\text{查树}\rightarrow\text{找 cut}.
\]

而是：

\[
\boxed{
\text{真实 policy-decision stream}
\rightarrow
\text{真实 consolidation}
\rightarrow
\text{MergeRecord}
\rightarrow
\text{等待未来 }Y_1,Y_2.
}
\]

树由 merge history 自然产生。

### ② High-Dimensional RPBE

真正 memory 仍然：

\[
\boxed{4096D}.
\]

Full balancing 不做：

\[
4096\times4096
\]

feature covariance。

而做：

\[
\boxed{
N\times N
\text{ sample-space exact dual balancing}.
}
\]

梯度：

\[
J
\rightarrow
\frac{\partial J}{\partial z_v}
\rightarrow
\text{local }\Gamma\text{ replay}.
\]

这两项是 MemoryVLA 适配真正需要解决的问题，其余尽量保持宿主原样。