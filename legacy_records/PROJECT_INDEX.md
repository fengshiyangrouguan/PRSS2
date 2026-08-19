# PRSS2 项目研究索引(供在线 AI 协作研究使用)

> 本文件是项目的自包含索引:记录背景、方法原理、模块与文件映射。
> 与在线 AI 协作时,可直接把本文件全文粘贴作为上下文;当 AI 需要看具体实现时,
> 再按第 8 节的文件路径提示,单独粘贴对应文件内容。
>
> ⚠️ 历史注记(2026-08-18):本文档描述的是 **old/ 归档版实现**(该目录已删除)。
> 文中所有 `old/...` 路径仅供历史参考;当前活跃实现见根目录
> `DEVELOPMENT v2.0.md`(v2.0 标准工程架构)。

---

## 1. 项目一句话概述

**PRSS(Predictive Relation-State Sheaf / Predictive Recursive State Sheaf,预测递归状态层)**
是一种面向递归/树式深度模型的**接口状态压缩方法**:在宿主模型(当前为 TGN)每个递归聚合接口上,
用"未来 continuation 读取矩阵"的谱(SVD)选出最有预测价值的固定维子空间,父节点只接收该子空间内的状态。
当前实验载体:官方 TGN(Temporal Graph Networks)× Wikipedia 动态图 × 动态节点分类任务。

---

## 2. 项目背景

### 2.1 目录结构总览(项目根 `E:\project\PRSS2\`)

| 路径 | 角色 |
|---|---|
| `PRSS_method_spec_v1_2.md` | **方法总规格书**(最高权威):理论定义、数学目标、训练算法、禁止事项、验收标准 |
| `PRSS-method/` | **当前实验实现**(推荐研究对象):官方 TGN 节点分类母本 + 单一 `vanilla\|prss` 开关 |
| `PRSS-method_old_021746/` | 旧版实现(已归档):宿主无关的 PRSS 包 + 7 项消融 + 合成树识别测试 |
| `tgn/` | 官方 TGN 代码 + **冻结态诊断插桩**(验证"改进动机"两个命题的前置工作) |
| `TGB-main/` | Temporal Graph Benchmark 基准库(未来标准化评测框架,当前未被引用) |
| `PINT-main/` | PINT(NeurIPS 2022,可证明表达力的时序 GNN),备用基线,当前未被引用 |
| `neural-sheaf-diffusion-master/` | Neural Sheaf Diffusion(NeurIPS 2022)参考实现,与 PRSS 仅结构类比 |
| `wikipedia.csv` | 原始数据:157,474 条真实交互,前四列 `user_id,item_id,timestamp,state_label`,后 172 维交互特征 |
| `processed_tgn_data/` | 官方格式预处理产物:`ml_wikipedia.csv` / `ml_wikipedia.npy` / `ml_wikipedia_node.npy` |
| `README_AUTODL_TGN_DIAGNOSTICS.md` | 诊断工作的完整说明(与 PRSS 动机直接相关) |
| `real_data_verification.json` | 真实数据的校验结果(行数、维度、时间单调、插桩只读性) |

### 2.2 数据与任务语义(已钉死,不可更改)

- **数据**:JODIE Wikipedia,bipartite reindex:0=padding,用户 1–8227,页面 8228–9227,交互编号 1–157474;每行 172 维 edge feature。
- **预训练任务**:官方自监督 future link prediction——每个真实交互为正样本,随机采一个页面作负 destination;`state_label` 不是该任务的 Y。
- **下游任务(PRSS 实验所在)**:冻结预训练 TGN,训练 MLP 分类器做动态节点分类,预测 `state_label`。
- **划分**:时间序切分 train/val/test = 110,232 / 23,621 / 23,621(70/15/15),正例极稀疏(test 仅 44 正 / 23,621)。
- **官方代码锚点**:twitter-research/tgn commit `d55bbe678acabb9fc3879c408fd1f2e15919667c`,核心文件 SHA256 冻结在 `PRSS-method/official_tgn/UPSTREAM_CORE_SHA256.json`。

---

## 3. 前置诊断:为什么认为 TGN 需要被改进(动机证据)

`tgn/` 目录的插桩只验证两个**先于方法存在**的命题:

1. **冻结 TGN 的递归状态 (h,C) 遗漏了仍影响未来的历史信息**(即 (h,C) 不是充分统计量);
2. **这部分 future-relevant 信息呈低秩结构**。

四个主诊断(实现在 `tgn/diagnostics/`):

| 诊断 | 文件 | 方法 | 判定标准 |
|---|---|---|---|
| 条件残差 | `conditional_residual.py` | 同一浅 probe 对比 `P(Y\|h,C)` vs `P(Y\|h,C,X^(K))` vs X 局部置换版,主量 pair-level ΔNLL + paired bootstrap 95% CI | ΔNLL>0 且 CI 不跨 0,置换后增益消失 |
| 近碰撞 | `collision_analysis.py` | 标准化 (h,C) 空间跨历史近邻匹配,报告距离与未来概率分歧曲线 | 极近 (h,C) 仍存在非平凡未来分歧 |
| Predictive SVD | `predictive_rank.py` | K-fold cross-fitted probe 生成 history×context signature 矩阵 G,中心化后 SVD 看谱与重建误差 | 出现 r ≪ 原始维度的稳定 elbow |
| 降秩回归 | `predictive_rank.py` | 未压缩 X → 未来 signature 的 reduced-rank regression(白化后秩约束) | held-out 上同样出现低秩 elbow |

关键实现约束:`extract_cuts.py` 导出必须**按时间重放**(官方 state_dict 不含 Python 层 pending raw-message 队列,直接加载随机访问会得到时间错误的状态);batch size 必须与训练一致(官方 memory 更新频率依赖 batch 边界)。

结论:"大方向成功"需四项证据同时成立;普通 TGN 的 AP/AUC-vs-depth 只是 supporting evidence。

---

## 4. 宿主模型 TGN:结构与训练

### 4.1 结构(五个机制)

- **Memory**(`modules/memory.py`):每节点一个向量压缩全部历史,附 last-update 时间戳与 pending raw-message 队列。
- **Message function**(`modules/message_function.py`):新事件消息 = concat[源 memory, 目标 memory, 边特征, 时间差编码] → MLP/identity。
- **Message aggregator**(`modules/message_aggregator.py`):同节点同批多条消息取 last / mean。
- **Memory updater**(`modules/memory_updater.py`):GRU 用聚合消息更新 memory,带时间单调断言。
- **递归嵌入 `GraphAttentionEmbedding`**(`modules/embedding_module.py`,默认):每层对每个节点用 NeighborFinder 取时间上最近的 n 个邻居 → **递归**计算邻居 embedding(注意:递归时沿用原始 query timestamp,而非父边时间)→ temporal attention 聚合(query=源节点特征+时间编码;key/value=邻居特征+边特征+时间差编码)→ MergeLayer 残差融合。用 memory 时层 0 输入 = raw 节点特征 + 当前 memory。
- **Time encoding**(`model/time_encoding.py`):TGAT 式 `cos(ω·Δt)`,ω 对数网格初始化。
- 其他变体:`GraphSumEmbedding`、`TimeEmbedding`(JODIE 式)、`IdentityEmbedding`。

### 4.2 训练协议(两阶段)

1. **自监督预训练**(`train_self_supervised.py`):future link prediction,BCE;每 epoch 重置 memory;每 `backprop_every` 个 batch 反传一次,反传后 **detach_memory**(截断梯度回溯);验证集 AP/AUC 早停选 checkpoint。
2. **冻结 + 下游节点分类**(`train_supervised.py`):宿主冻结,只训练 MLP decoder。**PRSS 实验就是在这个下游阶段把 PRSS 插进递归接口**;默认宿主冻结(`FINETUNE_HOST=0`),另有 `FINETUNE_HOST=1` 匹配共适应实验。

---

## 5. PRSS 改进了哪里(问题 → 改进点)

### 5.1 原问题

TGN 每个递归接口固定宽 172 维(k=172),"这 172 维装什么"**只由最终任务梯度间接决定**,导致:

- **历史混叠**:不同历史被压成同一状态;
- **提前遗忘**:某信息在当前节点单独看不重要,但与 sibling 聚合、穿过更高层 continuation 后才影响预测,根部梯度传回时它可能已被丢弃;
- **无预测语义**:固定宽度内的状态缺乏明确含义。

### 5.2 PRSS 的改进(四件事)

1. **候选加宽**:在 TGN 每层聚合后,把状态加宽到 d=256 维(`h = [原版聚合输出; 小网络学的补充]`)——多出的空间是"原料仓库";
2. **未来读取器 B(C)**:对每个计算树节点,训练一个小网络输入"挖掉当前节点后的上层 continuation 描述 C",输出 1×256 的读取向量 B(C),并用 `b(C)+B(C)·h` 直接预测标签(response loss 保证它真会读);
3. **SVD 选方向**:同层所有 B(C) 构成"算子库",统计预测 Gram `G=E[BᵀB]`,每 200 步 eigh 取 top-172 特征向量为投影矩阵 R,父节点只接收 `z=R·h`(仍 172 维);
4. **谱尾损失 L_spec**:`‖B(C)(I-RᵀR)‖²/(‖B(C)‖²+ε)`,逼迫 reader 把预测信息组织进当前 k 维商子空间,让神经网络与 SVD 互相配合(block-coordinate 交替优化)。

### 5.3 关键设计约束(规格书"禁止事项",与在线 AI 讨论时不要偏离)

- 从训练第 0 步起父节点就只接收商状态 z;**禁止**"先训 vanilla TGN 再离线压缩"、禁止 root-to-leaf 后处理压缩;
- **禁止**用 PCA 替代 predictive operator SVD;禁止用 MLP(d→k)删掉 SVD;禁止 Information Bottleneck / gate 作为主方法;
- R 是 detached buffer,SVD 不参与反向传播;谱更新用正交 Procrustes 对齐坐标;Gram 秩不足 k 时用旧商补全 null space(避免任意旋转);
- 训练期 reader/outside 可看未来 continuation,但**推理期禁止**;test label 不得更新 Gram 或 R;验证/测试阶段有硬审计(前后 Gram/SVD 计数与 R 矩阵必须零变化);
- 接口宽度 k 完全由宿主模型决定,PRSS 不规定统一宽度;同接口共享一个 R,不为每个 occurrence 单独参数化。

### 5.4 当前结果(seed 0,L=2,冻结宿主,候选 256→宿主 172)

| 运行 | test AUC |
|---|---|
| official_reference(逐字节官方脚本锚点) | 0.8933 |
| vanilla_matched(同母本对照) | 0.8855 |
| **prss_matched** | **0.9001** |

机制诊断:层 1 预测 Gram 有效秩 25、层 2 有效秩 10;每层 5 次 SVD 更新;energy@k=1.0;每次更新后 R 捕获能量上升;投影距离/主角度稳定。旧版实现(L=1,从头训练式)结果较差(prss 0.735 vs vanilla 0.785),已被"官方母本 matched"方案取代。

---

## 6. PRSS 原理(数学核心,精简版)

### 6.1 理论对象:上下文预测等价(contextual predictive equivalence)

对节点 v 的历史 H_v,令 C[□] 为"挖掉该子树后的合法上部 continuation"(父构造器、siblings、更高路径、最终预测)。

```
H ~ H'  ⟺  P(Y | C[H]) = P(Y | C[H'])   ∀ 合法 C
```

**关键定理(同余性)**:该等价关系对树 constructor 天然是 congruence——
若子节点历史等价,经任意父构造器 F_A 与 sibling 聚合后,父节点结果仍等价。
证明直觉:任一父层之上的场景 C_A 配上 sibling E 与 F_A 自动构成子节点的一个合法场景 C_D[□]=C_A[F_A(□,E)],故预测相等。
**意义**:允许从训练第 0 步起在每个递归接口同时压缩,层层安全。

### 6.2 可学习近似:lift + 线性读取

完整 H 与 C 高维连续,无法枚举所有 counterfactual context。因此:

- 用小型非线性 lift 得到候选表示 `h_v=Φ_ω(H_v)∈R^d`(d>k);
- 假设:lift 之后,未来 continuation 对历史的读取近似线性:

```
μ(h,C) ≈ b_η(C) + B_η(C)·h,   B(C)∈R^{p×d}  = future-reading matrix
```

### 6.3 SVD 的解析最优性(方法核心主张)

对算子库 B=[B(C₁);B(C₂);…],预测 Gram:

```
G = E_C[B(C)ᵀB(C)] = BᵀB / N
```

在"接口只能保留 k 维"约束下,最小化 predictive operator residual energy:

```
E_C ‖B(C)(I−RᵀR)‖²_F  ⟺  最大化 tr(R G Rᵀ)   (R 行正交, k×d)
```

**解析最优解 = G 的 top-k 特征空间**(等价于算子库 B 的 top-k 右奇异子空间)。
因此:SVD 不是启发式/初始化,而是固定秩 predictive-operator 近似的**解析最优子问题**。尾谱 tail@k 直接衡量接口预算无法承载的剩余预测能量。代码流式维护 G 的 EMA,从不显式构造巨型 B 栈。

### 6.4 总损失与交替优化

```
L = L_task + λ_resp·L_resp + λ_spec·L_spec
L_spec = E ‖B(C)(I−RᵀR)‖²_F / (‖B(C)‖²_F + ε)     (R detached)
```

- L_resp 防止 B→0 塌缩,并训练 reader/候选表示;
- L_spec 逼迫预测依赖集中到当前 k 维商子空间;
- SVD 周期性对算子库求 rank-k 最优 R —— 三者构成 block-coordinate / alternating spectral-neural optimization,不是后处理。

### 6.5 训练期 outside 上下文(仅训练期)

对每个节点 v 做一次 bottom-up + 一次 top-down outside pass:

- 根:`c_root = E(query/root metadata)`;
- 子:`c_v = O(c_parent, 父局部边/时间/mask 元数据, 关系类型, Δt, 兄弟候选的 detach() 摘要)`;
- **严格禁止**把当前 h_v 自己放进 c_v;Y 只能进 loss,不能进 context encoder。

### 6.6 理论谱系与边界

- 血统:context-history response operator → low-rank predictive state,与 Hankel operator / weighted (tree) automata / spectral predictive-state learning 同源;不做 sample Hankel table 补全,而是用神经网络把 C 映射为 functionalized response operator。
- 与 NSD 仅结构类比(`B(C)↔restriction map`、`G↔δᵀδ`),**不可声称 G 是 sheaf Laplacian**。
- 未闭合部分(论文不可过度声称):有限样本一致性、lift 的 identifiability(gauge 自由度,只关心 rowspace(R))、observed contexts ≠ 所有合法 contexts 的 support gap、rank-k operator 误差到 task risk 的全树上界(需 Lipschitz 条件)。
- 验收标准(规格 §30):reader 接近 unrestricted MLP、energy@k 高于 random/PCA、full PRSS 优于"同 lift + 直学投影"、history-mixing 样本上减少混叠、baseline parameter-matched、无未来泄漏。

---

## 7. PRSS 模块 ↔ 文件映射

### 7.1 当前实现 `PRSS-method/`(推荐)

**核心模块包 `prss/`:**

| 模块(概念) | 文件 | 作用 |
|---|---|---|
| 总装核心 | `prss/core.py` | `PRSSCore`:按层组装 quotients/readers/builders/outside;接口 `make_candidate` / `project` / `snapshots`;层 0(d₀=k₀)不训练 reader、不进算子库 |
| 候选加宽 lift | `prss/candidate.py` | `ExactPreAggregationCandidate`:`h_l=[原版聚合输出; φ(精确聚合输入)]`,φ 为 GELU+LayerNorm 小 MLP;输入 = source lower quotient + source time + 邻居 lower quotients + 边时间 + 边特征 + mask(全部为宿主聚合的原始输入) |
| Outside 上下文(仅训练期) | `prss/outside.py` | `Occurrence`/`Trace` 数据结构 + `OutsideContextEncoder`:`root_context`(标准化 log 时间+层嵌入)、`sibling_summary`(兄弟候选 detach 平均)、`child_context`(父上下文+父局部+关系嵌入+Δ t+兄弟摘要) |
| 未来读取器 | `prss/reader.py` | `ConditionalMatrixReader`:context→trunk(GELU+LN)→matrix_head 输出 B∈R^{1×d}、bias_head 输出 b;`logits = b + B·h`。另有 `UnrestrictedReader`(MLP(h,C),**仅监控**,独立 optimizer、detached 输入、不进 Gram) |
| 谱商(方法核心) | `prss/spectral.py` | `SpectralQuotient`:R buffer(`[I,0]` 初始化)、`accumulate`(Gram EMA,初始直接赋值)、`update`(eigh top-k → 有效秩 → 旧商补全 null space → Procrustes 对齐 → Grassmann 信任域步长部署,`spectral_step_size=1` 时精确部署)、`spectral_loss`(归一化谱尾,分母 detach 下界)、`snapshot`(谱诊断全集);工具:`row_orthonormalize`/`projector`/`procrustes_align_rows`/`principal_angles` |
| 辅助损失构建 | `prss/auxiliary.py` | `build_auxiliary`:自顶向下 outside pass(BFS),每根内先按层平均再跨层平均(防止低层 10× 数量优势);产出 response/spectral/unrestricted loss、`matrices_by_layer`(B.detach().clone() 不可变快照)、contexts、计数 |
| TGN 递归适配器 | `prss/tgn_adapter.py` | `PRSSTGNEmbeddingAdapter`:包裹官方 `GraphEmbedding`,不改宿主聚合逻辑;每个递归 return 前插入 `make_candidate → project`,父节点只接收 z;training-only trace 插桩(occurrence 树) |
| 监控与硬不变量 | `prss/monitoring.py` | `MonitorWriter`(step/epoch/alerts/monitor_summary/jsonl + 投影快照);有限性、Gram 对称、R 行正交、验证/测试谱状态冻结审计;`tensor_summary`/`matrix_stats`/`module_finiteness`/`grad_l2` |

**实验脚本 `experiments/`:**

| 文件 | 作用 |
|---|---|
| `train_supervised_prss_switch.py` | **主训练入口**(`--mode vanilla\|prss`):保留官方完整 `compute_temporal_embeddings(src,dst,dst)` 调用、官方 MLP decoder、memory 重置与时间序重放;新增 val 早停/held-out test/AP·NLL 监控/谱隔离审计/rolling checkpoint 断点续跑/`--finetune-host` 开关 |
| `train_node_classification_faithful.py` | 上者的 1 行入口别名 |
| `compare_nodecls.py` | 汇总三个 summary.json 并打印 PRSS−vanilla 差值 |
| `summarize_official_reference.py` | 从官方脚本工作目录提取结果到 summary.json |

**锚点与测试:**

| 文件 | 作用 |
|---|---|
| `official_tgn/` | byte-for-byte 上游脚本 + 冻结 SHA256 清单(`UPSTREAM_CORE_SHA256.json`、`UPSTREAM_COMMIT`),`source/` 为未改动上游源码 |
| `tests/test_prss_runtime.py` | 谱商 SVD 等价性(显式堆叠右奇异子空间 = 流式 Gram 特征空间)、Procrustes、梯度不穿 R 等 |
| `tests/test_official_parity.py` | 上游 SHA256 一致性 |
| `tests/test_monitor_finiteness.py` / `test_resume_rng.py` | 监控不变量 / 断点续跑 RNG 一致性 |
| `run_tonight_2gpu.sh` / `run_autodl.sh` | AutoDL(Linux GPU 云端)编排:GPU0 vanilla → 官方锚点,GPU1 PRSS,最后 compare |

**方法文档:** `METHOD.md`(运行时契约)、`PRSS_METHOD_SPEC.md`(严格规格)、`OFFICIAL_PARITY.md`(母本对比契约)、`MONITORING.md`(监控契约)、`HOTFIX_V4/V5/V7.md`(稳定性修复历史)。

### 7.2 旧实现 `PRSS-method_old_021746/`(已归档,含消融与合成测试)

| 概念 | 文件 |
|---|---|
| 接口维度契约 | `prss/config.py` |
| 候选 lift(含线性/恒等消融) | `prss/lift.py` |
| Outside 上下文 | `prss/outside_context.py` |
| 读取器(conditional + unrestricted) | `prss/reader.py` |
| 谱(Gram EMA/eigh/Procrustes) | `prss/spectral.py` |
| 损失 | `prss/losses.py` |
| 状态/系统/递归执行器/训练器 | `prss/state.py` / `system.py` / `recursive.py` / `trainer.py` |
| 七项具名消融(fixed_random / pca / direct / linear_reader_svd / no_nonlinear_lift / neural_svd_no_spec / full) | `prss/ablations.py` |
| TGN 集成 | `prss/integrations/tgn.py` |
| 监控 | `prss/monitoring.py` |
| **合成树识别测试**(Phase 2:已知子空间 D,E→A→Y,主角度 0.00256 rad) | `experiments/synthetic_tree.py` |
| JODIE 预处理 / 训练入口 / 对比 | `experiments/preprocess_jodie.py` / `train_tgn_prss.py` / `compare_prss_runs.py` |

### 7.3 诊断插桩 `tgn/`(动机证据代码)

| 概念 | 文件 |
|---|---|
| 冻结重建/重放/浅探针/paired bootstrap/指标 | `diagnostics/common.py` |
| 历史描述子 X(结构 12 统计 + 边特征 3×172 + 层 0 状态统计) | `diagnostics/history.py` |
| 冻结权重按时间重放导出 h/C/X/Y | `diagnostics/extract_cuts.py` |
| 条件残差 ΔNLL(M0/MK/MK-shuffle) | `diagnostics/conditional_residual.py` |
| (h,C) 近碰撞与未来分歧 | `diagnostics/collision_analysis.py` |
| cross-fitted predictive SVD + held-out RRR | `diagnostics/predictive_rank.py` |
| 基线汇总(层数×memory×seed) | `diagnostics/summarize_baselines.py` |
| 真实数据 fail-fast 校验 + observer 只读性 | `diagnostics/verify_real_data.py` |
| 只读 observer 插桩(默认关闭) | `modules/embedding_module.py`(末尾) |
| 编排 | `run_autodl.sh` |

### 7.4 周边代码库

| 目录 | 角色 |
|---|---|
| `TGB-main/` | Temporal Graph Benchmark:22 数据集、官方负采样、mrr/ndcg 评估器;PRSS 未来标准化评测(如 tgbl-wiki)的候选框架 |
| `PINT-main/` | PINT:单射聚合+位置编码的可证明表达力时序 GNN,基于 TGN 代码;未引用,备用基线 |
| `neural-sheaf-diffusion-master/` | NSD:stalk/restriction maps/sheaf Laplacian 扩散;与 PRSS 仅结构类比 |

---

## 8. 当前实验关键超参数(seed_0_l2 实际运行)

```
n_layer=2  n_degree=10  bs=100  n_epoch=10  patience=3  use_memory=True
host_dim(k)=172  candidate_dim(d)=256  candidate_hidden=128
context_dim=64  reader_hidden=128  trace_roots=8
lambda_response=1.0  lambda_spectral=0.1
gram_ema=0.05  spectral_warmup=200  spectral_interval=200  spectral_step_size=0.25
finetune_host=False(官方式冻结宿主;另有 =1 的匹配共适应实验)
```

---

## 9. 待办 / 下一步(与在线 AI 讨论时的候选方向)

1. L=1(官方论文默认深度)复现当前结论;多 seed;λ_spec ∈ {0, 0.01, 0.1, 0.5, 1.0} 敏感性;
2. `FINETUNE_HOST=1` 匹配共适应实验(与冻结宿主结果分开报告);
3. 规格 §22 八项必做消融(当前实现只做了 matched vanilla,消融在旧实现里);
4. 迁移到 TGB 标准协议评测(多数据集泛化)——**v2.0 已落地此方向**;
5. 理论空白:有限样本一致性(Davis–Kahan 类扰动界)、rank-k 算子误差到 task risk 的递归上界、support gap 泛化验证。

---

## 10. 与在线 AI 协作的使用建议

- **通用上下文**:粘贴本文件全文即可让 AI 理解项目全貌。
- **讨论原理**:引用规格书 `PRSS_method_spec_v1_2.md` 的节号(§2 等价关系、§4 SVD 最优性、§6 outside 编码、§9 谱尾、§12 伪代码、§16 泄漏边界、§21 诊断、§22 消融、§29–30 阶段与验收、§33 禁止事项)。
- **讨论实现细节**:让 AI 指定要粘贴的文件路径(7.1 表格),再单独粘贴该文件;核心算法集中在 `prss/spectral.py`、`prss/auxiliary.py`、`prss/tgn_adapter.py`、`experiments/train_supervised_prss_switch.py`。
- **讨论结果**:粘贴 `outputs/node_classification/wikipedia/seed_0_l2/*/summary.json` 与 `monitor/*.jsonl`。
- **重要提醒**:与 AI 讨论修改方案时,以 §5.3 的禁止事项为红线;任何"先训练后压缩 / PCA / 删 SVD / 推理期读取未来"的提议应直接拒绝。
