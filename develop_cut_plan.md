# develop_cut：cut 语义修正 + Moment-Adjoint Replay 执行计划（第四轮审阅修正版）

## Context（为什么改）

任务② 0.8606 实为 task-only（窗口 10120 次未关闭、Ky Fan 全程空转）；显存 14.3GB 根因是统计窗口被实现成反传窗口（宽图）。四轮审阅已收敛：同树 consumption 取代 next-event、cut 全局身份、多 horizon 分行加权、Welford replay、matched task-only 对照。本版落实第四轮意见的**三个拍板问题**（见任务 0）与实现级修正（四类 ID、self 链、路径编码、cluster ESS、回放顺序、A/B 测试形态）。

当前分支 `develop_cut` @ `15023e2`；`50d8a2f` 在历史供选择性恢复。

## 任务 0：三个拍板问题的答案（已定，执行基线）

### 拍板 1：Y1/Y2 = downstream consumption probe；Y★ = 正式方法必需项

- 历史交互 label（时间上早于 t_root、计算序在 cut 上方）**不是未来预测**：第一版作为固定 probe 使用，论文口径降级为"压缩状态保留了同树上层构造所需的可观测记录属性"
- **Y★（根任务结果）不再无限期推迟**：列为正式必需项，与任务 1 同期设计接口（h=★ 行结构预留），实现排在 replay 之后（任务 5.5）
- φ_Y(0)≠0：现有 `maps.future_table` 是 ±1 签名表，φ_Y(0) 天然非零；**加测试断言锁定**（两行均非零）

### 拍板 2：h1 读取 `parent(v).consumption`（shifted closure，定义 B）

- 审阅者两轮伪代码（`occurrence.parent_consumption_record`）均为 B，且 B 更接近 shifted closure
- 文档口径改写为：**Y_v^(1) = 完成一次父构造后所得状态的下一次 consumption record**；h=2 同理再上一步
- 四层树落点：cut@layer0：h1=layer1 的消费边、h2=layer2 的消费边；cut@layer1：h1=layer2 消费边、h2=root 任务 record；cut@layer2：h1=root 任务 record、h2 缺失；cut@layer3(root)：无行（Y★ 例外，见任务 5.5）
- 未来预测性质由 h 到达 root 时提供；layer0 接触根结果靠 Y★
- **测试锁定 off-by-one**：合成树每条边标不同 label，断言 leaf.h1 的 Y == layer1 被消费边的 label（不是 leaf 自己的边）

### 拍板 3：label 归属与对齐规则

- 代码事实：邻接表**双向**（utils.py:94-98 每条交互同时入 src/dst 邻接表）；`edge_idxs` 是 **1-based 的 graph_df.idx**（非交互流位置，sanity_check `edge_idxs_one_based` 佐证）；wikipedia label = destination（item/page）状态标签（官方 TGN node-classification 协议预测 dst 状态）
- **显式建表**（不假定 `labels[edge_idx]`）：`label_by_edge_idx = {int(idx): float(label) for idx, label in zip(data.edge_idxs, data.labels)}`（冲突时记录并审计）
- consumption record 字段：`{edge_idx, edge_time, endpoint_role(0=occurrence 节点是 source/1=destination), label_owner, counterpart}`
- **对齐规则**：历史边 probe 仅当 occurrence 节点 == label_owner（即该 occurrence 是交互的 destination）时 Y 有效，否则该 h 项 mask（C 仍记 endpoint_role）；root 任务 probe 天然有效（label 属 dst，counterpart=dst 进 C）
- **集成测试**：小型 NeighborFinder + 随机抽真实 CSV 行核对 idx→label 映射与对齐（数据在本机 datasets/ 则本机跑，否则云端）

## 任务 1：CutRecord 语义重写（第一步执行，文件级详细）

### `src/rpbe/hosts/jodie_tgn.py`

- Adapter 构造时接收 data 引用（sources/destinations/labels/edge_idxs 全量数组）
- 父层建树循环中，对每个邻居子 occurrence `nid` 后补 metadata：`consumption = {"edge_idx": int(edge_idxs_np[row,j]), "edge_time": ..., "endpoint_role": 0 if sources[edge_idx-1]==source_nodes[row] else 1, "label_owner": ..., "counterpart": ...}`；source 链子写 `consumption = {"role": "SELF", "event": None}`（**self 链不删行**）
- root occurrence 的 consumption（任务 record）不在 adapter 写，由 CutBuilder 从 batch 数据映射（root_rows → (src, dst, label, t_root)）

### `src/rpbe/records.py`

- `CutRecord` 加 `weight: float = 1.0`；**删 `FutureIndex` 类与 `query`**（neg_pool 无消费方，一并删）
- **四类 ID 分离**：
  - `cut_id = (tree_id, occurrence_id, tau)` —— gate 的 unique cut 计数
  - `row_id = (tree_id, occurrence_id, tau, horizon)` —— 窗口行去重
  - `overlap_id = (node, time, tau)` —— 相关性分组统计（**禁止硬删**）
  - `outcome_id = ("edge", edge_idx)` 或 `("root", root_event_id)` —— outcome 复用统计
- `build` 重写：① `parent_of` 逆映射（断言每 occurrence 至多一父）② **从每个实际压缩 occurrence 开始**（不管自身有无 consumption——self 链也遍历），沿祖先链找 h=1,2：h1 载体 = `parent_of[v]`，h2 载体 = `parent_of[parent_of[v]]`；载体是 SELF 或无 record → **继续向上找第一个有真实 probe 的 ancestor**（跳过层计入 C 路径编码，不删 cut）③ 载体 record 按对齐规则：历史边不对齐（occurrence 节点 ≠ label_owner）→ 该 h mask；root record 恒有效 ④ context = `{"horizon", "delta_t", "counterpart", "role", "query_type", "endpoint_role", "path": [(rel_1, Δt_1), ..., (rel_h, Δt_h)]}`（path 从 child_relations/child_delta_t 取，含 SELF 步）⑤ `weight = 1.0/|H_v|`（有效 horizon 数）⑥ cap 抽样**按 cut**（抽中 cut 保留全部 horizon 行）⑦ layer3 无上方不产行；`kf_taus` 白名单显式排除 layer3（config 加字段）

### `src/rpbe/loss.py`

- `dedup_cut_rows` → 分行版：返回 `(row_ids, cut_ids, tree_ids, zs, ps, weights)`；窗口去重键 row_id、gate 计数 cut_id
- `_close` 加权协方差（50d8a2f w 路径 + direct 中心化 + `den_w`），**W2 按 cut 聚类**：`W2_cut = Σ_v (Σ_h w_{v,h})²`、`D_cut = W − W2_cut/W`（行权重 w_{v,h} 用于矩，gate/ESS/自由度按 cut）

### `src/rpbe/maps.py`

- horizon 编码：categorical_c 追加 `(2, d_c)`（seed+6）；非法值 **raise**
- **路径编码 PathSketch**：新增固定 buffer，`path = [(rel, Δt)]` 逐步签名（rel hash + Δt RFF 的固定组合），`context_vector`/`pv_batch` 纳入
- φ_Y(0) 非零断言测试

### `src/rpbe/hosts/jodie_bridge.py` + `training/jodie_loop.py`

- bridge 删 FutureIndex，传 data 数组；loop 传 root 事件信息；kf_taus 过滤

### 测试（本机全绿后提交）

- records：完整四层合成树 fixture + consumption metadata 手工构造；**每条边不同 label 锁定 h1/h2 对应关系**；self 链向上找 probe；对齐 mask（in 边 carrier ≠ owner → 该 h 缺）；四类 ID 断言；cap 按 cut；parent_of 断言
- loss：row_id 去重 vs cut_id 计数分离；cluster W2 与行级 W2 的差异用例；加权协方差回归（50d8a2f 测试捡回）
- maps：horizon 编码 + 非法 raise + φ_Y(0) 非零
- 集成：小型 NeighborFinder + 真实 CSV 行核对 idx→label 与对齐

## 任务 2：样本审计（无 cap、无旧过滤；云端，跑前征得同意）

`scripts/audit_cuts.py`：no_grad 前向遍历训练区间，统计：`raw_compression_occurrences / neighbor_cut_coverage / self_cut_coverage / valid_Y1 / valid_Y2 / aligned_vs_unaligned_probe / missing_due_to_tree_depth / node_time_collision_count / unique_cut_id / overlap_group_count` + **outcome 侧**：`outcome multiplicity / unique outcome count / train-audit outcome overlap / outcome-cluster ESS` + 每 τ `N_unique_tau / N_root / ESS` + 各层 r_τ + pooled-vs-separate 闭合残差。M 不预设。

## 任务 3：树等权 + 采样校正（50d8a2f 选择性恢复，依赖任务 2）

`w_v = 1/该树切口数`、采样校正 `n/k`；不恢复软去重与旧 cut 语义。总权重 `w = w_tree × w_sample × 1/|H_v|`，测试锁定。

## 任务 4：FP64 加权 Welford/Chan + replay adjoint 接口

第一遍 no_grad 加权 Welford/Chan 一遍累计 `W, W2_cut, mean_z, mean_p, M2_zz, M2_pp, M2_zp`（detached、FP64）得全窗口 μ̂_z、μ̂_p、D_cut；小矩阵求 A_zz/A_zp/A_pp；第二遍每 batch 用同一全局 detached 均值中心化 → 各 batch 可加 → 精确。固定权重下 W/W2/D 对 θ 导数为零，无需单独反传。

## 任务 5：Moment-Adjoint Replay（顺序明确）

```text
memory_shadow = backup_memory()       # 先备份
rng_shadow = backup_rng()             # torch cpu/cuda + numpy + python
cached_samples = sample_window_once() # 树/邻居/负样本/cut paths/weights/dropout masks 离散缓存
pass 1: run_no_grad(cached_samples)   # Welford 累计
restore_memory(memory_shadow); restore_rng(rng_shadow)
pass 2: run_with_grad(cached_samples) # 逐 batch task loss(÷宏窗口 batch 数) + KF-VJP，立即 backward
每 batch detach_memory()；宏窗口结束一次 optimizer.step()，scheduler 按 step
```

第二遍结束的 memory 状态保留；第一遍的丢弃。日志记录第一遍**真实 F(S_W)**。

## 任务 5.5：Y★（根任务结果）接口与实现

h=★ 行结构任务 1 已预留；实现：cut 沿祖先链直达 root 任务 record（counterpart=dst、label=root label）。layer3 在 Y★ 上线后重新纳入 kf_taus 评估。

## 任务 6：同输入 A/B 梯度对比（两种测试都做）

固定相同 z/p/w/masks：① `z` 设 `requires_grad=True` 比较 z-gradient ② 小确定性 MLP `z_θ=f_θ(x)` 比较参数梯度。A = direct-stack 损失核（仅损失核参考），B = replay。报告 max abs、相对误差、梯度 cosine；谱间隙好的合成矩阵严格测试；<1e-5 不作统一硬门槛。

## 任务 7：matched macro-window task-only + 0.8606 改标

同宏窗口、同 task loss 缩放、同 optimizer/scheduler、λ_KF=0 重跑 TGN+Adapter。0.8606 改标"旧工程 sanity check（旧协议）"。

## 任务 8：云端短跑验收（跑前征得同意）

`kf_updates == expected`；N_unique ≥ 审计定的 M；有效秩 ≥ 32；`kf_grad_norm > 0` 有限；task=0 时 adapter 仍被 KF 更新；峰值显存不随窗口 batch 数线性增长；summary 区分训练期累计 KF 与事后 audit；**outcome-disjoint audit**（排除与训练集重叠 outcome 的审计，证明非同一批历史 label 的记忆）。

## 任务 9：正式矩阵

baseline（0.8948）/ matched task-only（任务 7）/ RPBE 三组。`train_auc → online_train_auc`，可选 epoch-end `train_eval_auc`。

## 验证方式总表

| 阶段 | 位置 | 标准 |
|---|---|---|
| 任务 1 | 本机 | compileall + records/loss/maps 测试全绿（含 off-by-one 锁定、对齐 mask、四类 ID） |
| 任务 2 | 云端 | funnel 表 + outcome 复用表 + pooled-vs-separate 残差 |
| 任务 3-6 | 本机+云端 | 单测 + A/B 两种梯度报告 |
| 任务 7-9 | 云端 | 三组 summary 对照出表 |

## 风险清单

| 风险 | 对策 |
|---|---|
| in 边 probe 对齐率低（mask 砍样本） | 审计 aligned/unaligned 比例；过低则议 endpoint_role 条件 probe |
| graph_df.idx 与 label 映射的 1-based/重复约定 | 显式建表 + 集成测试核对 + 审计冲突计数 |
| 路径编码增加固定映射维度 | PathSketch 复用 sketch_indices 机制，m 不变 |
| Y★ 使 h 结构变三档（1/2/★） | 任务 1 预留行结构，任务 5.5 实现 |
| self 链向上跳层使 h 语义混入跳过步 | C 路径编码含 SELF 步；测试锁定 |
