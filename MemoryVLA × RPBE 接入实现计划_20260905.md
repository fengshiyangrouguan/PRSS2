# MemoryVLA × RPBE 接入实现计划（main 分支，embodied 跨域第三站）

日期：2026-09-05
规格源：《MemoryVLA × RPBE：单卡 A100 实现任务说明.md》（31 节设计 + 12 条禁令 + Task 1-9），本计划是对它的可执行实现拆解（含 2026-09-05 审阅裁决 5 点修正）。

## Context

TGN（图时序）与 CCM（对话 LLM）两个跨域实验已建立 RPBE 方法验证链。本计划是第三个宿主：**MemoryVLA**（ICLR 2026，机器人操作 VLA，vendored `openvla-codebase` 分支 @ d732ea9）上的 LIBERO-Mem 实验。

核心问题：MemoryVLA 的 cognitive memory 通过 **ToMe adjacent-cosine 选择 + 算术平均**压缩记忆（`CogMemBank._consolidate_with_token_merge`，memory_vla.py:210-232），平均法使合并前后信息不可逆、未来不可判。我们用 **Γ 残差合并器**替换平均（selector 不动），并用 **RPBE 未来充分性损失**（两条未来 decision 观测 + Ky-Fan 谱得分）训练 Γ——这是"未来充足递归压缩"在具身序列上的第三次验证。

**与 TGN/CCM 的两个结构性差异**（任务说明最核心两项）：
1. **Chronological Merge Sampler**：不查树、不重建树——memory bank 在真实 policy-decision stream 上自然合并，**一次真实 consolidation = 一个 RPBE cut**；树由 merge history（left/right provenance）自然产生。
2. **High-Dimensional RPBE**：cognitive state 是 **4096 维**，禁止 4096² 特征协方差 → **sample-space dual**（N×N Gram，N=窗口行数）与特征空间 scale-normalized 目标**严格等价**（§20 小矩阵等价门禁测试不过，禁止训练）。

环境现状（已完成）：A800 80GB 服务器（autodl-vla，SSH 29513）就绪；openvla-7b-prismatic 30GB checkpoint 下载完成并 **smoke 加载通过**（SMOKE_OK：LlamaForCausalLM 32 层、cog 4096 维、memory 模块含官方预训练权重、action_model 随机初始化）；Llama-2-7b 本地权重（ModelScope，LLAMA2_LOCAL_PATH 注入）；pip 依赖齐（无 TF——py3.12 装不了 TF 2.15，RLDS 栈已惰性化）；flash-attn 统一禁用（三臂一致 eager）。

## 审阅裁决（2026-09-05，5 点必须修死，已吸收进本计划）

方案整体成立，无需换宿主或重构 Γ。以下裁决写入规格：

1. **监督事件 = 严格 causal cut（数据构造不可违反协议）**：merge 发生在 post-decision memory update 时刻 `τ_v`，监督**只能**是 `Y_v⁽¹⁾=A_{τ+1}`、`Y_v⁽²⁾=A_{τ+2}`；且 **两个未来同时存在**才是 valid RPBE cut（`valid ⟺ Y⁽¹⁾,Y⁽²⁾ both observed`）。`τ+2 ≥ T_episode` 的 cut 一律 **censored：不进入 RPBE window**（仍正常参与 task loss）。废除"缺 horizon 省略"的宽松语义——不用"有一条算一条"。
2. **公平性措辞 + Fixed-trace diagnostic**：selector 不动 ≠ merge tree 不变（z^Γ≠z^Avg 会改变后续 cosine 选择，merge tree 内生分叉——这是特性不是 bug）。论文措辞："the consolidation rule and memory budget are identical; the merge operator differs, allowing subsequent memory trajectories to evolve endogenously"（禁止说 schedules identical）。补 Fixed-trace 诊断：Avg 臂跑出 merge trace，在 held-out 子集强制三臂用相同 pair sequence 对比——回答"相同 compression tree 下 RPBE merger 本身是否更好"。
3. **repr update window ≠ RPBE statistics window（两个概念显式拆分）**：Gamma-Task 臂没有 RPBE loss **不代表没有 macro-window**——它必须与 Gamma-RPBE 臂在**完全相同的 macro boundary** step Γ。三臂 repr 时间表一致：`g_Γ = g_task`（Task 臂）/ `g_Γ = g_task + λ·g_rpbe`（RPBE 臂），同一边界更新。
4. **压缩样本覆盖率统计（比 λ 更优先）**：L=16 时每 episode 可能只有几次 merge。Task 1/2 后必须先出统计表：decisions/episode、merges/episode、valid 2Obs cuts/episode、merge depth 分布、% decisions after first merge、% retrieved merged states。**若 valid cuts/episode ≤ 2，调整 decision stride 或 memory budget**。补 L=4,8,16 compression-pressure 曲线（预期：小 budget ⇒ RPBE advantage 增大——VLA 最有价值的机制结果）。
5. **Task 0.5 checkpoint integrity audit**：action_model 随机初始化必须查明（故意 or loader 缺陷）。检查 missing/unexpected keys、action expert/memory module 参数来源、LoRA attach 前后参数 hash、跑官方 eval 1 task 确认非随机。不通过则 15 个正式 run 不得启动。**另加 replay gradient 门禁**（Task 8）：tiny synthetic merge 上 direct backward vs replay 梯度，cos > 0.99999（task 与 RPBE 两条路径都做）——TGN 被梯度实现坑过的教训。
6. **论文措辞**：写 "replace the averaging operator **in cognitive-memory consolidation**"；perceptual stream 保持官方平均（这本身是优点：修改更小、归因更清晰）。

## 修订备案（2026-09-05，Task 0 执行中）

1. **官方 val 未发布**：论文承诺 120 轨迹/task（100 train + 20 val），但 HF 数据集 `libero-mem/LIBERO-Mem` 每任务仅含 100 个 demo（demo_1..demo_100），无 val 文件/分支/标记。**用户裁决 A**：自划 80/20——`demo_1..demo_80 = train`、`demo_81..demo_100 = val`（确定性固定规则，5 seeds 共用同一划分，报告如实声明）。划分依据：按官方 demo 编号顺序（无重排）。
2. **数据集体量修正**：10 任务全量 ~210GB（非此前信息 19.9G）。**实验改为单任务 T3**（KITCHEN_SCENE1_3，碗 3 次举放，18.4GB，~400 帧/轨迹）——用户裁决（强时序性 + 训练速度平衡）。joint 协议的相关内容保留供后续扩展。
3. **实验定位锁定（Task 0.5 审计）**：openvla-7b-prismatic checkpoint 的 state_dict 仅含 llm_backbone/projector/vision_backbone——**机器人侧全部随机初始化**（action_model 408.8M + cog_mem_bank 423.7M + per_mem_bank + per_compr 共 ~835M）。定位 = "training a MemoryVLA-based host under matched initialization"，不得称 fine-tuning。
4. **训练步数缩减**：单任务 80 条轨迹 vs 官方 500 条——40k 步会过拟合。**max_steps 初定 10k**（每 1k 步 val 验证，best-val 选模），烟测后按收敛调整；Go/No-Go 由 Task 1 烟测数据驱动。

## 前置条件（Task 0 / 0.5，需用户操作）

1. **磁盘扩容**：数据盘 50G 现余 ~6G，装不下 LIBERO-Mem（~19.9G HDF5）→ AutoDL 控制台扩容数据盘（或挂新卷）。
2. **下载 LIBERO-Mem**：HF 数据集 `libero-mem/LIBERO-Mem`（hf-mirror）。格式为 HDF5（非 RLDS）：`data/demo_i/{actions(T,7), obs/{agentview_rgb(T,256,256,3), ...}}` + `metainfo.json`（task_description/success/subgoal/boxes）。10 tasks × 4 memory 维度 × 120 轨迹（100 train/20 val）。
3. **probe 脚本先行**（`scripts/probe_libero_mem_hdf5.py`，只读）：确认文件命名、train/val 划分来源（metainfo 或文件名约定，**核查无轨迹重叠**）、dtype、instruction 归属、总大小。oracle 键（subgoal/object ID/mask/success）**只进 eval 路径**，训练路径 grep 审计。
4. **Task 0.5 checkpoint integrity audit**（审阅裁决 5）：在 smoke 加载基础上核对 30GB checkpoint 的 state_dict key 全景——`action_model` 是否真缺（官方 openvla-7b-prismatic 是否本就只含 VLM）；若缺，**明确实验定位 = "training a MemoryVLA-based host under matched initialization"（不得称 fine-tuning）**，并记录"三臂从相同随机 action expert 出发"；检查 memory 模块（cog/per bank、per_compr）权重是否来自 checkpoint、LoRA attach 前后参数 hash；Task 1 烟测后加跑官方 eval 1 task 确认不是随机水平。

## 模块设计

### A. 新包 `src/rpbe_embodied/`（纯 torch，零 memoryvla 依赖，不 import rpbe——§31.12）

| 文件 | 职责 |
|------|------|
| `config.py` | `EmbodiedRPBConfig`（m=64 sketch 维、d_c=64、d_f=64、ridge_eps=1e-4、rpbe_seed=0、kf_min_abs=128 merges、gamma_rank=64、lambda_rpbe 校准后写入） |
| `records.py` | `MergeRecord`（episode_id/merge_id/merge_decision_time/left·right·merged_state(detached)/left_id/right_id/depth/start/end/param_version）、`PendingMergeQueue`（两级 future 成熟，strict future_decision > merge_decision）、`EmbodiedCutRow`（薄行：cut_id/horizon/z/context/outcome/weight）。复用 cut_id/row_id 元组约定与 HORIZON_OMEGA=(0.5,0.5)；**完全替代** TGN 的 CompactCutTrace/JodieFutureIndex/JodieCutBuilder |
| `maps.py` | `EmbodiedFixedMaps`：固定 vision context map（冻结 DINOv2+SigLIP 特征均值）+ instruction hash + Δs/h 时元 + 连续 action RFF → 张量积 CountSketch P=ψ(C,Y)。复制 FixedMaps 的 CountSketch 构造模式（base 全覆盖 + k-1 随机重复、固定 seed buffer、`[2,nnz]` indices、批 index_add_），换掉 JODIE 语义 phi |
| `loss.py` | `dual_full_score`/`dual_latent_z_adjoint`、`diag_score`/`diag_latent_z_adjoint`、`EmbodiedRPBEWindow`（薄行窗口）、`gamma_replay_loss` |
| `__init__.py` | 导出 |

### B. MemoryVLA vendored 改动（最小 diff，本机 `third_party/memoryvla/` ↔ 服务器 `/root/autodl-tmp/vla/` 同步）

1. **新文件 `vla/gamma_merger.py`**：`GammaMerger(dim=4096, rank=64, alpha_init=1.0)`——`m_avg + α·U·MLP([P·m_a; P·m_b])`；P 随机（std=dim^-0.5）、MLP **全随机**（local generator，不吃全局 RNG）、**U 零初始化**、α=1 → **起点严格 = 官方 AvgMerge**（1·0·h=0）且学习路径畅通（审阅裁决 B1：α=0 或 MLP 末层零初始化会把全部梯度封死）。参数量 ~575K。
2. **`vla/memory_vla.py`（唯一宿主修改点）**：
   - `CogMemBank.__init__` 增 `gamma=None / record_merges=False / capture_task_grad=False` + provenance 并行结构（`prov` dict、`next_merge_id`、`merge_log`）——**retrieval/gate/selector 代码一行不改**。
   - `_consolidate_with_token_merge`：selector（adjacent-cosine argmax）不动；`fused_feat = self.gamma(feat_i.detach(), feat_j.detach()) if self.gamma else 0.5*(feat_i+feat_j)`（**仍在 @torch.no_grad() 内，detach/no_grad 全保留**——§24 禁令）；capture 模式写入 bank 时 `requires_grad_(True)` 成 detached leaf；record 模式发 MergeRecord。
   - 新增 `refresh_leafs()`：backward 后清 leaf 梯度并重建（leaf 仅单步存在，无跨步图）。
   - `MemoryVLA.__init__` 增 `use_rpbe_gamma/gamma_rank/gamma_alpha_init/rpbe_merge_records/rpbe_task_grad`；**只传给 cog_mem_bank**，`PerMemBank` 恒官方 average（§4、§31.10——perceptual 分支第一阶段不动）。
3. **新文件 `vla/datasets/hdf5_dataset.py`**：HDF5 DecisionStream（见下）。不 import TF 链。

### C. 新训练脚本 `train_libero_mem_rpbe.py`（单卡，不碰官方 train.py）+ `scripts/diag_rpbe_embodied.py`（λ 校准）+ `scripts/eval_libero_mem.py`（评测）

## Γ 与 dual 数学要点

### 监督协议（causal cut，不可违反——审阅裁决 1）

第 d 个 policy decision 处理完后发生 merge `(m_i,m_j)→z_d`，`τ_v = d` 为 **post-decision memory update time**（本次 consolidation 真正能影响的是后续 decision）。监督**只能**取：
- `Y_v⁽¹⁾ = A_{τ+1}`（下一个 policy decision 的 action chunk，112 维归一化）
- `Y_v⁽²⁾ = A_{τ+2}`

**Valid RPBE cut ⟺ Y⁽¹⁾、Y⁽²⁾ 同时存在**。`τ+2 ≥ T_episode` 的 cut 标记 **censored**：不进入 RPBE window（仍正常参与 task loss）。`PendingMergeQueue` 机器化断言 `d(Y) > τ_v`；缺任一 future 的 merge 在 episode drain 时丢弃并计数（censor 计数器进诊断输出）。不实现"缺 horizon 省略、有一条算一条"的宽松语义。

### Γ 插入的梯度边界（§23/§24/§25）
- bank 内 Γ 只做推理（no_grad 保留）；Γ 的**全部梯度来自训练脚本的 replay**。
- **Task 路径**：z_v 以 detached leaf 写入 bank → 后续 retrieval 真实参与 action loss → backward 后读 `g_v^task = leaf.grad`。图只有单步 `[retrieval→fusion→DiT→L_action]`。
- **RPBE 路径**：window close 时 detach 的 Z 临时变 leaf → `dual_latent_z_adjoint` → `g_v^RPBE` → replay `Γ(m_a^det, m_b^det)`。图只有 `[m_a,m_b→Γ→ẑ]`。
- 总 Γ 损失：`L̃_Γ = L̃_task + λ_rpbe·L̃_rpbe`，其中 `L̃_task=Σ⟨sg(g_task), ẑ⟩`（最小化）、`L̃_rpbe=−Σ⟨sg(g_rpbe), ẑ⟩`（最大化 J）。

### dual_full_score（§17-19，O(N²) 代替 O(4096²)）
1. 加权中心化 + 簇自由度 D=W−W2_cut/W（与 WeightedWelford 同契约，fp64）。
2. `s_Z = Σw‖zc‖²/(D·4096)`（= mean diag C_ZZ，**不构造 4096²**）；`X̃=zc/√(D·s_Z)`、`Q̃=pc/√(D·s_P)` → 与特征空间 scale-normalized 目标精确等价。
3. `K_Z=X̃X̃ᵀ`、`K_P=Q̃Q̃ᵀ`（N×N）；`H=K@cholesky_solve(I, chol(K+εI))`（无显式 inverse）。
4. `J_dual = tr(H_Z·H_P)` ≡ ‖A^{-1/2}CB^{-1/2}‖_F²。
5. 失败契约同 `_score_from_covs`（cholesky 失败 → 可微零 + diag；strict 抛异常）。
- `diag_score` 变体：只存 `diag(C_ZZ)` [4096] + C_ZP [4096×64]，全程 O(4096·m)。
- **§20 门禁测试**（`test/test_rpbe_embodied_loss.py::TestDualExactEquivalence`）：N=20/d_z=32/m=16 随机矩阵 × 4 配置（等权/双 horizon 簇/零权行/大 DC 偏置），对照 `rpbe.loss._score_from_covs` 的特征空间路径，断言 `|J_feature−J_dual|≤1e-8(1+|J|)` 且 `∇_Z` allclose(atol/rtol 1e-6)；另加 gradcheck。**不过禁止训练**。

## 训练步编排（§26/§27 参数版本一致性）

**两个概念显式拆分（审阅裁决 3）**：
- **repr update window**（三臂完全一致的时间表）：`opt_repr`（LoRA + Γ）只在 macro boundary 时 step。**Gamma-Task 臂没有 RPBE loss，但必须有相同的 repr update window**——否则 Γ_task 只积梯度不同频率 step。三臂 macro boundary 定义相同（每 N_merges 个有效 merge 或 episode 边界，配置里写死同一个 N）。
- **RPBE statistics window**（仅 Gamma-RPBE 臂）：薄行收集 → close 时 dual adjoint。窗口门禁独立于 repr window。

```
每 epoch：shuffle episodes → 整条 episode 进 → 按 ≤16 行切 chunk（bank 跨 chunk 保持，stream 语义）
  for chunk:
    autocast(bf16): loss_action = vla(...) → backward（task 优化器梯度）
    读 merge_log → PendingMergeQueue.register（记 param_version）
    读 leaf grads → (merge_id, g_task)；offer futures（context=冻结 vision_feats 均值+Δs+h+instruction hash，outcome=actions 归一化 112 维）
    Γ task replay backward（Γ.grad 累积）
    opt_task（retrieval/gate/per_compr/DiT）step + clip 1.0
    cog_mem_bank.refresh_leafs()
  episode 边界：drain_episode 回填 tree_weight=1/n_merges；censored cut 计数
    （仅 RPBE 臂）window.add(valid rows)
    repr window 达边界时（三臂同判据）：opt_repr step + clip 1.0；param_version += 1
    （仅 RPBE 臂）RPBE window.ready() 时 close → dual_latent_z_adjoint → λ_rpbe·L̃_rpbe backward（Γ.grad 累积，随下一个 repr boundary step）
```
- 双优化器：`opt_task`（retrieval/gate/per_compr/DiT）每步；`opt_repr`（LoRA+Γ）只在 repr macro boundary step——window 内 LoRA/Γ 版本固定（§26/§27，merge 的 param_version 断言机器化检查）。
- `update_fused=False` 是前提（bank 存 raw token，只有 LoRA+Γ 决定 cognitive 内容）——官方默认，不改。
- **Replay gradient 门禁（审阅裁决 5，Task 8 验收）**：tiny synthetic merge 上允许直接建图 `L(Γ(m_a,m_b))` backward 得 `g_Γ^direct`，再走正式 leaf/replay 路径得 `g_Γ^replay`；要求 `cos(g^direct, g^replay) > 0.99999` 且相对误差 ~0。task 路径与 RPBE 路径各做一遍。

## HDF5 DecisionStream（§8/§9 的落地）

- `HDF5DecisionStreamDataset`：启动扫描 h5+metainfo → episode 清单；`decision_stride = future_action_window_size+1 = 16`；每 episode 输出 decision d=0,1,…（帧 k=d·16，监督 `actions[k:k+16]`），尾部残缺 decision 丢弃并计数；`timesteps=decision 索引`（对齐 predict_action 的 cur_timestep 语义）；episode_ids 填充使 stream 语义正确。
- `HDF5BatchTransform`：复刻 RLDSBatchTransform 的 prompt/labels 逻辑 + image_transform + **动作 BOUNDS_Q99 归一化（numpy 复刻）** + 同构 dataset_statistics（q01/q99/mask）。
- collator 复用 `PaddedCollatorForActionPrediction`。

## 三臂与 λ 校准

| 臂 | gamma | task grad | RPBE window | repr update window |
|----|-------|-----------|-------------|--------------------|
| MemoryVLA-Avg（host-matched control） | ✗ | ✗ | ✗ | ✗（只有 opt_task） |
| MemoryVLA-Gamma-Task | ✓ | ✓ | ✗ | ✓（与 RPBE 臂**同边界同频率**） |
| MemoryVLA-Gamma-RPBE（full_dual 主 / diag 变体） | ✓ | ✓ | ✓ | ✓ |

- Γ 更新时钟（审阅裁决 3）：`g_Γ = g_task`（Task 臂）/ `g_Γ = g_task + λ·g_rpbe`（RPBE 臂），在**完全相同的 macro boundary** step——三臂 repr 时间表逐位一致。

- 公共协议：LoRA r=32/lora_alpha=32/all-linear 挂 `vla.vlm.llm_backbone.llm`（**顺序：load → freeze 基座 → attach → requires_grad 审计 → 不再调 freeze_backbones()**）；BF16、clip 1.0、AdamW lr 2e-5（cosine+warmup 100）、repeated_diffusion_steps=4、mem_length=16、tome、image_aug、seeds 42-46（5 seeds × 3 臂）。
- max_steps 候选 40k（官方值）；**Go/No-Go：实测 >2 周/run 则三臂同步降 20k**。
- checkpoint：只存 trainable（LoRA 只存 lora_* 键，不含 13G 冻结权重）+ lora_config + param_version。
- λ 校准（Task 8 后，CCM 规则迁移、常数不复用）：θ₀ 下 λ=1 收集 4-8 windows 测 `r_eff=‖∇Γ(λ·L̃_rpbe)‖/‖∇Γ L̃_task‖` → `λ_rpbe = target_r_eff/median(r_eff(λ=1))`，target 0.15（∈[0.05,0.30]），校验 p95<1 后写死进 config，5 seeds 共用。

## 实现顺序与验收

| Task | 内容 | 验收 |
|------|------|------|
| 0 | 磁盘扩容 + 下载 LIBERO-Mem + probe | probe JSON 六项齐、train/val 无重叠、oracle 键隔离审计过 |
| 0.5 | **checkpoint integrity audit**（审阅裁决 5） | missing/unexpected keys 清单；action_model 缺失定性（故意/缺陷）+ 实验定位措辞锁定；memory 模块权重来源确认；LoRA 前后 hash；官方 eval 1 task 非随机 |
| 1 | HDF5 loader + LoRA attach + 单卡烟测（100 steps，无 RPBE） | loss 降、显存 <80G、ckpt round-trip、predict_action 烟测 |
| 2 | Decision Stream | 合成 h5 单测：timesteps=0,1,2…、chunk 与帧对齐、尾部丢弃计数；**出压缩样本统计表**（审阅裁决 4：decisions/merges/valid 2Obs cuts per episode、depth 分布、% 合并态参与 retrieval）——cuts/episode ≤2 则调整 stride/budget 重测 |
| 3 | Merge tracing（官方 avg 下 record_merges） | 单测：merge 树可由 left/right id 重建、depth/start/end 正确 |
| 4 | Γ 插入 | 起点 ‖Γ−avg‖<1e-6；perceptual bank 逐位仍 avg；forward 烟测 |
| 5 | Future queue | 单测：strict `d(Y)>τ_v`、h=1/2 分派、**缺任一 future → censored 计数**（裁决 1）、drain 权重归一 |
| 6 | Fixed embodied maps | LoRA 扰动前后 P 逐位不变；pv==pv_batch；oracle 键 grep 审计 |
| 7 | 高维统计（diag → dual → 窗口） | **§20 门禁全绿**；D/W2_cut 与 WeightedWelford 对照 |
| 8 | 梯度回路 + 双优化器（repr/RPBE 双窗口）+ λ 校准 | bank 无跨步图断言；param_version 一致性断言；**direct-vs-replay 梯度门禁 cos>0.99999（双路径）**；r_eff 报告落带 |
| 9 | 三臂 × 5 seeds + val 评测 | per-task subgoal 完成率 mean±std；Avg 臂为 host-matched control；**Fixed-trace 诊断**（裁决 2：held-out 子集三臂强制同 pair sequence）；**compression-pressure 曲线 L=4,8,16**（裁决 4：小 budget ⇒ RPBE advantage 增大） |

## 风险

| 风险 | 缓解 |
|------|------|
| 磁盘 6G vs 19.9G 数据集 | Task 0 用户扩容（阻塞项） |
| HDF5 划分/dtype 未文档化 | probe 先行；泄漏核查 |
| 单卡 7B 显存/吞吐 | grad ckpt + chunk 16→8；Go/No-Go 降 steps（三臂同步） |
| LIBERO 仿真评测（robosuite/mujoco × py3.12） | 备选：独立 py3.10 环境，或 allenai/vla-evaluation-harness 的 libero-mem Docker 镜像（官方协议） |
| Γ task-replay 与 action loss 双重作用 | Gamma-Task 臂本身是隔离对照；监控 r_eff 曲线 |
| dual 实现数值漂移 | §20 门禁 + fp64 窗口组装 + DC 偏置用例（TGN 教训） |
| **valid RPBE cuts/episode 过少（≤2）**（审阅裁决 4） | Task 2 统计表先行；调整 decision stride / mem_length / 窗口门禁（kf_min_abs）；L=4,8,16 pressure 曲线作机制证据 |
| **action expert 随机初始化淹没 Γ 信号**（审阅裁决 5） | Task 0.5 定性锁定；若随机为官方 checkpoint 事实，报告按 matched-initialization 口径撰写，并监控三臂 DiT 收敛速度 |
| **repr update clock 不一致**（审阅裁决 3） | repr window 判据单一来源（config 写死），三臂共享同一宏边界函数，单测断言边界序列一致 |

## 验证方式

1. Task 1：烟测 loss 下降、显存峰值记录、ckpt round-trip、predict_action 与官方代码路径一致
2. Task 7：§20 等价门禁测试全绿（J 与 ∇_Z 双等价，含 diag 变体）
3. Task 8：bank 无跨步图、window 内 param_version 一致、r_eff 校准带 [0.05,0.30]
4. Task 9：三臂 per-task 完成率表 + 5-seed mean±std + Avg 臂 host-matched 对照；LIBERO-Mem 论文数字作外部引用
5. 全程：本机 `third_party/memoryvla` 与服务器同步；vendored 修改仅 3 处 patch + 本计划文件清单，commit 记录

时间估算（单卡串行）：Task 0 约 2 天；Task 1-3 约 1 周；Task 4-6 约 1 周；Task 7-8 约 1.5-2 周（门禁是最大不确定性）；Task 9 约 3-4 周（15 runs）。总计约 7-9 周。
