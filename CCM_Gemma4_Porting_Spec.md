# CCM × RPBE 移植到 Gemma-4-E4B：实现规格（feature_GEMMA，2026-09-20）

## 0. 决策记录（用户已拍板）

1. 宿主 = **Gemma-4-E4B base**（`google/gemma-4-E4B`，非 -it）——与 llama-7b-hf / qwen3 base 对齐。
2. 任务线 = **DailyDialog 1000 步**（LaMP 线已证伪抛弃）。严格按官方两阶段流程：
   - Stage-1：default LoRA 微调（Table 13：1000 步、batch 128、lr 3e-4、cosine；Table 14：q/k/v/o_proj、r8、α16、dropout 0.05）→ merge 进权重
   - Stage-2：压缩训练（conditional LoRA + COMP embeddings；DailyDialog COMP=2、T=12）
3. macro-batch 可调高（用户批准）：micro 2 × accum 128 = macro 256（官方 2 倍），lr 保持 3e-4。
4. 服务器 = 36.139.27.63:22（双 A100-PCIE-40GB，125G 内存，168G 磁盘），裸机新装环境。

## 1. 环境（大版本迁移，gemma 线最大风险源）

| 包 | llama/qwen3 线（4.x 栈） | gemma 线（5.x 栈） | 说明 |
|---|---|---|---|
| transformers | 4.56.2 | **5.17.0** | gemma4 需要 ≥5.5；4.x 不注册 gemma4 |
| peft | 0.4.0 | latest | peft_custom 依赖 peft.tuners/utils 内部 API，需兼容验证 |
| torch | 2.8.0+cu128 | **2.10.0+cu128**（aliyun flat index，手动 wget） | aliyun pytorch-wheels 是导航页非 PEP503，pip 26 解析不了 `&#43;` 转义 |
| python | 3.10（conda） | 3.10.12（系统） | 清华 conda 源已删，跳过 conda |

**5.x 兼容点待验证**（装完立即 smoke）：
- peft_custom（`third_party/ccm/src/peft_custom/`）对 peft latest 的 API 依赖（peft.tuners / peft.utils / import_utils）
- 官方 Step-1 入口 `src/train.py`（HF Trainer 版）在 transformers 5.x 的 API
- llama/qwen3 host 代码在 5.x 下 import 不炸（gemma 服务器上不跑，但 import 链共享）

## 2. Gemma-4-E4B 架构对位（transformers 5.17 modeling_gemma4.py 已研读）

规格：42 层、hidden 2560、FFN 10240、8 Q heads / 2 KV heads、262K vocab、
**5:1 hybrid attention**（5 滑窗层 512 + 1 全局层，每 6 层 1 个全局，最后一层强制全局）、
**KV sharing**（layer 24-41 无 k/v_proj，复用 layer 22/23 的 KV，按 layer_type 分桶）、
**PLE**（每层 256 维 per-layer input，token-identity 查表 + context 投影合成）、
滑窗层 head_dim 256 / 全局层 512、p-RoPE（全局层）、final_logit_softcapping。

| CCM 机制 | gemma4 落点 |
|---|---|
| conditional LoRA（LinearMask q/k/v/o） | 非共享层全挂；**共享层（24-41）只有 q/o**（无 k/v 投影可挂） |
| SUM merge（post-RoPE 加权平均） | 非共享层 RoPE 后、`store_full_length_kv` 前；**共享层自动继承**（KV 复用） |
| 可见性 mask | full_attention + sliding_attention **双 mask dict** 各自叠加（物化 4D additive） |
| Γ 递归 + mem_callback | 同 qwen3 落点（merge 之后、attention 之前） |
| q/k/v norm | q_norm/k_norm 后 RoPE；merge 消费 post-norm+RoPE 状态 ✓ 与 llama 同构 |
| heads×head_dim ≠ hidden | 滑窗 8×256=2048、全局 8×512=4096 vs hidden 2560 —— qwen3 断言删除教训直接适用 |
| PLE | 冻结；**双表 resize**：embed_tokens 262144→262148（comp 行可训）+ embed_tokens_per_layer 262144→262148（新行零冻结） |
| sliding window 512 | DailyDialog 序列 ~200-300 token < 512，滑窗无信息损失；mask 物化时保留滑窗语义 |
| softcapping | loss 与评估 logits 同用 softcap 后结果，口径一致 |
| use_cache | 全序列模式（qwen3 v1 同款，无 cache） |

## 3. 代码改动清单

1. **新 `third_party/ccm/src/arch/ccm_gemma4.py`**（qwen3 复制改写路线，~800 行量级）：
   - `Gemma4CCMTextAttention`：复制 Gemma4TextAttention + 4 个 CCM delta（LinearMask、SUM overwrite、Γ 递归、mem_callback）；手写 eager attention（GQA repeat + additive 4D mask + fp32 softmax）
   - `Gemma4CCMTextDecoderLayer` / `Gemma4CCMTextModel`：mask dict 物化 + CCM 可见性叠加；PLE 管线原样保留
   - `Gemma4ForCausalLM_CCM`：CCM 输入处理（comp ids → 位置 → masks → merge）
2. **新 `src/data/dialogue/gemma4_data.py`**：DailyDialog 协议（T=12、COMP=2），gemma4 tokenizer 对位（bos=2/eos=1/pad 手动、comp ids 262144-262147）
3. `scripts/train_ccm.py` + `scripts/eval_ccm_depth.py`：`--host gemma4`
4. `path_config.py`/`run.py`：gemma-4-E4B 条目（官方 Step-1 入口）
5. peft_custom 5.x 适配（按 smoke 结果）
6. 权重加载：`Gemma4ForConditionalGeneration` 全量加载 → 取 text 部分；或 `Gemma4TextConfig.from_pretrained` + 过滤 state_dict

## 4. 验收门禁（照 qwen3 先例，核心语义不可省）

- **G1** 无压缩 parity：comp/sum 关闭时 forward 与原模型逐位一致（2e-7 量级）
- **G2** SUM mask 源正确性 + 行归一化（1/2/2 源分布）
- **G3** Γ 零初始化 step0 identity（loss 一致）
- 数据协议块结构验证（train/val/test 数量、comp 插入位置、target 对位）
- 继承 ccm-line-state 教训：EOS 排除口径、每 checkpoint 评估、同口径对比纪律

## 5. 训练计划

- Stage-1（default LoRA，官方入口 train.py 或等价自写）：1000 步、micro 2 × accum 128、lr 3e-4 cosine、bf16（gemma 权重原生 bf16，官方 FP16 为 llama 特定）、q/k/v/o_proj r8/α16/dropout0.05 → merge 成 foundation
- Stage-2（conditional LoRA + COMP，train_ccm.py --host gemma4）：同配方 1000 步
- RPBE：DailyDialog 线协议沿用（dialogue_records、λ 校准规则迁移、常数不复用）
- 评估：协议 A/B/C（eval_ccm_depth.py），官方 merge 基线 vs ours

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| transformers 5.x 大版本破坏 peft_custom/官方 train.py | 装完立即 smoke；peft_custom 必要时 vendor 完整 peft 0.4 逻辑（去掉库依赖） |
| 共享层无 k/v 投影 → LoRA 覆盖不全 | 这是 gemma 架构事实；Table 14 的 q/k/v/o 目标在共享层退化为 q/o，实验记录即可（论文写明） |
| E4B base 仓库名分歧（-pt vs 无后缀） | modelscope/hf API 实际列出后确定 |
| 8B total 权重 + 40GB 卡训练显存 | bf16 + grad checkpointing；micro-batch 2 起步，OOM 降 1 |
| PLE comp 行语义 | 第一阶段冻结零行（= 官方 llama 无 PLE 语义的忠实移植）；若效果差再议 |
