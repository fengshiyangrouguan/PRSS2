# 协议偏差记录：COMP/SUM 输入嵌入的 trainable 状态

日期：2026-09-17。记录者：Qwen3 移植线（feature_QWEN）。

## 事实

| 协议 | COMP/SUM 输入嵌入 | 出处 |
|---|---|---|
| **官方 CCM 论文/代码** | **trainable**——SeparatedEmbedding.comp_embeddings 独立可训练（官方专门设计这 4 行可学习；参数预算 = 条件 LoRA 5,898,240 + comp 10,240 = 5,908,480） | src/utils.py SeparatedEmbedding |
| llama 线 build_official_host（R8 线 ours、官方 merge 评估臂） | trainable ✓ 与官方一致 | scripts/train_ccm.py:496-497 |
| **llama 线 build_model（R10 v5 三臂：ccm_merge / gamma_task_only / ours）** | **冻结**——resize_token_embeddings 追加的随机行，peft wrap 后非 lora 全冻；代码注释自述 "The COMP/SUM rows appended by resize_token_embeddings are frozen"（train_ccm.py:1005） | 未经显式审阅的隐含偏差 |

## 影响

1. R10 v5 的**正式三臂数字**（协议 C s50 平均 PPL 6.740、-1.2% vs merge 等）是在 **comp 冻结**底座上测得的——严格口径上是"官方协议的一个变体"（comp 行不可学习）。
2. R10 自训的 ccm_merge 臂与官方 merge adapter **不同质**（comp 冻结 vs trainable）——四臂表用官方权重评估 CCM-merge 臂的既有决策因此有了协议依据。
3. 论文写作时需在 deviations 中如实披露，或补跑 comp-trainable 版（待用户决策，未排期）。
4. 另：R10 三臂的 lora_dropout=0.0（两遍 replay 精确性要求）vs 官方 LoraConfig 0.05——merge 阶段无 replay 需求，Qwen3 线已按 0.05。

## Qwen3 线的处置（用户 ruling 2026-09-17）

- 新独立入口 `scripts/train_ccm_merge.py`：**严格官方 Step-2 协议**——SeparatedEmbedding（comp trainable）、lora_dropout 0.05、lm_head 不扩展（comp 无输出行）、random_k 截断、shuffle epoch 流、accum 128 对话/步、lr 3e-4 cosine/warmup 3%、grad_clip 1.0、max_steps 1000。
- RPBE 机制（深度分层/窗口/chain-wise cut）完全不进入 merge 阶段；train_ccm.py 保持 RPBE 入口纯净。
- Qwen3 merge 模型将是官方等价物（5,908,480 trainable，与规格文档 §4.1 allowlist 逐位一致）。

## 待办

- [ ] llama 线 comp-trainable 重训（R10 协议 + comp 可训练）——用户后续拍板，未排期
