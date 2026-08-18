# PRSS2 上下文交接备忘(迁移会话前必读)

> 用途:换新会话 / 换 AI 协作时,把本文档全文粘贴作为初始上下文。
> 更新日期:2026-08-18

---

## 1. 项目一句话

**PRSS**(Predictive Relation-State Sheaf / Predictive Recursive State Sheaf)= 寄生在时序模型
递归接口上的预测性谱压缩方法:候选 h(256 维)→ 商矩阵 R(k×d,k=宿主宽度)→ z=R·h,
父节点只收 z;R 由"未来读取矩阵 B(C)"的预测 Gram `G=E[BᵀB]` 的 top-k 特征分解周期更新。
训练损失 = L_task + λ_resp·L_resp + λ_spec·L_spec;reader/outside/Gram/SVD 仅训练期存在。

**当前目标(甲方要求)**:多数据集泛化证据——"必须在多个数据集上都拿到最佳,或者都寄生了都有效"。
载体:v2.0 新架构 + TGB 标准协议(tgbl-wiki 链路预测 mrr)。

## 2. 目录状态(2026-08-18)

```
PRSS2/                            # git 仓库(尚无提交!)
├── DEVELOPMENT v2.0.md           # ★当前活跃实现开发日志(先读它)
├── notes/                        # ★文档区
│   ├── PRSS_method_spec_v1_2.md  # 方法规格书(理论权威,§33 禁止事项是红线)
│   ├── PROJECT_INDEX.md          # 研究索引(历史版,描述已删的 old/)
│   ├── DEVELOPMENT_PRSS2.md      # 理论对照调研总录(A/B/C/D 清单,历史版)
│   ├── ABC_FINDINGS_EXPLAINED.md # 25 条问题非专业版详解(历史版)
│   └── CONTEXT_HANDOFF.md        # 本文件
├── src/prss/                     # ★新架构核心包(τ 索引,零宿主硬编码)
│   ├── compressors/              # 可插拔压缩器 5 变体:vanilla/random/pca/direct/spectral
│   ├── hosts/                    # PyG-TGN 宿主适配 + TGB 链路桥 + vendor 基线(pyg_models/)
│   ├── data/ training/ experiment/ monitoring.py spectral.py ...
├── scripts/{train,inference}.py  # 入口(--variant/--dataset/--seed/--output/...)
├── test/                         # 全部 unittest.TestCase 类(68 项)
├── configs/experiments/          # tgbl_wiki_smoke.yaml / tgbl_wiki_matrix.yaml
├── datasets/  outputs/           # 数据与实验输出(gitignore 已排除)
└── (old/ 已删除,git 未提交过,无法恢复)
```

## 3. 五阶段状态(如实)

| 阶段 | 状态 | 验证证据 |
|:---|:---|:---|
| 1 骨架+核心移植 | ✅ | 40 项测试已实测全绿 |
| 2 可插拔压缩器 5 变体 | ✅ | 61 项已实测全绿(含合成树门禁 ≤0.20 rad) |
| 3 TGB 接入 | ✅代码完成 | 3 个 PyG 宿主冒烟测试**各自单独通过**;修复时间戳 dtype 后未做完整三连(遵用户要求) |
| 4 实验框架 | ✅ | 4 项已实测全绿 |
| 5 DEVELOPMENT v2.0.md | ✅ | 已撰写 |

**从未执行**:真实 tgbl-wiki 训练、数据集下载(tgb 包未装)、runner 跑真实任务。

## 4. 关键决策记录(已拍板,勿偏离)

- TGB 接入**先只做 tgbl-wiki 链接预测(mrr)单数据集**;数据集维度已参数化(换 name 即可铺开)。
- 压缩器第一批 = **核心 5 变体**(vanilla/random/pca/direct/spectral);规格要求的
  linear_reader_svd / no_nonlinear_lift / Jacobian-SVD 未做。
- 实验配置 = **纯标准库 YAML**(不用 Hydra)。
- 宿主 = **PyG 版 TGN**(TGNMemory + GraphAttentionEmbedding 单层子图卷积),
  τ=`tgp:node_conv`,host_dim=emb_dim(100),候选 256,outside 树深度=1;
  twitter-TGN 宿主迁移 deferred(hosts/base.py 抽象已预留)。
- old/ 归档版结果(官方锚点 0.8933 / vanilla 0.8855 / PRSS 0.9001,节点分类 AUC)
  是**历史数字**,与新架构的 mrr 协议不可直接对比。

## 5. 红线(不可破坏)

1. 压缩从训练第 0 步开始;禁止先训后压缩 / root-to-leaf 后处理(规格 §33 全清单在规格书末尾)。
2. spectral 是唯一主方法谱求解;PCA/random/direct 只能作为**命名变体**存在。
3. R 是 buffer(非参数,除 direct 变体);R=[I,0] 初始化保证与宿主数值平价。
4. reader/outside/Gram/SVD 仅训练期;验证/测试三零隔离审计(`training/isolation.py`)。
5. 核心包禁止宿主硬编码(test 断言 core.py 无 "tgn_layer_" 等)。
6. vendor 的 pyg_models 有两个**注明文档的本地补丁**(msg_agg.py 的 scatter_reduce 实现、
   memory_module.py:176 的 int64 保持),勿当上游 bug 改掉。

## 6. 注意事项(用户明确要求,务必遵守)

1. **千万不要跑全量测试**(`pytest test`)——用户明确禁止,因合成树/TGB 测试含训练循环耗时;
   任何测试/训练执行前先征得同意。
2. **任何真实训练/下载数据集前必须先确认**。
3. 本机环境:Windows,torch 2.10.0 CPU(无 CUDA)、PyG 2.8 已装(仅 torch-geometric)、
   **tgb 包未装**;有 numpy DLL 警告(Anaconda env,暂时非致命,报 numpy 错误时
   `pip install numpy==1.26.4 --force-reinstall`)。
4. 数据放置:TGB 数据放 `datasets/`(默认 TGB_ROOT),关键文件
   `datasets/tgbl-wiki_edgelist_v2.csv` 存在即跳过下载;JODIE 的 reddit.csv 属于旧体系,
   新架构不读。
5. 沟通语言:中文回复;用户是工程背景、ML 初学者,讲解要用白话+类比。
6. 用户是"传话人",重大方向决策(数据集范围、协议选择)由甲方(ultraviolence)拍板。

## 7. 下一步(按优先级)

1. **AutoDL GPU**:装 `tgb==2.3.0` → 跑 `configs/experiments/tgbl_wiki_smoke.yaml`
   (1 epoch×1 seed×5 变体)→ 记录 vanilla 锚点 MRR(对照 TGB 官方基线)→ 跑正式矩阵
   (30 epoch×3 seed×5 变体)→ `summarize` 出表。
2. λ_spec 敏感性 {0, 0.01, 0.1, 0.5, 1.0}(矩阵加一维即可)。
3. tgbl-uci / tgbl-enron 铺开(enron 3054 万边,算力评估)。
4. twitter-TGN 宿主适配迁移 + 平价测试(从规格书与 DEVELOPMENT v2.0.md §6 的语义差异表入手)。
5. git 首次提交(仓库尚无 commit;.gitignore 已配好,验证过 422 文件入库、产物零泄漏)。

## 8. 历史研究结论速览(old/ 归档版调研,供论文写作引用)

- 理论对照:核心数学对象与因果红线忠实;不严谨处 = 阻尼谱部署(A1)、trace 采样正样本优先(B1)、
  基线组/机制诊断缺失(C 类)。详见 `notes/DEVELOPMENT_PRSS2.md` 与 `notes/ABC_FINDINGS_EXPLAINED.md`。
- TGN 论文官方基线(已下载 PDF 在 old/papers/ 时被删,可重下 arXiv:2006.10637):
  链接预测 AP wiki transductive 98.46±0.1 / inductive 97.81±0.1;节点分类 AUC wiki 87.81±0.3。
