# TGB tgbl-wiki 递归闭合实验 进度记录（2026-09-07 云端 session）

## 目标
数据换 TGB 2.0 tgbl-wiki link prediction + 官方递归 Twitter-TGN + 真实
child–parent boundary，完成 1Obs/2Obs-aligned/2Obs-mispaired/task-only 四臂消融。

## 关键语义（已钉死，勿偏离）
- child occurrence (node,time) 是父聚合实际消费的 NEIGHBOR child，query_time = 父
  时刻 repeated_times，**禁改成 edge_time**（否则 z 泄漏 edge_time..t 信息）。
  relation_time(=edge_time) 单独存且 ≤ query_time。
- 2Obs: Y_v^(2) := Y_p(v)^(1)，经 exact parent occurrence 查询，**禁同 node 第二
  时间观测代理**。
- C_v 只含 cut 时已知结构（layer/relation lag/slot/path/root role），禁 future 字段。
- 每 pair 一条 (z,p) row；每树总权重 1；负样本只进 root link loss，禁进 PRBE。
- 四臂共享同一 surviving cut/pair 候选集；mispaired 用约束完美置换只动 parent
  future，unchanged=0，donor parent future.time > receiver.parent_time。

## 步骤完成状态
1. ✅ tgbl-wiki 数据层 (src/rpbe/data/tgb_link.py) — 服务器 gate 过
2. ✅ bounded consumed child-parent trace (jodie_tgn._compute + ConsumedPairCandidate)
3. ✅ LinkFutureIndex + BoundaryRecord (src/rpbe/link_records.py)
4. ✅ BoundaryMaps 联合 S_v 固定映射 (src/rpbe/pair_maps.py)
5. ✅ pair_arms mispaired + shared-cut gate
6. ✅ PairKFWindow exact-replay (pair_window.py)
7. ✅ TGBPairLinkLoop + train_tgb_link.py + eval_tgb_link.py + run_tgbl_wiki_recursive_abl.sh
   — 四臂 smoke 服务器全过（窗口关闭/aux grad/best.pt 落盘）
8. ⏳ GPU 门禁报告 — smoke 已隐含通过；eval MRR 正在验证
9. ⬜ 5-seed 正式运行 + 交付报告

## 服务器 smoke 结果（epoch0，24 batch）
| arm | link_loss | closed | below | aux_batches |
|---|---|---|---|---|
| gamma_task_only | 1.261 | 0 | 0 | 0 |
| 1obs | 1.264 | 6 | 0 | 24 |
| 2obs_aligned | 1.262 | 6 | 0 | 24 |
| 2obs_mispaired | 1.264 | 6 | 0 | 24 |

## 已修集成问题（服务器 smoke 暴露）
- TGN 构造需 numpy 特征（非 Tensor）
- ctx vector CPU generator + CUDA device 冲突 → CPU 造完 .to(device)
- message 取到 maps buffer device
- 服务器 loss.py 需同步 variant 透传（7a6807c 本地方案）

## git
- 分支 feature_abl，已推送至 a70b4e3 等（步骤1-7 对应提交）
- 本地 201 passed；服务器 193 passed（含 tgb 数据测试）

## 下一步
- 确认 eval_tgb_link.py MRR 在 smoke best.pt 上跑通
- 写 gate_recursive_pairs.py / audit_aligned_pairs.py（步骤8 明确门禁）
- 跑 5 seed 正式四臂
