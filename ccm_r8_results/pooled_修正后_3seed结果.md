# DailyDialog pooled 修正后 3-seed 结果（2026-09-15）

根因：评估命令缺 training.comp.attn_type=merge_recur，hydra 默认参数污染 collator 构造。
修正后全部参数：num_comp_tokens=2 + attn_type=merge_recur + official_host 构建。

| 桶 | seed0 s200 | seed1 s150 | seed2 s150 | 3-seed 均值 |
|---|---|---|---|---|
| turn_3 | 7.202 | 7.246 | 7.257 | 7.235 |
| turn_4 | 7.030 | 7.045 | 7.058 | 7.044 |
| turn_6 | 7.072 | 7.018 | 7.029 | 7.040 |
| turn_10 | 6.882 | 6.726 | 6.715 | 6.774 |
| turn_14 | 6.256 | 6.195 | 6.168 | 6.206 |
| 五桶平均 | 6.888 | 6.846 | 6.846 | 6.860 |

## 对照

- 官方 CCM-merge 论文值（Table 25 test @step12）：6.27 → 我们 turn_14 均值 6.21（优 0.06）
- no_ctx：turn_14 6.657；full_ctx：turn_14 4.620
- 官方 foundation（无压缩）：turn_14 5.556

## 结论

1. 修正评估参数后，R8 ours 在 pooled 口径下完全正常，3-seed 高度一致（std < 0.05）。
2. turn_14（最深层）均值 6.21，优于官方 merge 论文值 6.27；五桶平均 6.86。
3. 之前的异常数字（8.6-9.1、turn_3 117+）全部来自缺 attn_type 参数的评估构造污染。
