# 多尺度 MRGR：学习率 4e-4 正式 300 轮训练完成记录

完成时间：2026-10-18 03:00:23（北京时间）。GPU2 完成全部 **300 轮 / 24,900 次更新**，实耗 **2.8023 小时**，含验证与保存；预热后每轮 33.536 秒。

结构保持 344,678 参数的多尺度 MRGR；仅 AdamW 初始学习率由 2e-4 提至 4e-4，其余原损失权重（含一致性 .05）、批量、裁剪、归一化、P96 投影、300轮余弦调度均保持。种子 20261011 的初始权重、166训练/36验证案例顺序及全部300轮站位/打乱记录与旧多尺度和新控制匹配。

最佳普通 H 投影相对 L1：**2.977327%（第 291 轮）**；第300轮为 2.979932%。新控制的最佳为3.174475%，旧多尺度为3.109370%；本次最佳分别降低约 **6.21% / 4.25%**。这些是投影测量指标，三维重建效果以主 agent 统一冻结评价为准。

## 相同训练轮数的学习动态

下表均为相同 epoch 的普通 H 投影相对 L1，括号为截至该轮最优值；记录不能替代三维检验。

| Epoch | 旧多尺度 | 新控制 | LR4e-4 |
|---|---:|---:|---:|
| 50 | 5.7690% (5.6924%) | 5.9330% (5.8720%) | 5.8314% (5.8233%) |
| 100 | 4.4490% (4.3288%) | 4.5249% (4.3401%) | 4.4190% (4.2686%) |
| 200 | 3.3935% (3.3809%) | 3.4123% (3.3764%) | 3.2242% (3.1825%) |
| 300 | 3.1203% (3.1094%) | 3.1763% (3.1745%) | 2.9799% (2.9773%) |

## 审计与产物

正式 runner 的源文件、数据与筛查证据验证通过。独立 CPU 审计通过：300轮与24,900更新完整、学习率符合解析余弦、损失与原目标代数一致、全部轮数站位及loader RNG匹配、最佳权重选择和best/last一致性正确、12个每25轮快照有限、末态优化器/调度器正确。训练与本子任务审计没有读取三维GT、GT128或最终35例测试集。

[正式晋级决定](../../outputs/research/mrgr_autonomous_optimization_v2/lr4e4_formal_decision.json)、[协议](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/protocol.json)、[完成摘要](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/summary.json)、[独立训练审计](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/independent_training_verification.json)、[相同初态审计](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/matched_initialization_audit.json)、[筛查完整报告](mrgr_parameter_optimization_v2_zh.md)。

![完整300轮曲线](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/curves.png)


主agent已完成统一三维评价，详见[300轮冻结重建测试与最终判读](mrgr_autonomous_optimization_v2_formal_test_zh.md)。候选聚簇布局明显改善，普通布局未满足预登记整体收益门槛。
