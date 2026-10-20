# MRGR 追加开发实验结果索引（2026-10-20）

后续已按用户要求完成[35例补充测试](mrgr_extension_test35_v1_zh.md)，包含新增反馈OARI、描述符结构删除和双重简化，并核验复用均匀融合既有测试。反馈OARI相对匹配对照的普通/聚簇三维L1为8.5089%→8.3140% / 31.3236%→29.7930%，峰值误差增加。该补充测试沿用此前已使用的35例，不构成全新独立保留集确认；下文仍为开发集结果，不与测试数值混用。[补充测试验收](artifacts/mrgr_extension_test35_v1/acceptance.json)、[完整原始指标（gzip）](artifacts/mrgr_extension_test35_v1/all_metrics.jsonl.gz)、[导出SHA256清单](artifacts/mrgr_extension_test35_v1/export_manifest.json)。

GCA/可观测性描述符消融、反馈OARI的20轮筛查及300轮延长训练均已完成并通过独立重算。本轮只使用166训练案例和36个此前已使用的开发案例，不读取test投影、test GT64或任何GT128。全部结果为开发集探索性观察，既有35例正式测试与冻结主模型保持原样。

## 反馈OARI：相同300轮预算

三组均从各自20轮last.pt继续到300轮，保留原cosine300、优化器、RNG和历史指标；种子20261011。仅按验证H4选择最佳轮，17布局预测封存后再用开发GT64评分。聚簇五draw先在每个案例内平均。

| 模型 | 普通V4三维L1 | 聚簇V4三维L1 | 聚簇质心误差 | 聚簇峰比绝对误差 |
|---|---:|---:|---:|---:|
| 本轮原始MRGR对照 | 9.4688% | 33.7562% | 12.0359 mm | 0.2576 |
| 反馈加入OARI | 9.0465% | 30.5974% | 9.3359 mm | 0.2667 |
| 同参数零残差OARI | 12.8679% | 35.2059% | 13.2674 mm | 0.3184 |

反馈OARI的L1和聚簇定位改善，峰值存在取舍。单种子、观察20轮开发结果后追加预算，不能作为独立确认或最终测试优越性证据。原始MRGR匹配对照与历史冻结模型是独立运行产物，不混用基线。

[完整300轮报告](mrgr_feedback_oari_long_v1_zh.md)；[20轮筛查及负结果](mrgr_feedback_oari_pilot_v1_zh.md)。

## GCA替换与描述符结构删除

以下为两个固定种子的开发结果描述均值；每个新训练任务完整300轮，A/B复用已核验产物。描述符结构删除与输入置零诊断并非相同实验。

| 模型 | 普通V4三维L1 | 聚簇V4三维L1 |
|---|---:|---:|
| A 原始MRGR | 9.5729% | 35.3603% |
| B 有效视角均匀融合 | 9.5431% | 30.6883% |
| C 删除可观测性描述符 | 9.5065% | 34.7597% |
| D 均匀融合＋删除描述符 | 9.4373% | 30.7706% |

均匀融合在聚簇布局的改善较明显；单独删除描述符收益较小。案例配对统计、交互效应、全部指标、成本与固定切面见[完整消融报告](mrgr_gca_uniform_observability_pruning_v1_zh.md)。两种子均值及案例bootstrap不代表训练种子总体不确定性。

## 证据与代码

- [消融证据](artifacts/gca_pruning_v1/summary.json)、[逐例CSV](artifacts/gca_pruning_v1/per_case_metrics.csv)、[配对统计](artifacts/gca_pruning_v1/paired_statistics.csv)。
- [OARI 300轮统计](artifacts/feedback_oari_long_v1/summary.json)、[逐例CSV](artifacts/feedback_oari_long_v1/per_case_metrics.csv)、[独立重算](artifacts/feedback_oari_long_v1/independent_verification.json)。
- [OARI 20轮统计](artifacts/feedback_oari_pilot_v1/summary.json)、[逐例CSV](artifacts/feedback_oari_pilot_v1/per_case_metrics.csv)。
- [导出SHA256及来源清单](artifacts/mrgr_development_sync_20261020.json)：原始验收逐项核验，代码逐字复制；展示报告与压缩重复字段的统计JSON另行说明变换。原始验收报告保存在各证据目录的source_report.md。
- 代码：[消融](../../scripts/pruning_v1/README.md)、[OARI筛查](../../scripts/feedback_oari_pilot_v1/README.md)、[OARI续训](../../scripts/feedback_oari_long_v1/README.md)、[并行调度](../../scripts/feedback_oari_parallel_v1/controller.py)。

源码保留实验时的服务器路径和来源哈希，属于冻结复现记录；在其他机器运行需适配数据/历史产物路径并生成新协议，不能直接复用旧机器的协议哈希。大型权重、预测数组、完整访问明细保留在`/scratch/djc/GeoPhy-Set3D/`，不提交Git。
