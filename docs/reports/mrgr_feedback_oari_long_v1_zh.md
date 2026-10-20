# MRGR 残差反馈 OARI：300轮延长训练

更新：2026-10-20T02:44:53.333489+08:00；状态：long_training_complete。

根据20轮开发结果追加的探索性实验。三组均从各自第20轮last.pt继续至300轮，每组追加280轮；保留原cosine T_max300、优化器、全部RNG与数据流、历史最佳轮和前20轮指标，不重置学习率。原20轮结果独立冻结。

仅166训练例与36已用开发例，不读test投影、test GT64或任何GT128。训练和选轮只用投影；选轮为全300轮H4相对L1最小值，无提前结束。此为观察开发结果后追加的单种子实验，不能视作独立确认。

| 版本 | 进度 | 最佳H4投影相对L1 |
|---|---:|---:|
| feedback_oari | 300/300，complete | 2.9629% |
| baseline | 300/300，complete | 3.1410% |
| feedback_oari_zero_residual | 300/300，complete | 5.1177% |

调度经用户授权修订为最多两组并行，每组仅在完全空闲GPU上启动。共享既有UUID锁，连续两次15秒检查无计算进程、已用<100MiB、可用≥20000MiB且利用率≤5%，启动前再次核查。正在运行的消融继续执行。

三组依次为真实残差OARI、原始MRGR、同参数零残差OARI。模型结构与已通过的20轮CPU/GPU/跨进程恢复检查完全相同；新检查逐字段验证断点迁移，模型/优化器/调度器/RNG/计数/配置完全相同，只有protocol_sha256显式绑定新预算协议。

[预登记](artifacts/feedback_oari_long_v1/design.json)；[迁移验收](artifacts/feedback_oari_long_v1/migration_verification.json)；[实时状态](artifacts/feedback_oari_long_v1/status.json)。

| 布局 | 版本 | 三维L1 | 三维L2 | 峰比绝对误差 | 质心mm |
|---|---|---:|---:|---:|---:|
| Val-A-V4 | feedback_oari | 9.0465% | 16.0322% | 0.1781 | 1.5136 |
| clustered-V4 | feedback_oari | 30.5974% | 44.5903% | 0.2667 | 9.3359 |
| Val-A-V4 | baseline | 9.4688% | 16.2452% | 0.1778 | 1.5711 |
| clustered-V4 | baseline | 33.7562% | 49.5664% | 0.2576 | 12.0359 |
| Val-A-V4 | feedback_oari_zero_residual | 12.8679% | 18.3085% | 0.1595 | 1.9875 |
| clustered-V4 | feedback_oari_zero_residual | 35.2059% | 50.6850% | 0.3184 | 13.2674 |

长程开发结果存在取舍，尚未同时改善L1、峰值与定位。

全部17布局先封存预测，再用开发GT64评分；聚簇五draw先按案例平均后配对bootstrap20000次，区间不覆盖训练种子不确定性。

[完整统计](artifacts/feedback_oari_long_v1/summary.json)；[独立重算](artifacts/feedback_oari_long_v1/independent_verification.json)。

原20轮结果：`/scratch/djc/GeoPhy-Set3D/mrgr_feedback_oari_pilot_v1`。本轮产物：`/scratch/djc/GeoPhy-Set3D/mrgr_feedback_oari_long_v1`；入口：`/home/djc/projects/GeoPhy-Set3D-feedback-oari-pilot-v1/scripts/feedback_oari_long_v1/controller.py`。

调度修订：用户明确授权使用空闲GPU3后，本实验改为最多两组训练并行；相同的300轮预算、模型和数据协议保持不变。以[调度修订记录](artifacts/feedback_oari_long_v1/parallel_scheduling_amendment.json)为准。

## 配对比较与取舍

下表是反馈OARI减去本轮匹配的原始MRGR；三维L1差值以百分点表示，负值代表误差下降。

| 布局 | 三维L1差 | 探索性95%案例区间 | L1降低/升高案例 | 质心误差差 |
|---|---:|---|---:|---:|
| 普通四视角 | −0.4223 | [−0.6231, −0.2204] | 33/3 | −0.0576 mm |
| 聚簇四视角（五draw案例内平均） | −3.1589 | [−4.5582, −1.7919] | 32/4 | −2.7000 mm |

普通视角L1相对降低约4.46%，聚簇视角约9.36%。峰比绝对误差普通视角由0.1778变为0.1781、聚簇由0.2576变为0.2667，未满足此前要求所有关键指标同时不恶化的筛查规则。未经多重校正的案例区间不能覆盖训练种子不确定性。本轮baseline为从自身20轮断点续训的匹配对照，CUDA非确定性使其数值与历史冻结模型略有不同；不混用两套baseline。

## 仓库同步说明

本文件是便于GitHub浏览的报告副本，证据与图片使用仓库相对链接。原始验收时报告逐字保存在同目录证据中的 `source_report.md`；原验收及源码SHA继续指向原执行位置，保留历史溯源含义。大体积权重和预测数组仍在原scratch目录。汇总与导出校验见[本轮结果索引](mrgr_development_results_20261020_zh.md)。
