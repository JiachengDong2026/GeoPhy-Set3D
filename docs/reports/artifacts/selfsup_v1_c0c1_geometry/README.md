# 冻结 SelfSup v1 C0/C1 诊断标量归档

[中文报告](../../selfsup_v1_c0c1_geometry_diagnostic_zh.md)完整说明结论、范围和局限。
[冻结协议](../../../selfsup_v1_c0c1_geometry_protocol_zh.md)在首次新推理前提交。
本归档不含checkpoint、GT、measurement或prediction体积；外部目录与执行范围见execution_status.json。
诊断完成状态与GitHub同步状态分开，远端提交一致性由最终终端检查确认。

| 文件 | 内容 |
| --- | --- |
| protocol.json / protocol.sha256 | 冻结config、900清单来源、checkpoint/metadata/source hash、预算及推理前算术勘误 |
| station_subsets.json / geometry_layouts.json / geometry_descriptors.csv | 原900输入/H4身份及25布局描述量，直接复制并验证旧hash |
| checkpoint_provenance.json | 原checkpoint路径/完整config/身份/状态/参数量/训练来源 |
| implementation_manifest_original.json | 首次新推理前封存的数值/统计/测试源码hash |
| implementation_manifest_height_label.json | 仅修正端点对齐高度标签后的版本；zero及shortcut采用此manifest |
| implementation_manifest.json | 最终报告版本，仅再修改固定图的共享显示色标；数值推理/统计源码hash与首版相同 |
| paired_900.csv / case_group_180.csv | C0/C1十指标、幅度和Δ、身份/hash；case内先平均五draw |
| zero_paired_900.csv / zero_case_group_180.csv | zero C1对normal的配对差，同时保留zero内部C0/C1 |
| shortcut_18.csv | 固定0/18/35×draw0×G1/G3/G5×两种模式，9行/模式 |
| paired_statistics.csv / statistics.json | 五个主L1与95个辅助比较、case bootstrap CI、Holm、反向事件 |
| geometry_correction_900.csv / amplitude_case_group.csv | 有效区域幅度、几何身份、local descriptor摘要和误差Δ |
| correlations.csv / correlations.json | 672项预登记探索性几何/幅度关系，含组内、G1–G4、macro、常量null |
| independent_scalar_audit.py / .json | 不重新推理的独立IDs/差值/聚合/bootstrap/保存数组/边界审计，含补充探索性L2事件 |
| *_predictions_complete.json / *_gt_complete.json | 保存预测row hash及独立GT结果完成hash |
| inference_events.jsonl / inference_sessions.jsonl | 实际started/returned/saved次数、运行设置、参数不变及访问计数 |
| *_gt_access.json / figure_gt_access.json / access_sessions.jsonl | GT后处理与绘图访问边界，不把open事件数当案例数 |
| integrity.json / execution_status.json | 数据/历史metadata、保护源码、checkpoint/来源hash、实际1818前向和交付状态 |
| full_tests.log | 256项仓库回归通过，独立临时fixture |
| normal_reproduction.json | 正常C1全部900行×十指标复现，继承容差无放宽 |
| figures/ | 10个预设图的PNG/PDF；三例各含固定y32及高度均值 |
| artifact_hashes.json | 本归档文件SHA256，排除清单自身 |

C0不是无MRGR独立训练模型；normal的delta_*是C1−C0，负值改善。
zero_minus_normal_*和shortcut_minus_normal_*是受损残差信息后的C1减normal C1，正值恶化。
持平阈值1e−8，ratio零分母阈值1e−12，原始case/layout频率只描述，统计单位为36case。
图示坐标标签/共享显示范围修订都有独立commit与旧manifest，不改案例、切片、输出或统计。
相对误差无量纲；MAE/RMSE/peak是CH4质量分数；measurement raw幅度为ppm·m；context特征RMS没有物理浓度单位。

独立核验：

```bash
python docs/reports/artifacts/selfsup_v1_c0c1_geometry/independent_scalar_audit.py \
  --output /data/djc/GeoPhy-Set3D/outputs/diagnostics/selfsup_v1_c0c1_geometry
python scripts/diagnose_selfsup_v1_c0c1.py verify
```
