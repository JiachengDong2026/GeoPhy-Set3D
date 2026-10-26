# 经典算子基线：仓库同步与存储迁移检查

检查时间：2026-10-26T14:20:52.421097+08:00。此前封存的实验源码、门槛、模型状态和结果保持原样；此记录是后续存储检查，不重新生成历史实验manifest。

用户已将`/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_paper_final_v1/`下的`classical`与`evaluation`迁到`/data/djc/GeoPhy-Set3D/outputs/research/mrgr_paper_final_v1/`，原路径保留绝对软链接。本次确认两个链接解析正确，全部目录项的类型/权限/修改时间及文件大小与迁移清单一致；各抽查5个JSON文件SHA及通过旧路径读取的字节一致。

| 目录 | 文件数 | 目录项数 | 逻辑字节 |
|---|---:|---:|---:|
| classical | 14820 | 14900 | 11538350051 |
| evaluation | 19478 | 19550 | 37258098547 |

迁移执行报告已记录全部34,298个文件源/目标SHA一致。本次完整复核元信息并抽查10个JSON内容，不声称再次完整读取约45.5GiB内容，也未读取三维GT数组。原始迁移报告：`/data/djc/migration-reports/20261026_mrgr_paper_final_v1/migration-report.md`。

此前“存储不足”的根因是home个人软配额50GiB超限且宽限期结束，不是分区总空间耗尽。当前home账户用量约15.6GiB，配额已恢复。可继续用旧路径；新增大数组/checkpoint仍只写既有scratch/data工作区，不复制到Git。

经典算子实验仍位于`/scratch/djc/GeoPhy-Set3D/classical_operators_v1`，其88项交付文件和4个pilot checkpoint SHA复核通过；历史DeepSets/主模型/旧队列证据SHA复核通过。不需要更改冻结配置里的数据路径。

本次源码及报告同步采用独立实验分支与main快进，不改历史模型或科学结论：GeoLift-DeepSets-Sum冻结复用，DeepONet/FNO当前适配NO-GO；未追加训练、GT评价或修复搜索。

机器可读检查：[migration_check.json](artifacts/classical_operators_v1_sync_20261026/migration_check.json)。
