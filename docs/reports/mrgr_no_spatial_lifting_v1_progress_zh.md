# GeometryLifting 结构消融执行进度

完整对照与消融均采用正式 Multi-scale MRGR + Uniform Fusion，342725参数。
两个正式对照的种子、最佳轮次与checkpoint SHA256已由最终模型登记和实际产物核验：

| 种子 | 对照最佳epoch | 对照SHA256 |
|---|---:|---|
| 20261011 | 285 | `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc` |
| 20261019 | 286 | `19bb5f698349c976358be782e9d43301afa9e1eec4a89ddb4e1ceeb8d9bd1424` |

同种子所有共享初始化张量逐一相同。真实CUDA工程门槛和两种子的跨进程恢复检查已通过。
两任务已分别以原训练协议启动，每任务300epochs/24900updates，不重新训练对照。

实验已完成两种子各300轮/24900次更新、1190份新增预测封存、GT64评分和独立指标/统计/文件完整性核验。
两消融最佳epoch分别为278、275；完整对照和消融始终均为Uniform Fusion。
普通V4两固定种子描述性C1相对L1由8.5727%升至43.1325%，差值+34.5598个百分点，案例bootstrap95%区间[25.9406,44.6841]个百分点。
clustered-V4由28.9706%升至46.6178%，差值+17.6472个百分点，区间[11.0334,25.2122]个百分点。
两匹配种子方向一致；这是已使用测试集上的补充机制证据，且固定平面替代额外改变采样覆盖、密度和深度结构。

[正式报告](mrgr_no_spatial_lifting_v1_zh.md) · [执行与边界补充](mrgr_no_spatial_lifting_v1_execution_addendum_zh.md) · [最终验收](artifacts/mrgr_no_spatial_lifting_v1/acceptance_recovery_v1.json)。

完整服务器产物与状态见：
`/scratch/djc/GeoPhy-Set3D/mrgr_no_spatial_lifting_v1/status.json`。
不可变协议和工程检查位于同一目录。全部新增预测封存后才读取GT64；GT128始终禁止。
会话中断后进程退出、完成状态重复seed及报告模块同名导入三项执行异常均已登记、解决并保留日志，详见正式报告与补充说明。

替代结构：二维双线性缩放到32×32后沿X广播；主干和反馈两次提升均替换。
保留原visibility、geometry_descriptor、OARI、Uniform Fusion和物理反馈。
替代也改变采样覆盖/密度及深度结构，不是纯算子效应。

源码入口：`scripts/no_spatial_lifting_v1/`。原冻结源码未改；恢复入口均独立登记，不重复训练、预测或评分。
