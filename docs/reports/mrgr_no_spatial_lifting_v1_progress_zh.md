# GeometryLifting 结构消融执行进度

完整对照与消融均采用正式 Multi-scale MRGR + Uniform Fusion，342725参数。
两个正式对照的种子、最佳轮次与checkpoint SHA256已由最终模型登记和实际产物核验：

| 种子 | 对照最佳epoch | 对照SHA256 |
|---|---:|---|
| 20261011 | 285 | `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc` |
| 20261019 | 286 | `19bb5f698349c976358be782e9d43301afa9e1eec4a89ddb4e1ceeb8d9bd1424` |

同种子所有共享初始化张量逐一相同。真实CUDA工程门槛和两种子的跨进程恢复检查已通过。
两任务已分别以原训练协议启动，每任务300epochs/24900updates，不重新训练对照。

本文件记录启动阶段，实时状态见服务器：
`/scratch/djc/GeoPhy-Set3D/mrgr_no_spatial_lifting_v1/status.json`。
不可变协议和工程检查位于同一目录，正式结果尚未完成。
全部新增预测封存前不读取GT64；GT128始终禁止。

替代结构：二维双线性缩放到32×32后沿X广播；主干和反馈两次提升均替换。
保留原visibility、geometry_descriptor、OARI、Uniform Fusion和物理反馈。
替代也改变采样覆盖/密度及深度结构，不是纯算子效应。

源码入口：`scripts/no_spatial_lifting_v1/`。正式报告将在完成封存、评分及独立核验后生成。
