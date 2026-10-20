# 均匀融合 MRGR 正式消融进度

更新时间（北京时间）：2026-10-21T01:38:02.295847+08:00

均匀融合主模型固定为 seed20261011 / epoch285 / 342725参数。反馈 OARI 已停止。
本阶段为已使用35例的补充配对消融，不是独立确认；GT128始终禁用。
复用15个已封存历史方法；两种子无反馈、零残差及三项单种子消融需要补跑。
原14项校正分母保留；三项单种子仅探索性95%案例区间，不替代原双种子比较。

| 任务 | 状态 | 轮次 | 更新 | 最佳验证H4 | GPU UUID |
|---|---|---:|---:|---:|---|
| uniform_no_feedback_seed20261011 | complete | 300 / 300 | 24900 | 0.05294603210169865 |  |
| uniform_no_feedback_seed20261019 | complete | 300 / 300 | 24900 | 0.05622815157361254 |  |
| uniform_zero_residual_seed20261011 | complete | 300 / 300 | 24900 | 0.05143357614574154 |  |
| uniform_zero_residual_seed20261019 | complete | 300 / 300 | 24900 | 0.055353856467545495 |  |
| uniform_no_ray_relations_seed20261011 | complete | 300 / 300 | 24900 | 0.031293950972848025 |  |
| uniform_no_consistency_seed20261011 | complete | 300 / 300 | 24900 | 0.030296035547312472 |  |
| uniform_single_scale_seed20261011 | complete | 300 / 300 | 24900 | 0.032228146037551614 |  |

阶段状态：`experiments_accepted`。
运行根目录：`/scratch/djc/GeoPhy-Set3D/mrgr_uniform_ablations_v1`。
训练完成并冻结全部新增预测前，不读取本阶段GT64、不报告三维改善结论。

执行入口：`/scratch/djc/envs/gas-torch/bin/python scripts/uniform_ablations_v1/stage.py controller`。
调度复用原UUID锁/两次15秒空闲检查/worker继承锁FD/2→最多4并发门槛；失败保留证据并阻断后续评分。
