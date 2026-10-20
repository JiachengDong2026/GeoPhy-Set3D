# 均匀融合骨干上的匹配消融

全部单因素实验使用原始数据/损失/采样/300轮预算；一致性消融只将对应权重置零。种子20261011/20261019的均匀融合对照复用历史已核验产物；新训练来自相同初始构造和数据流，CUDA原子累加存在跨执行非确定性。描述符结构删除复用已完成的D组。

本阶段按用户要求，先完成两种子均匀融合与两种子反馈OARI的三维配对评价；7个尚未启动的骨干消融暂缓，3个第二种子此前取消。暂缓任务不自动恢复，反馈OARI是否作为最终模型等待用户审阅，本报告不作自动晋级。

## 暂缓的新消融（未训练、未评价）

- `uniform_no_feedback_seed20261011`
- `uniform_no_feedback_seed20261019`
- `uniform_zero_residual_seed20261011`
- `uniform_zero_residual_seed20261019`
- `uniform_no_ray_relations_seed20261011`
- `uniform_no_consistency_seed20261011`
- `uniform_single_scale_seed20261011`

[阶段调整及未启动证明](artifacts/mrgr_uniform_final_v1/evaluation_stage_amendment_v1.json)

以下结果表仅为已有均匀融合和描述符删除证据，不代表暂缓消融已经完成。

| 模型 | 普通V4 L1 | 聚簇V4 L1 | 聚簇L2 | 聚簇质心mm | 聚簇GT峰处比 | 聚簇峰比绝对误差 | 聚簇Dice50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_fusion_seed20261011 | 8.5928% | 28.0246% | 45.0057% | 8.9787 | 0.3322 | 0.1752 | 0.4136 |
| uniform_fusion_seed20261019 | 8.5527% | 29.9165% | 44.1267% | 10.1516 | 0.3716 | 0.2145 | 0.4104 |
| uniform_pruned_observability_seed20261011 | 8.7227% | 29.7149% | 46.0687% | 8.6973 | 0.3414 | 0.1986 | 0.4174 |
| uniform_pruned_observability_seed20261019 | 8.7217% | 28.9359% | 41.4762% | 8.6650 | 0.4137 | 0.1945 | 0.4850 |

## 两种子案例配对差值

| 因素（消融−均匀融合） | 布局 | L1差百分点 | 未校正95%案例区间 |
|---|---|---:|---|
| pruned_descriptor-minus-uniform2seed | Val-A-V4 | +0.1495 | [+0.0346, +0.2808] |
| pruned_descriptor-minus-uniform2seed | clustered-V4 | +0.3548 | [-0.4659, +1.3219] |

## 对机制解释的支持程度

pruned_descriptor-minus-uniform2seed：普通V4主要终点同时区间跨零，效应尚不确定；聚簇探索性L1差为+0.3548个百分点。该判断受容量/计算差异和训练种子数量限制。


旧GCA骨干上的历史消融不当作本骨干的同配置消融。无反馈删除编码器/适配器且不计算反馈投影，容量减少12512；零残差保留容量但跳过反馈投影，计算成本不同；无射线关系将前三关系通道置零，保留描述符；无一致性只改权重；单尺度只改空间拓扑；描述符删除为实际结构删除384参数。名义参数相同不表示有效容量与梯度路径相同。

[逐种子、逐案例、全部指标及C0/C1](artifacts/mrgr_uniform_final_v1/per_case_metrics.csv)；[完整差值及区间](artifacts/mrgr_uniform_final_v1/paired_statistics.jsonl.gz)；[同设备成本](artifacts/mrgr_uniform_final_v1/benchmark.json)。两种子不足以估计训练种子总体不确定性。
