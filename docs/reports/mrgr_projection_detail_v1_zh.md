# 多尺度 MRGR：投影梯度细节损失匹配筛查

本轮仅改变训练目标：原损失加 `0.05 × 共同 H 投影梯度 SmoothL1`。模型结构、参数量344,678、初始化、166训练/36验证、站位采样、20轮（各1,660次更新）及300轮余弦调度前缀均匹配。H 从未输入重建网络。

梯度项使用水平/垂直相邻射线差，预测和测量均除以原始固定 Y_SCALE，SmoothL1 beta=0.01。仅统计两端均有效的相邻对；每案例先合并两方向所有有效对平均，再跨案例及 A/B 重建平均；无有效对的案例贡献0。不除以像素物理间距，不额外归一化小梯度，不新增前向计算。

| 留出投影相对 L1 | 原多尺度 | 加投影梯度损失 | 相对变化 |
|---|---:|---:|---:|
| 普通均值 | 9.3338% | 9.2174% | -1.25% |
| 普通P90 | 13.8832% | 14.0521% | +1.22% |
| 聚簇五布局均值 | 48.5929% | 48.0911% | -1.03% |

完整晋级检查：`{"ordinary_H": true, "ordinary_P90": true, "clustered_H": true, "runtime": true, "five_hour_budget": true, "smoke": true}`；总判定 **通过**。

预热后每轮：原多尺度 33.74s，处理组 33.93s；300轮外推 2.83h。该外推取决于当前GPU负载。

普通案例配对均值差95% bootstrap区间（处理−对照，相对L1原始比例）：[-0.003931897403558903, 0.0015464672886674753]，胜/负 20/16。此为单种子短程测量筛查，不代表收敛后的三维精度或统计显著性证明。

本轮未读取三维GT、未评价最终35例测试集、未启动300轮训练。源代码、数据、协议、权重和全轮采样/打乱证据见输出目录；所有旧文件保持原样。

证据：[协议](../../outputs/research/mrgr_projection_detail_v1/protocol.json)、[smoke](../../outputs/research/mrgr_projection_detail_v1/smoke.json)、[结果](../../outputs/research/mrgr_projection_detail_v1/gate.json)。

## 补充审计与解释

本轮通过全部工程晋级门槛，但普通均值改善仅1.25%，配对均值差区间跨零；同期不同GPU、相同初始化与采样的重复对照也存在轨迹漂移。因此目前只保留为**待重复候选**，尚不能判定是稳定的科学改进，也不自动开展300轮。跨GPU基线漂移应单独报告，不能混合不同实验对照计算因果收益。

零权重审计中，总损失、全部三维输出及投影均与原objective严格bitwise一致，包装后全局binding恢复。20轮后在预先固定的epoch0前两条训练案例上只读复算，处理组新增加权梯度loss占原目标 **0.19984%**，全参数梯度范数占比 **0.26627%**，与smoke同一context输出参数梯度占比 **0.16060%**。干预确实非零但较弱；该局部探针不能代表所有批次。未调lambda、未优化权重、未修改gate。

额外证据：[零权重等价](../../outputs/research/mrgr_projection_detail_v1/zero_lambda_equivalence.json)、[后期干预审计](../../outputs/research/mrgr_projection_detail_v1/late_intervention_audit.json)、[完整汇总](../../outputs/research/mrgr_projection_detail_v1/summary.json)。
