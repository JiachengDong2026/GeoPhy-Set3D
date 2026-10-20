# MRGR 当前模型与实验安排（2026-10-20）

按用户在四项训练和35例三维评价完成后的决定，暂定 **MRGR-UniformFusion-v1（均匀融合MRGR）** 为主模型，停止反馈OARI的后续研究。历史OARI代码、权重、结果和负结果保留，不继续训练、搜索或晋级。

主模型沿用既定身份：seed20261011，第285轮，342725参数；只将GCA学习融合替换为有效视角平均，保留原有多尺度、射线关系、可观测性描述符、一致性约束及原残差反馈。停止的是额外反馈OARI研究，原模型第一阶段的OARI和原残差反馈仍保留。

配置：[mrgr_uniform_fusion_v1.json](../configs/mrgr_uniform_fusion_v1.json)。权重SHA256：`64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。不根据这次气袋或此前已使用的35例测试重新挑选checkpoint或种子。

7个尚未启动的骨干消融继续暂缓，此次小测试结束后也不自动启动。消融前先执行实心气袋的冻结模型推理；同权重重算圆环作形态参照，目标和布局先登记，保留所有结果，不对新目标微调。

气袋测试入口：[test_uniform_nonplume_gasbag_v1.py](../scripts/test_uniform_nonplume_gasbag_v1.py)。大型产物位于`/scratch/djc/GeoPhy-Set3D/mrgr_uniform_nonplume_gasbag_v1`。

气袋小测试已完成并验收：[完整报告](reports/mrgr_uniform_nonplume_gasbag_v1_zh.md)、[结果解读](reports/mrgr_uniform_nonplume_gasbag_v1_interpretation_zh.md)。扩大角度覆盖后实心团块外形有所恢复，但仍明显低估浓度，不据此宣称通用非羽流泛化；不开展额外调参或训练。

历史评价：[双种子反馈OARI三维结果](reports/mrgr_uniform_oari_v1_zh.md)。此前验收记录的“等待用户审阅”描述当时状态，本文件记录用户审阅后的后续决定，历史验收不覆盖。
