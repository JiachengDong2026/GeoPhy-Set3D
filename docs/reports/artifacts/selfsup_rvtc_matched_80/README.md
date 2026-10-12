# RVTC 严格匹配 Control / Treatment 有限对照

[中文报告与图表](../../selfsup_rvtc_matched_80.md)。

协议在任何实验更新前登记并提交于 a2484c3；两组均fresh random、相同初始化/数据/顺序/站位/优化器/320步，只改变一致性0/.1。最终epoch80评价，不挑best。原v2 NO-GO保留。

本目录归档真实104条测量、52条配对指标、192条梯度、640次更新日志、协议/source/state/checkpoint哈希、测试及实际访问审计。Control假设0.1梯度单独标记，实际一致性梯度为零。35个不同测试通过，审计日志补充后复跑9项。

不含权重、测量数据或GT；完整checkpoint/optimizer/RNG与metadata inventory在外部独立目录。SHA256SUMS校验所有归档文件，artifact_manifest.json记录大小。无GT/Pool B/test/foreign checkpoint访问，无正式300轮或后续实验。
