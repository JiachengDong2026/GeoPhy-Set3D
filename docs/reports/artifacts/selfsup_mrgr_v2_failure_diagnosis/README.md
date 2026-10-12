# RVTC失败诊断结果（无GT、无参数更新）

[中文诊断报告](../../selfsup_mrgr_v2_failure_diagnosis.md)

审阅基准为52fab4f。原v2 NO-GO、原门槛、历史日志和gate checkpoint均保持原样。
本轮实际完成gate8原H、新站位U4与Val-A36的真实测量评估，以及全部44例共享参数梯度。
完整52条域×case测量结果在measurement_per_case.csv；44条梯度结果在gradient_per_case.csv。
case_station_protocol.json保存逐case站位、seed和checkpoint SHA256。
summary.json汇总三类评估、case配对bootstrap区间及梯度统计；historical_recalculation.json
区分轮内变参数统计和固定参数历史记录。初始/中间checkpoint没有保存，不能重新复评；
历史绝对体积师生差异也不能由平均比值和平均分母恢复。

本轮9项针对性测试通过；历史179项仅引用。审计记录GT/Pool B/foreign checkpoint/forbidden
projection尝试均为0。没有optimizer创建、step或gradient clipping；checkpoint、参数与缓冲区、
原实验文件、原数值源码均未变化。数据完整性仅检查metadata，没有打开GT做hash。
entry_config_error.log保留初版独立配置解析错误，发生在读取checkpoint之前；console.log
记录成功诊断，两者不能混作实验结果。

proposed_control_protocol.yaml是**待登记、未执行**的A/B对照草案：相同富视角measurement
loss，关/开原teacher consistency。保留v1，不从本轮诊断启动正式训练或改判v2。
本目录不含数据或模型权重，SHA256SUMS及artifact_manifest.json记录归档内容指纹。
