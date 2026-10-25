# 固定几何、变化测量诊断 v2：运行前协议

本轮只运行 fresh GNOT/DeepSets 各 200 updates，seed20261011，复用 v2 测量尺度初始化与 v3 齐次一致性。v1 的 2000 加本轮 400，共 2400，不超过 2500。工程测试/梯度审计不创建 optimizer，不占额外更新。不恢复任何正式队列；不启动正式训练或第二种子。

geometry-only 适配器直接读取唯一 train_geometry.npz，不实例化任何历史 dataset，不调用 source_dataset/selected_sample/_projections/_measurement_sample/_views。预备阶段只查看过该几何的 schema 和站位；没有读取真实投影、GT 或统计。固定站位为 A=[0,3,6,9]，B=[1,4,7,10]，H=[2,5,8,11]。全部其余字段（包括恒定 case_id/sample_id）在所有羽流上完全相同，模型不能从 metadata 识别羽流。

8 个解析定义、6/2 划分、28 个无序配对、阈值固定在 design.json。前四例复用 v1，新增紧凑低幅度、宽扩散偏移两例训练场及旋转、三羽流组合两例未见场。无噪声解析 FP64 真场和独立 128/256 阶积分；目标实际保存 FP32，真场保存 FP64。独立参考与冻结逆投影 K96/64³ 的偏差同时记录。生成期可计算解析场；训练期没有真场访问能力。

AdamW2e-4/wd1e-4，batch2，固定三批循环，FP32/TF32关闭/固定随机性与采样顺序，clip1，无调度，loss=.5cross+.1observed+.05homogeneous_consistency。训练函数只接收 A/B。训练 worker 全程 CPython open audit 拒绝 H、真场、未见 AB、真实数据及外部数组/checkpoint；这是文件访问层边界，不声称是对恶意 native code 的 OS 沙箱。访问拒绝测试使用真实路径验证。评价使用独立 worker；为完全切断 H/真场结果反馈，训练在 0/10/25/50/100/150/200 固定保存权重，训练退出后统一按这些节点评价。终态固定 update200，无 H/真场选模。

每节点、8例、A/B 两分支，保存正常/置零/固定射线置换输出和全部测量投影；正常与干预重复逐位校验。交换包含全部28对的两个方向：几何完全相同，替换测量后输入逐位等于 donor 原始输入，重复 donor 前向逐位核验；可复用 donor 输出数组，禁止虚构独立预测。记录 MAE/RMSE、固定 C_init 归一化、空间相关、峰值位移、预测差分与真场差分余弦、匹配/错配真场 L1 优势。差分余弦是完整差分向量的夹角，不用均值去除来忽略幅度差。分组为 seen-seen、unseen-unseen、seen-unseen，另按 recipient seen/unseen 分别报告匹配优势。

全部 GO 条件在 design.json 登记：seen 各例 AB<=.99 且较初态至少下降.01；所有8例 H末两次<=.99；全部分支 mean>=1e-8、CV>=.05；全部28配对 MAE/C_init>=.001且可重复；每个配对组差分方向余弦>0占比>=80%、中位数>=.1；seen 至少4/6、unseen2/2三维L1<=.99；seen/unseen recipient 分别至少75%有向交换的匹配真场L1优势>=.01；各例输入测量通道梯度L2>1e-12且梯度有限。全部条件同时满足才 GO_DIAGNOSTIC_ONLY。响应或近零失败为 NO_GO；其余预算内证据不足为 INCONCLUSIVE，不能通过加步或改阈值解决。A/B两分支分别检查，不用宏平均掩盖失败。

零场使用相同掩码/目标重算测量L1/L2和体积L1=1。正常场统计、全体积观测损失的 measurement-token 通道/末层 weight/bias 梯度在各固定评价节点记录，零 optimizer。禁止真实 GT/GT128/val/test/Pool B、GT统计和监督 warm start。

GPU0 只通过既有全局 UUID flock、15秒双空闲快照、冲突停止及最多60秒获取；不触及GPU1–3。任何失败保留证据，停止后分析，不静默重试。CPU测试、源码、协议、初始化和输入 SHA 全部在第一次 optimizer update 前封存。独立 NumPy 脚本直接从原始数组重算，不调用运行期 metrics。Git只同步小型证据，原始数组和权重保留scratch。

启动前工程修订：首个GPU smoke因冻结grid_sample三维CUDA反向不支持框架严格确定性开关而退出，0次optimizer更新。恢复冻结流程原有的严格确定性开关关闭设置，不修改物理算子；CUDA反向原子累加不保证重跑逐位相同，固定种子和采样顺序仍生效，重复推理仍必须逐位一致。失败记录及修订前源码/协议封存，明确recover-reason后重做smoke，不增加训练预算。阈值、模型及羽流定义未改。
