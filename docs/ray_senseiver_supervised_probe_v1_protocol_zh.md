# Ray-Senseiver 训练集 GT64 有监督可学性诊断 v1（训练前登记）

本轮固定一次单样本→条件八样本检查，不追加结构/loss/学习率搜索。原始 RaySenseiverSS（254913参数、64latents、原sigmoid），不加载任何已有权重，不使用全局latent残差。历史 REPAIR_FAILED_NO_GO 结论及所有历史保护不变。本实验及权重不得用于纯自监督榜单或后续标称无GT的warm start。

- 固定历史train序号0–7、Pool A、站位[26,27,45,42]；登记实际source-index原始行号、sample/case ID、frame60/2s、文件hash及所有valid ray（含精确零）。新独立SupervisedProbeGuard仅白名单当前阶段的训练projection/GT64；禁止GT128、Val/test/Pool B及外来权重，数据只读。prepare配对登记可读8例，A训练/检查只读0，B仅在A通过后读取8例。
- 直接复用ProjectionPoolDataset的GT64读取验证：FP32 XYZ CH4质量分数[0,1]，y为高度，[0,.6]^3端点网格。GT64为已有endpoint-aligned三线性预处理，不新缩放/插值。配对依靠source index、sample_meta、frame_manifest和来源JSON的实例/时刻/源hash一致性，不读取GT128重算。Pool A投影来自128³源场/256采样；GT64降采样与原P_diff/K96投影存在已知离散差异，本轮不将测量投影纳入训练损失，也不重投影拟合。
- 唯一训练目标：mean_i(mean_voxel((pred-GT)^2)/max(mean_voxel(GT^2),1e-12))。FP32 AdamW lr2e-4/wd1e-4、clip1、固定LR无scheduler。完整64³查询/所有有效输入射线，seed20261011。输出层weight×.01，bias=logit(该阶段拟合GT全部体素均值)，clamp[1e-6,1-1e-6]，记录是否生效；A只用GT0统计，B仅GT0–7，重新fresh，不继承A或smoke权重。
- A batch1，最多1000更新/900秒，B有效batch2固定连续配对(0,1)…(6,7)，最多2000更新/1800秒；计时包含本阶段加载/核验、训练、评估和检查点开销（更保守）。每100更新完整评估，epoch0保存完整状态。时间上限在新更新前检查；必要最终评估/保存可略超墙钟上限，但不追加更新。
- A连续两次100步评估满足平均NMSE≤.10、relativeL1≤.30才通过；否则B不执行。B平均NMSE≤.20、relativeL1≤.40且较八例逐体素平均模板NMSE降低≥30%；固定0/1/2至少2例normal-zero relativeMA≥1e-3（分母normal meanabs+1e-12）。B每200步做normal/zero/donor[1,2,0]测量干预，geometry/path/mask不变；联合GO只在有当前干预的200步检查判定，连续两次联合检查通过；中间100步若NMSE/L1未达标重置通过计数。不依赖旧干预宣布通过。
- 每100步保存逐例/平均NMSE、三维relativeL1/L2、预测均值/峰值/空间std/总量比、GT峰值10%以上区域误差、原始logit统计、sigmoid导数；每次更新记录三组梯度/更新、真实step/case/站位、时间/显存。基准：零、每例GT均值均匀场（oracle幅度、不是独立预测器），B增加八例逐体素平均模板。A相同模板就是GT本身，仅作身份参照，不能用于A判定。
- 平台规则固定为相邻、互不重叠200步训练NMSE均值比较；A从600、B从1000步起每100步检查，相对改善<2%且未达到通过条件，连续两次→PLATEAU_FAILED_TO_LEARN。损失窗口包含固定顺序的完整batch周期。未通过而达更新/时间上限→BUDGET_EXHAUSTED_UNRESOLVED，不宣称架构不可学。
- NaN/Inf、配对/轴序/梯度/更新/访问错误立即中断并保留最后有效checkpoint及错误，不静默修run；明确工程错误修复需版本化重新登记。OOM最多一次batch2→1/accum2，保持完整ray/query；A本已batch1或再次OOM则安全暂停，不杀他人进程。SIGINT/TERM边界保存完整optimizer/RNG/更新数/已用时间/退出计数，只从本阶段last恢复，科学终止不得恢复。所有权重/体素留外部独立目录。

配置、入口、测试、登记/初态hash提交后，先独立smoke（1步）再重新fresh科学训练。只根据上述门槛分四类结论；未通过暂停当前Senseiver路线并给一个独立替代基线建议；通过只提出正式有监督基线方案。本轮不启动全量/300轮、不追加第三次修复。
