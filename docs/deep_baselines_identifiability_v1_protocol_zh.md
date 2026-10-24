# 条件学习诊断v1：运行前固定协议

本轮固定seed20261011；只使用train前4例（A为嵌套1/2/4例，C为前2例）和Pool A几何。使用原DisjointABHSampler在epoch0固定A4/B4/H4，H不作为输入，不进入任何训练或C梯度目标。新命名空间独立保存，不改主模型/投影器/原loss/正式队列。

数值设计以 `configs/baselines/identifiability_v1/design.json` 为准，首次实验前登记其SHA256和所有新脚本源码SHA256。A为每模型3个规模、原v2 consistency；B为每模型同初态v2/v3两个条件、固定4个解析非均匀场；每项200次更新，共10项2000次，上限低于用户2500。CPU工程检查不执行optimizer.step；C为0次更新。只允许工程异常或资源冲突停止；不按H早停/调参/选checkpoint。

全部fresh v2构造，使用此前合法train A/B测量闭式尺度0.004195626187323315，不读取真实GT统计，不重新拟合初始化，不读取失败权重用于训练。相同模型全部条件初态逐位一致。batch=min(2,N)，固定顺序循环；AdamW lr2e-4/wd1e-4、clip1、FP32/TF32off，诊断固定LR、无scheduler。该诊断目标独立于原300epoch协议：0.5cross+0.1observed+0.05consistency，完全无hold项。使用原measurement_discrepancy和K96物理投影；v3仅替换consistency为已验证的齐次版本。

H能力隔离：准备阶段只抽取A/B，H只登记站位/文件身份；评价函数才从train投影文件抽取H。训练函数仅接收A/B字典，H张量不在其闭包或参数中。合成场生成、训练、评价分开；训练进程只读合成A/B测量包，真场只在生成/独立评价中读取，不用于初始化、loss或选择。合成目标由独立NumPy FP64解析场Gauss-Legendre256积分生成，以128阶差异核验，并比较64³场/K96投影离散误差。无真实GT、GT统计、val/test数组或Pool B访问。归一化采用冻结Y_SCALE常量，数据适配器不调用读取历史混合统计JSON的构造函数；复用原measurement-only getter。

在updates0/10/25/50/100/150/200评价，H连续两次<=.99才满足H门槛；训练A/B宏平均L1<=.99且相对初态至少降低.01。全部案例输出mean>=1e-8、空间CV>=.05；测量置零或有效射线内固定置换引起的绝对MAE/固定初始化浓度尺度>=.001，重复前向逐位一致。输入测量梯度L2>1e-12且有限、敏感性AD/FD有可辨认证据，才允许初步GO。所有case/branch须通过，不以近零分母夸大响应。相对变化一律用固定0.004195626187323315稳定尺度，另报绝对MAE/RMSE。

C分别检查v1 fresh、v2 fresh、原失败best、可用v2/v3 best及last（每模型7项）；允许相同state哈希的结果只算一次并明确别名。对前2例A/B分别做原值、0、.5、1.5、有效射线固定置换，保持所有其他字段逐位一致。记录encoder/聚合/算子/decoder/logit中间表示的值和响应；逐层gradient记录全部encoder/latent、首个query chunk的operator/query/logits及末层weight/bias，明确query抽样范围，不能冒充全query梯度。H仅无梯度评价。输入幅度方向的输出均值/固定尺度及平均logits分别做AD与中心FD，epsilon=.01/.001，误差门槛5%或绝对1e-8；AD/FD均低于绝对分辨率仅记precision-limited，不能作有效信息流证明。C单独标记响应证据/不确定，不能用无训练C满足“训练改善”门槛。

没有通过不自动增加搜索。若未过且训练L1不能<=.99，只有近零或无进展等明确证据才NO-GO；非近零但预算不足为INCONCLUSIVE。训练A/B改善而H失败归A2；H好但条件/结构不足仍不能GO。B额外独立报告合成体积L1/MAE/RMSE、峰值、结构，不将测量GO等同三维恢复或真实基线成功。C解释信息传播位置必须同时给绝对量、层间相对量和替代解释。

GPU只允许UUID为GPU0，复用全局UUID锁、15秒双空闲快照、继承FD与原控制器进程组冲突保护。获取资源最多60秒；不满足则BLOCKED，不无限等待或转GPU1–3。资源冲突只停止本任务。CPU可先完成数据生成/工程测试；所有哈希、拒绝和异常独立保存。

独立复核用另一个脚本从保存的逐射线投影/测量/掩码重算L1/L2、零场、逐case宏平均和GO条件；从保存全场重算空间/合成指标/干预响应。不调用训练汇总函数。原始数组和权重只留scratch，Git仅小日志、配置、CSV、哈希和报告。主模型在相同合法train输入上建立前后逐位锚点，核验双种子checkpoint、冻结源码/配置及历史结果哈希；不为回归读取val/test。正式队列始终暂停。任何诊断GO均不授权正式300epoch或第二种子。
