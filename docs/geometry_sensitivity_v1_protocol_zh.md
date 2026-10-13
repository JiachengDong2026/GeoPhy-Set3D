# 原始 SelfSup v1：五档 V4 几何敏感性协议

本次接续58b2ed2，首先只做冻结推理。原SelfSup v1 SHA256为1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026，332,070参数、FP32、TF32关闭，不训练或调整损失/架构。上一轮同q A/B采样NO-GO保留；Control本身自然tertile近似均衡，Treatment主要改变A/B联合分布，不能当成显式极端几何增强已经失败。附新任务的“完整测试集”在本阶段指全部36例Val开发集，最终test仍按历史最终冻结门禁保护，Pool B/GT128不访问。

## 只依赖几何的五档定义

Pool A的635,376个V4组合全枚举，沿用原中心方向mean(1−abs(dot))分数，不按误差改公式。独立边界取总体q05/q40/q60/q80；平界归右。G1≥q80；G2[q60,q80)；G3[q40,q60)；G4[q05,q40)；G5<q05且四站最小圆周方位跨度≤5°。G5明确为尾部压力组；其他低于q05的对向/非clustered组合不纳入主五档，登记排除数，不能把主组均值当自然分布无偏风险。G1–G4是自然总体分层，G5是极端物理可实现压力测试。组名仅描述预设互补程度，不保证所有指标及所有case单调。

每档固定PCG64 seed2026101502，无放回五布局，站位顺序冻结，跨36case共用25个geometry ID。所有新模型结果出现前保存IDs、边界和描述量；旧三档/单站替换结果用于背景，不能用其误差选布局。每case/draw共同H4在五档输入并集之外，seed2026101503；H不输入网络或选择任何参数。V固定4。

## 指标与算子谱

记录真实XYZ origin（y高度、x–z水平）、pairwise/min/mean/max距离m、水平凸包面积m²及周长m、三维凸包体积m³及表面积m²、最小/平均中心锐交角、方位最小跨度、中心及真实三维射线abs-cosine冗余、真实射线方向张量三特征值/比值、高度/距离/半径、覆盖及至少两站覆盖。沿用16³/K96三线性非零权重触达proxy，可能饱和，常量关联导出null。

谱诊断复用经典RayMatrix：实际origin/direction/near/path length/valid mask，全部有效射线、K96、中点三线性**摩尔分数chi**未知量，ppm·m单位。只构建8³/512未知量的稀疏A，每站真伴随内积验证；累计稀疏AᵀA后只对512×512 float64 Gram求全谱，不构造64³稠密A。保存512个sigma、最大/原始最小/最小保留奇异值、sigma_min/max、数值秩/零空间、熵有效秩exp(H(sigma/sum sigma))、稳定秩sum(sigma²)/sigma_max²。秩阈值sigma>1e−6 sigma_max；Gram法小奇异值可能受舍入影响。若数值秩不足512，整体condition number为null（数值无穷/不可识别），另报告非零保留子空间条件数，不能冒称全空间有限条件数。

这是完整低分辨率线性chi离散谱，**不是**冻结P_diff(C)的Jacobian或完整64³的condition。P_diff仍先插值C再非线性转换chi。64³未知262,144，而4站有效射线远少于此；记录full64 nullity下界=未知数−有效测量数，其完整线性算子必有零空间。粗谱可能有区分力也可能分辨率依赖，不证明真实连续场可辨识。缓存只复用站位Gram，记录冷构建与谱耗时，不保存大型算子缓存。

## 实测、smoke与统计

900条case×geometry实际冻结推理，每行保存站位/坐标/描述量、checkpoint/seed/split、输入及H4相对L1/L2、三维relative L1/L2、MAE/RMSE、峰值/积分proxy、同步前向/评价耗时和显存。先case0×draw0五档smoke，再按成功行恢复余895条；不得重复覆盖已有完整结果。与历史完全相同输入/H时核对误差abs1e−8+rel1e−5。不使用GT/H构造prediction。代表case预先固定0/18/35、draw0、G1/G3/G5，统一y=32水平切片、GT决定的每case统一色标，不按误差挑图。

五draw先在case内平均；36case配对bootstrap10,000、seed2026101504，均值/median/sample std/q25/q75/max及worst case ID。所有10组配对差记录CI、胜负数、Wilcoxon（跨10对Holm），不把900行作独立样本。

几何关联：25固定geometry先平均36case误差，再Pearson/Spearman；case bootstrap每次重算25均值和相关CI，10,000次。另对25几何均值做10,000次geometry置换（seed2026101505）、每相关类型跨预登记标量描述量Holm，p值仅固定布局描述性推断，不给重复case×layout伪p值。报告G1–G4子集敏感性，防止全关联只由极端G5驱动。常量或非有限描述量不做强行填0/截断。

输入measurement-fit vs voxel误差不能预设弱：报告全900行描述性Pearson/Spearman、每档内关联、每case跨25布局关联分布；再去除case和geometry双向均值（加总均值）关联，按case bootstrap CI；保留固定case内geometry干预证据。原始pooled相关没有独立样本p值。共同H4作为独立评价另报，不混称input fitting。

标量关联的完整候选列表及质量方向固定在同一登记提交的 `scripts/report_geometry_sensitivity.py:DESCRIPTORS`（24项），report源码随登记commit冻结；后续若只修复导出必须记录diff，不改变统计候选。

结果后如提出质量score，只选预登记非恒定正向质量指标中25几何均值|Spearman|最大者，按其方向做minmax（仅geometry范围），明确**exploratory/GT-informed**。绝不改旧score/边界/模型、不将该分数用于当次组定义或训练。不做多指标权重优化，G1–G4关联也完整报告。多描述量选择偏倚与单seed/Val工况相关性保留。

## 解释与下一阶段

本干预可支持“当前冻结v1高度依赖几何、极端几何是实际瓶颈”，不能证明“几何比网络容量/优化/损失更主要”，因为没有容量/优化/loss干预。测量拟合与体积正确性是否一致由实际关联回答。

完成分析后才判断独立50% natural/25%moderate/25%hard增强是否值得实施；保留v1架构/损失/自然Val selector。该采样与旧同q A/B方案不同，若实现必须新协议、读取保护和smoke；不得用新Val GT挑混合比例，不启动curriculum、hard weighting或容量实验。正式训练与其预算/配对设计须独立明确登记，不能把只有sampler实现说成训练收益。

源码及协议哈希在推理前commit。读取审计只允许Val Pool A/GT64与原checkpoint；历史数据/输出/权重/已登记数值源码内容及metadata核对。提交只含代码、配置、标量、报告、PNG/PDF和审计，不提交数据/GT/体积/权重/大型算子。
