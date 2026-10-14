# 冻结 SelfSup-MRGR v1：C0/C1 五档几何配对诊断协议

本轮仅诊断 original frozen SelfSup-MRGR v1，不训练、不新增网络模块、不改历史结论。
基线 HEAD 为 `4735d7d`，main 工作区干净，未发现适用 AGENTS.md。原 v1、same-q 及显式增强 NO-GO 全部保留。
原 checkpoint 从 frozen_checkpoint.json 解析，必须 SHA256=
`1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026`，
config 完全一致、332070 参数、300 epochs 完成、仅 measurement 训练、best epoch 278。

## 身份、顺序与边界

完整复用原 geometry_sensitivity_v1 的 station_subsets.json 900 条、25 个 V4 布局、36 Val case、
G1–G5 各五 draw，input IDs 顺序及同 case/draw 跨组共同 H4 不变。
沿用原几何边界、valid-ray/view mask、训练 p99.5=32284.894580078173 ppm·m、P_diff/K96、
64³ XYZ 质量分数。y 为高度，域 [0,0.6]³ m。历史 clustered V4 / 原25布局G5 / 增强确认G5身份严格分开。

先提交本协议/config/冻结评价清单与来源 hash，再提交诊断实现及源 hash，之后才能新推理。
登记不读 GT 或 measurement 内容；核对冻结 checkpoint provenance 不做前向。
所有模型 eval、no_grad、FP32，seed_everything(20261011)，TF32=false、cudnn benchmark=false，
不强制更改历史 deterministic 设置；batch1、原 view chunk2、两 CPU threads。

顺序：normal 900 预测全部保存→独立进程读取 saved C0/C1 和 Val GT64→核对全部 normal C1
十项指标复现旧900行→zero 900 与固定 shortcut 18 次→独立 saved-output GT64 评价→统计图表。
normal 先以 case0/draw0 五组作为 smoke，这五条属于900条，不重算。
zero/signflip/wrong_pairing 只使用已有 evaluation-only residual_mode；
signflip/wrong_pairing 仅 Val 索引0/18/35、draw0、G1/G3/G5，各9次。
总计划1818次 forward；C0 是同次 return_stages 的中间结果，不另运行 C0 模型。
成功预测逐项原子保存并写 hash/计数；恢复仅计算缺项，已有完整预测/结果先核验后复用。
推理进程拒绝所有 GT；独立 GT 评价拒绝 checkpoint/measurement；
GT128、Pool B、final test、非 Val 内容与任何 optimizer step 均禁止。

## 指标、归约与复现

主终点：每组 3D relative L1 的 ΔE=E(C1)−E(C0)，负值改善。
辅助：3D relative L2、MAE、RMSE、voxel-sum integral relative proxy、peak absolute error，
input/H4 measurement relative L1/L2，全部直接复用原 VolumeMetrics/relative_projection_metrics。
每行记录 case/group/draw/input/H IDs、config/source/protocol/checkpoint/prediction/来源 hash。
normal C1 对旧900行逐指标容差 abs1e−8+rel1e−5，继承原 FP32 复现规定，不按结果放宽。
超差立即停止后续 zero/shortcut，先排查；不改输出以满足复现。

context0 用原 fusion 第一次调用的只读 pre-hook 获取精确 tensor，return_stages 获取 delta/context1。
不修改 forward 或中间 tensor。context RMS 在 visibility_base AND real view_mask 的
全部有效 (batch,view,channel,32³ voxel) 元素上 double 累加；不包含 padding/不可见区域。
delta 与 context0 同一 mask；ratio=delta_RMS/context0_RMS，分母≤1e−12或空区域时为 null 并计数，
不把小分母截成巨大数。C1−C0 MAE/RMS 为完整64³体素；signed measurement residual
只在 valid_ray AND view_mask 上记录 raw ppm·m 的 MAE/RMS/mean 及固定尺度 RMS。
记录 mask/count/hash，zero 与 normal C0、输入、几何及 mask 的一致性；zero 仍保留encoder/path/adapter，非移除 MRGR，也不假定等于 C0。

## 统计、相关与图示预先固定

先对每 case/group 五布局平均，再以36 case配对；PCG64 seed2026101701，10000次 case bootstrap，
同一重采样索引用于所有指标，percentile95%CI，报告均值、中位数、改善/恶化/持平数量。
持平阈值 |Δ|≤1e−8（同时导出 exact sign 计数）；900行不作为独立样本。
五个主 L1 配对 Wilcoxon p 跨G1–G5 Holm，CI为逐项未校正95%，不称同时置信区间。
其余所有 zero/辅助指标 p 单独合为一个 Holm family；shortcut 仅9项/模式，描述性，不给显著性结论。
没有按显著性/GT重选指标、布局、checkpoint或阈值。

输入改善但3D/H4恶化用同一Δ定义及1e−8阈值，分别导出原行频率（仅描述）与case/group频率、
条件平均/最大幅度和每组比例的case bootstrap CI，明确聚合层次。

几何复用原描述量，分析名单在独立config固定（10项，包括常量coverage，不新建评分或拟合权重）。
几何→delta/context0 ratio、delta RMS、3D ΔL1：先对36case平均得到25固定布局，
分别报告G1–G5全部、G1–G4及每个组内五布局 Pearson/Spearman，case bootstrap CI。
幅度→3D/input/H4 ΔL1：按case/group均值和G1–G4等权case macro做36-case相关；
另报告全组等权case macro。常量/无效值导出null，不填0。
全部相关为探索性、逐项未校正区间、不做p值或显著性筛选；有限组内五布局不能支撑稳定几何排序。
不将修正大/测量残差下降直接解释为错误/三维正确。

固定图：组C0/C1 L1、36×5 Δ热图、3D与input/H4 Δ关系、几何/幅度/Δ关系、
case0/18/35×draw0×G1/G3/G5 GT/C0/C1/C1−C0。
代表图同时用 y=32 的 XZ 水平切片及沿 y 的均值投影，均不按 GT/误差选slice；
每case九个浓度面板共用0到该case GT/C0/C1最大值色标，signed差分以0对称发散色标，
若显示绝对误差则独立色标。轴标 x/z (m)，标题注明 y高度及浓度单位质量分数。

## 解释和交付

C0 是联合训练中通过C1目标间接监督的中间输出，不是独立无MRGR训练消融；
内部配对与shortcut仅诊断当前模型，不能认定模块因果收益或最佳几何门控。
明确回答C0/C1准确性、G5误差来源、各组修正是否失败、输入/3D/H4方向不一致的频率幅度、
几何与修正关系、残差信息依赖以及几何条件残差方向证据。
如不支持该方向，只给一个最合理替代建议；即使支持也仅为后续设计假设，不确定门控/阈值/参数。
单seed、历史Val开发集、25固定布局及工况相关性限制外推，case CI不含训练seed或布局总体不确定性。

外部独立目录见config；大预测/GT/checkpoint不提交Git。提交900配对、180case/group、
zero/shortcut明细、幅度/几何统计、图表中文报告、状态及hash/访问/完整性/实际forward账本。
必要新测试仅覆盖stage获取、mask归约、零分母、case聚合与访问边界；运行现有适用回归。
只更新本轮README导航，不改历史报告/结论。完成后经本机7897代理推送main并停止。

推理前预算勘误：18b4dec 中曾将3案例×3组×1draw误计为每模式27次；实际为9次，
两种shortcut共18次，总1818次。子集与统计定义完全不变，尚未执行新推理。
