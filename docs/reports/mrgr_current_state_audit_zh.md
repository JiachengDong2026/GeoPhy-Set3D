# 多尺度 MRGR 当前证据审计与非 OCMSF 并行筛查

审计日期：2026-10-17，北京时间。本轮用户明确要求并行验证其他优化路径，故在完成只读证据核查后，开展独立的短程实验；没有启动OCMSF实现或新的300轮训练。

## 1. 仓库与运行状态

本地HEAD与fetch后的origin/main均为`6f8408b4653b6a1dc16017f75a64990f6bf55f59`，分支main。审计开始时的未跟踪文件为用户新增执行计划和本地outputs/research产物，未执行同步覆盖、reset或清理。检查时没有本用户训练进程；GPU0有他人进程驻留，GPU1/2/3计算与显存基本空闲。后续三条路线分别使用GPU1/2/3，不终止他人任务。

## 2. 正式模型身份

正式入口为[scripts/train_mrgr_multiscale_formal_v1.py](../../scripts/train_mrgr_multiscale_formal_v1.py)，构造函数复用[probe_mrgr_multiscale_v1.py](../../scripts/probe_mrgr_multiscale_v1.py)的`model_for(config,'multiscale',device)`。主体类为`GeoLiftOARIMRGRRes3D`，共享decoder经`ContextDecoder`包装。

在原三个32³残差块后增加12通道16³/8³上下文，回到32³并残差相加，再执行原64³解码。C0/C1共享同一decoder。参数344,678，零输出初始化。正式配置以run/protocol.json内的完整config为准，来自原selfsup配置，没有根据后来的v2配置倒推模型身份。

正式run：[protocol.json](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/protocol.json)。300轮/24,900更新，最佳第278轮，投影验证H为3.1093698%，训练约2.789小时。最佳权重[best.pt](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/best.pt) SHA256：`650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7`，本轮重新计算匹配。

原formal verify实际通过：116个登记源码与205个数据/元数据文件身份均保持一致。没有改写旧源码、权重、指标或协议。

## 3. 数据、目标与评价

237工况划分166train/36val/35test，仅末帧t=2s。测量自监督训练，loss仍为hold1+cross0.5+observed0.1+consistency0.05，内部含归一化SmoothL1及relativeL1。训练A/B各3–5视角且H2不输入重建；验证固定A4/B4/H4，选模仅用验证H投影相对L1。

已完成独立冻结三维评价，17布局×36案例=612次重建，C0/C1共1,224场先保存封存、后评分GT64。原训练summary的volumetric_evaluation_completed=false是当时训练结束快照，不能据此误认后续评价未完成；完成证据在evaluation/frozen_model_test_v1/summary.json。

本轮另从1,224条封存逐场标量独立复算全部8个C1面板均值，与原summary一致：普通V4三维L1为9.369713%，聚簇五布局为33.001164%。见[本轮标量复核](../../outputs/research/mrgr_parallel_optimization_20261017/baseline_scalar_recheck.json)与[正式测试报告](mrgr_multiscale_completed_model_test_zh.md)。本轮没有重新推理、重算GT评分或读取GT数组。正式报告此前的独立12场重算记录保留；此次标量聚合复核不是宣称再完成一次从体数组的误差重算。

最终35test未用于本轮；已有验证GT参与过开发，不再称盲测。当前主要证据单种子。圆环等合成分布外失败仍保留，未将其改写为通用场重建能力。

## 4. 已失败候选与本轮区别

局部细节版：20轮普通H均值改善1.05%、聚簇改善2.90%，普通P90恶化4.78%，超过预先固定2%门槛，未晋级。反馈gate：16train/8val的小筛查普通H未改善，困难H小幅改善，实际调制已证实，但没有正式训练价值证据。两者都不是所有局部路径或所有反馈控制无效的证明。

本轮不重复这两个模块，也不实施OCMSF。三条独立路线为：

| 路线 | 唯一改变因素 | 对照与预算 |
|---|---|---|
| 固定后期权重平均 | 同一次正式训练的250/275/300轮参数等权平均，buffer须一致 | 原冻结best278；不训练，只评估测量 |
| 投影细节损失 | 原objective附加固定权重的有效相邻射线梯度误差 | 同GPU2 fresh两臂，各166train/36val、20完整轮 |
| 低比例困难课程 | 不改变样本/射线数量，在少量训练样本引入预定聚簇几何，比例逐步增加 | 同GPU3 fresh两臂，各166train/36val、20完整轮 |

这些实验是与OCMSF不同的信息路径，分别检验参数稳定性、测量目标和训练几何分布。投影细节及采样改变是独立分支，不与OCMSF同时叠加后宣称OCMSF单因素收益。

## 5. 晋级与后续边界

每路线在优化器更新前登记精确设置、源码/数据哈希、测量面板、成本与门槛。一般门槛为普通H均值改善至少1%、普通P90及五聚簇案例内均值不恶化超过2%、正确性及零GT读取通过；训练路线另检查新增耗时不超过20%、300轮外推不超过5小时。若针对困难问题设不同主要终点，须在运行前明确。

固定末轮比较，不搜索多组权重、窗口或采样比例。以案例为统计单位，报告配对区间、胜负和失败。通过仅产生进一步完整训练的候选建议；本轮不自动启动300轮，也不打开最终test。测量收益不能直接认定峰位、三维L1/L2或物理扩散规律改善。

并行登记：[registration.json](../../outputs/research/mrgr_parallel_optimization_20261017/registration.json)。后续结果统一汇入独立并行筛查报告，不覆盖原结果。

## 6. 审计后的执行结果与新增授权

上述第1–5节为启动时审计快照。非OCMSF三路线现均完成，父验证器已独立复算全部逐案例均值、P90、五布局聚合、门槛及采样计划，见[并行结果报告](mrgr_parallel_optimization_20261017_zh.md)。固定平均与课程未晋级；梯度损失仅通过工程门槛，鉴于区间跨零及训练轨迹漂移，保留待重复，未启动长训。原正式模型source/data verify再次通过。

用户后续明确指令“目前还有空余的GPU吗，有的话可以同时规划验证OCMSF方案”，授权使用空闲GPU实施OCMSF及短程对照；这不同于启动审计时的范围。OCMSF采用独立模块、协议与四臂消融，不修改原数值源码、损失、采样或已完成实验，仍不自动300轮。见[OCMSF设计](mrgr_ocmsf_design_zh.md)、[OCMSF筛查](mrgr_ocmsf_screening_zh.md)。

后续四组各20轮均已完成，共6,640次更新。OCMSF普通留出投影相对L1为8.8710%，较多尺度8.9636%改善1.03%，但较同参数非条件融合8.6521%恶化2.53%，联合晋级未通过。非条件融合保留为待重复候选；本轮没有三维收益结论。首2验证案例同权重关闭额外条件的检查确认条件实际影响输出，权重hash不变、GT访问计数为0。关联68项回归及父agent独立结果复核通过。实验全部退出后GPU1/2/3显存各约1MiB、利用率0%；未启动新的300轮训练。
