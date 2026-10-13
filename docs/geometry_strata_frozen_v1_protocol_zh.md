# 冻结模型几何分层验证 v1：优化前登记

本协议接续 13d6282；原研究修订、RVTC 和经典算法报告保留。研究问题是几何描述量能否排序冻结网络的重建难度。网络/投影器/训练目标/332,070 参数结构不改。本次先完成 A–C，只有登记的 Stage C GO 才进入另行提交协议的匹配采样训练 D；NO-GO 即停止。

## A：前瞻纠正清单

本项目 x–z 为水平面，y 为高度。新清单将 `origin[:,2]` 修为 `origin[:,1]`，并验证全部冻结站位高度 0.25–0.35m。增加升序 λ_min/λ_mid/λ_max 和以 max(λ_max,1e-12) 为分母的两个比值；另保存水平半径与方位。历史 v1 清单及其 z 轴错误原样保留，不追溯改写。原经典协议对全部 src 的 hash 将无法用于验证修改后的清单源码，这是明确授权的前瞻修复；旧协议、数值代码、结果和 hashes 不改，也不重跑经典矩阵。

## B：只依赖几何的稳定总体

`configs/evaluate_geometry_strata_v2.yaml` 登记 V3/V4 全组合与 V6 100,000 唯一组合。只在源 origin 上计算原中心分数 mean(1−|dot|)。边界计算不构建 K96 覆盖、不读取测量/GT、不加载模型/checkpoint。V 各自按 1/3、2/3 分位分层，恰好等于边界归右。固定 PCG64 SeedSequence(population_seed,V)；V6 去重后按组合字典序排列。统计总体、分位、几何/源码 hash 归档。没有改变分数公式或做描述量搜索。

每 V×stratum 无放回抽 5 组，固定 seed 和输入顺序，所有36 Val case及两个模型共用。覆盖仅对选择完成后的实际评价布局计算：16³ endpoint-aligned 诊断网格、K96中点三线性非零权重、至少一个/两个站位触达及三轴平面覆盖；全部有效真实三维射线构成方向张量。它们是诊断 proxy，不是正式可观测性、条件数、秩或唯一恢复证明。

## C：配对实测与统计

受控9格×36case×5draw×2模型=3,240次前向；另自然随机 V3/V4/V6 为1,080次，原 clustered V4 为360次，共4,680次真实前向。自然随机使用原 seed20261011与原 view sampler，站位顺序逐条核对历史。clustered重放单站位诊断 Original IDs和H4，不进入tertile统计。复现原四项体积与两项input误差，容差abs1e-8+rel1e-5，越界停止，不根据结果修改容差。

两个冻结checkpoint分别用指定 hash核对，自监督训练不使用体积GT，监督训练及其Val模型选择使用GT；不把二者差异单因果归于标签。推理FP32/eval/inference_mode、关闭TF32，模型parameter/buffer前后canonical hash检查。仅Val Pool A测量与Val GT64可读，禁Pool B/test/GT128、数据或checkpoint写入及optimizer。新评价保护独立，原自监督训练guard不改。

受控共同H4在同case/V的15输入并集之外，三个stratum和五draw同H。自然随机H4按input identity独立键控，在输入之外；clustered沿用原共同H4。均在测量结果前冻结，只用于最终评价。

保存relative L1/L2、MAE/RMSE、input/H4 relative L1/L2、冻结积分proxy与峰值误差。relative指标无量纲，MAE/RMSE/峰值为质量分数；积分proxy并非kg。五draw先在case内平均，再36case均值及10,000次配对case percentile95% bootstrap；seed2026101404。low−high差为正代表low更难。

连续关联逐V、逐模型，仅受控15个独立几何ID；先对case聚合每几何L1再算Spearman rho。bootstrap只重采样36case，每次重新聚合15几何误差再算rho。报告区间，不做将case×draw视作独立样本的p值，不混合V。描述量为center score、λ_min/max、λ_mid/max、coverage、coverage≥2；不按结果筛选/改公式。

**预登记的稳定排序操作判据：** 至少一个模型在V3/V4/V6三个V均满足 low>middle>high，low−high区间下端>0，且该V至少一个指定gate描述量（center score、λ_min/max、coverage≥2）rho≤−0.30。CI只作为关联的不确定性报告，不另增加门槛。三个V均满足是“稳定排序”的保守定义；0.30是工程参考，不是物理定律。未通过则停止，不重新抽样、改变边界或发明新score。统计多重比较未校正，单checkpoint与同工况家族相关性限制外推。

## 解释边界与后续

新文字称历史Random为Matched-Random，历史字段原样保留：它与Targeted共用删除ID，只随机新增ID。Original→Matched-Random打破极端集中；Matched-Random→Targeted才是额外几何指导收益。SelfSup历史两段改善为56.54%、16.71%，不将63.8%全部归于center score。

经典结果只能称登记有限计算预算下的性能；1,270次预算耗尽不能称理论下限或包装为网络根本优势。未来主表5个方法，不重复两种恰好同配置的经典选择身份。

Pool B、最终test本轮不读取。未来Pool B主分析沿用Pool A边界；test须待最终结构、训练、消融与多seed冻结后。Stage D只有C通过才能登记；正式优化前另commit协议、配置、hash、初态和station sampling rules；保持原natural Val测量selector，fresh同初态、同V/工序/预算且只sampler改变。本协议并非提前批准不满足C门槛的训练。
