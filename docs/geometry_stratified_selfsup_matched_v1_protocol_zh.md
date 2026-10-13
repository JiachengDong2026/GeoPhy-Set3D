# 自监督几何采样匹配实验 v1：优化前协议

Stage C的冻结评价满足预登记GO：两个模型在V3/V4/V6均出现low>middle>high、low−high case CI高于0且指定方向描述量rho≤−0.30。参见独立[验证报告](reports/geometry_strata_frozen_v1_zh.md)。本次只干预A/B几何子集的联合采样分布，不改变冻结网络、投影器、损失、V分布、优化器、训练预算或模型选择器；RVTC路线保持关闭。

## 匹配规则与采样

Control逐case/epoch严格使用原 `DisjointABHSampler`，A/B各3–5均匀，H2，无交集。Treatment先均匀抽q∈{low,middle,high}，A/B均在q且互不重叠，H在输入之外均匀抽2站。Treatment该case/epoch的A/B数量直接取原Control sampler的数量，因而完整49,800个sample的V_A/V_B完全一致。q用独立purpose identity，PCG64/SeedSequence(seed,epoch,source index,blake2b(case))；在可用站位中均匀无放回抽subset，按原center score及V专属冻结边界拒绝抽样，单subset上限10,000次，若达到上限记录失败，不fallback、不重新加权。

自然tertile定义下Control单个输入的q边际本来就近似1/3；Treatment主要改变A/B同层的联合分布和少量站位边际，不能简单称“提高优质几何占比”。几何/站位分布审计保存完整A/B的3×3 stratum联合表、per-station A/B/H频数、方位/半径/高度、pairwise separation和center score直方图。只报告差异，不后验强行匹配或按GT调权重。极端clustered不是训练层；自然low偶然包含极低分数并不意味着主动采样极端压力布局。

训练涉及V5，独立使用Pool A origins与同固定总体seed生成100,000唯一V5组合并冻结自己的tertile；不借V6边界。V3/V4沿用Stage B v2。V6只评价不训练。边界/计划阶段不读取测量、GT或checkpoint，初始化仅从fresh构建。

## 初始化、工序与预算

seed20261011，两组独立fresh同初态tensor SHA256，无warm start。166train、300epochs、FP32、effective batch2，AdamW lr2e-4/wd1e-4，clip1、cosine T_max300，K96/all rays/view_chunk2。网络GeoLift-OARI-MRGR-Res3D/332,070参数；调用原 `training.selfsup.objective/run_epoch`，总目标原样为 L_hold+.5L_cross+.1L_obs+.05L_cons。loss的SmoothL1、relative L1、measurement scale也原样保留。DataLoader workers0，两组一致，确保本进程审计覆盖所有访问；与旧v1运行时并发worker数量不同，但不是Control/Treatment之间的干预。

以同一个torch generator预生成300个case顺序，两组共享；每epoch83次实际optimizer step，每组24,900步。plan与order都在优化前冻结hash。外部完整49,800×2训练station计划不提交大型JSON，Git提交规则、hash、epoch计划hash和分布审计；独立完整目录保留可重放计划。每组只可由自己的last.pt恢复，恢复配置/source/input/protocol/hash和RNG/optimizer/scheduler一致，不从其他任何权重开始。完整结果存在先核验后退出，不重复训练。

Control cuda:0、Treatment cuda:1，同型号共享RTX4090；设备及输出路径是执行元数据，唯一科学干预为sampler。TF32关闭，网络初态/parameter count验证；不同设备竞争影响时间，不能称专用硬件基准。OOM或非有限失败记录实际预算，不根据Val修改batch、loss或采样。

## 相同 checkpoint selector 与访问边界

两组都使用原SelfSup v1的Val-A natural随机dual-input A4/B4/H4，epoch固定0，以两个输入预测的平均held-out measurement relative L1选择best。不是受控层macro，也不使用GT。36个selector A/B/H IDs在优化前登记，对两组完全相同。

训练/验证/选择只允许Pool A train/Val投影与冻结Pool A geometry；GT64/GT128、Pool B、最终test、监督/旧SelfSup/RVTC checkpoint内容一律禁止。沿用原GTReadGuard并增加独立计数/数据写阻断，不更改原训练保护。只可读自身谱系checkpoint。保存每epoch审计、实际步数、case order、station digest、输入loss/measurement动态；数据/checkpoint/历史源码与产物不能覆盖。

## 冻结后独立评价和决策

必须两组300epoch完成、best选择结束并冻结两个SHA后，先在Stage C同2340输入记录/模型上measurement-only评价，保存输出/hash及input/H误差，仍禁GT；完成该阶段后，单独进程才打开36 Val GT64评价已保存64³输出，不再选checkpoint或改sampler。自然V3/V4/V6、受控9格与clustered V4分开，五draw先case内平均，再36case配对bootstrap10,000、seed2026101404。最终自然V宏平均为三个V等权，每个case先宏平均后配对；low/middle/high宏同理；clustered只secondary。

Stage D GO在看到新结果前操作登记为：自然macro Treatment−Control CI下端≤0（不得支持明确误差增加）；low macro均值差<0，CI完整报告；high macro若均值相对增加>5%且差值CI下端>0则判材料性受损；三项均通过才GO。5%是本实验工程容忍尺度，不是理论常数；同时报告middle、每V每层/自然、极端压力和测量误差，不把clustered改善替代普通场景。单seed、自然low内有限draw和Val模型选择限制外推，不按posthoc GT结果调边界、weights、score或重做v1。

## 优化前提交与重放

本协议、config、sampling实现、V5总体、初态hash、selector/最终评价IDs、case order、计划及源码hash、分布审计、测试和Stage C验证将一起commit并同步，之后才运行optimizer。入口：

```bash
/scratch/djc/envs/gas-torch/bin/python scripts/run_geometry_stratified_matched.py register
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests /scratch/djc/envs/gas-torch/bin/python -m unittest test_geometry_stratified_matched -v
/scratch/djc/envs/gas-torch/bin/python scripts/run_geometry_stratified_matched.py train --arm Control
/scratch/djc/envs/gas-torch/bin/python scripts/run_geometry_stratified_matched.py train --arm Treatment
# 两组完成后
/scratch/djc/envs/gas-torch/bin/python scripts/evaluate_geometry_stratified_matched.py measurements
/scratch/djc/envs/gas-torch/bin/python scripts/evaluate_geometry_stratified_matched.py gt
```

不执行新模块、loss、RVTC/teacher、网络TV/PDE先验、经典矩阵、Pool B或最终test。结束后无论GO/NO-GO都停止，不自动开展v2或多seed。
