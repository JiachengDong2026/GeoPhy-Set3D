# 显式几何增强 v1：独立采样实现与无GT smoke登记

本轮主任务为原SelfSup v1的Geometry Sensitivity Analysis，已完成900次冻结推理。G5−G1 L1为+0.209632，case95%CI[+0.177679,+0.244722]，36/36案例变差；去除G5后中心分数和误差仍相关。证据足以支持下一阶段采样器实现，不能证明增强会有收益或容量不是瓶颈。本文件登记**准备阶段**，不启动正式网络训练。本入口没有optimizer/train分支，训练步数为0。

## 与旧负结果的区别

旧same-q A/B Treatment约束两个输入同自然tertile，而Control单输入本身自然tertile近似均衡，平均score几乎相同。它完成300轮后自然macro/low/clustered均受损，NO-GO不改。本次独立采样每个A/B的category，概率natural=.50、moderate=.25、hard=.25；不强制A/B同档。不是重跑旧sampler或用更好的测试布局当网络改进。

使用原中心方向描述量，不用本轮按Val GT关联挑出的exploratory hull-area score。Moderate为各V总体score[q05,q40)，V3/V4全枚举，V5固定100,000唯一MC、seed2026101506；hard为最小圆周方位跨度≤10°的全部组合，纯几何预生成。训练hard的10°与本轮压力组G5的5°不是同一条件，明确登记，不能声称覆盖所有退化布局。10°是前瞻工程采样定义，未用新训练Val误差搜索。自然部分仍可能包含困难布局；模式身份不等同误差身份。

A3–5/B3–5数量沿原DisjointABHSampler逐case/epoch精确匹配；两种mode独立抽一次，固定不重抽。两个子集各按其模式提议后，若交叠只重抽站位对（最多10,000次），不改变mode、V或用fallback；H2在A/B之外均匀抽，完全disjoint。联合拒绝导致站位边缘分布在disjoint条件下变化，这是设计的一部分，记录频次/联合类型/最大拒绝次数，不冒称每个条件中的站位仍无条件均匀。所有V组合的hard/hard disjoint可行性预先检查；不可行立即停止。

## 实际执行与准备预算

外部独立子目录由 `configs/geometry_augmentation_preparation_v1.yaml` 指定，部署路径不硬编码在源码。实际物化166train×300planned epochs=49,800对（99,600输入）的站位计划，不读取任何train测量或GT。**planned不等于训练已经执行**。逐条核对旧Control全部49,800条A/B/H IDs与原sampler精确相等，新计划A/B数量完全匹配。大站位计划只在外部目录，GitHub归档hash、纯几何边界、分布频次和9个smoke样例。

网络仍为332,070参数的原v1架构；fresh seed20261011初始tensor hash与旧Control精确相等。准备阶段不读取任何checkpoint内容。独立源码和计划先commit，然后才执行首见的9种A/B mode组合真实measurement-only前向与原objective反向；fresh模型eval模式以验证梯度流，不做optimizer step、不保存新权重、不选模型。不将H传入重建网络，原loss不改。原自监督GTReadGuard不改，复用已有独立SamplerTrainingReadGuard阻断GT64/GT128、Pool B/test、外部权重与数据写入；9种组合应有限loss、输出[0,1]、有限非零梯度、前后parameters/buffers hash一致，否则停止，不按Val调参重试。

正式300轮训练尚未运行。后续实际训练若启动，须先完整核对旧Control的数值训练/selector/case order/初态/FP32/optimizer/scheduler及预算等价性，才能合法复用Control；本次仅核对station plans和初态，不冒称全等价验证完成。否则采用fresh匹配Control/Treatment。selector沿用原自然Val measurement H4，禁止GT64参与训练或选模；两模型完成并冻结后再独立相同输入GT64评价。不能依据本轮Val GT继续优化比例、阈值、停止点或score。当前不执行curriculum/hard weighting、不换backbone、不追加teacher/先验/loss权重。

```bash
python scripts/prepare_geometry_augmentation.py --config configs/geometry_augmentation_preparation_v1.yaml --stage prepare
# 注册已commit后：
python scripts/prepare_geometry_augmentation.py --config configs/geometry_augmentation_preparation_v1.yaml --stage smoke --device cuda:2
```

已完成准备/smoke应核对退出，不重复或覆盖。准备不是正式增强收益结论；当前决策是值得开展一项独立匹配增强训练，原v1基线和旧NO-GO继续保留。
