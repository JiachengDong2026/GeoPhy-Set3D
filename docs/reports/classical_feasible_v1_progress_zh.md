# 经典基线正式训练与调试：进度快照

2026-10-27T19:23:30.865464+08:00

仅train测量自生成场：166/166，GT未用于训练。此文件为进度快照，最终状态以各阶段summary及测试封存记录为准。

## U-Net完整训练

当前阶段：physical；epoch 175；该阶段更新 14525；完整36Val H=0.09502709。

![U-Net训练曲线](artifacts/classical_feasible_v1/unet_progress.png)

当前训练结束状态：STOP_PERSISTENT_COLLAPSE；物理epoch=175；bestVal H=0.08731182。

## U-Net：独立稀疏场判据修订 v2

上图和原始STOP记录属于v1。v2保留原175轮停止结论，恢复同一完整边界继续训练；只修订饱和区域的统计权重，不改变网络或原目标。

v2当前物理epoch 300/300，累计更新 24900，完整36Val H=0.08031696。

[完整对比、判据变化和测试结果](classical_feasible_v1_training_test_zh.md)。

## 其他算法：单案例固定预算诊断

| 模型 | 稠密预热后H | 原目标末态H | E（未参与优化） | 状态 |
|---|---:|---:|---:|---|
| deeponet | 0.5796684 | 0.3968394 | 0.4067489 | 局部通过，需多案例验证 |
| fno | 0.5336229 | 0.1551847 | 0.1706934 | 局部通过，需多案例验证 |
| senseiver | 0.6673309 | 0.3366177 | 0.3442248 | 局部通过，需多案例验证 |
| gnot | 0.5726452 | 0.3152614 | 0.3853716 | 局部通过，需多案例验证 |
| deepsets | 0.7566687 | 0.3860486 | 0.4220619 | 局部通过，需多案例验证 |

## 其他算法：多案例检查

- deeponet：physical epoch30，完整Val H=0.6673426。
- fno：physical epoch30，完整Val H=0.244623。
- senseiver：physical epoch29，完整Val H=0.5976599。
- gnot：physical epoch29，完整Val H=0.6007109。
- deepsets：physical epoch30，完整Val H=0.8317025。

正式测试仅在U-Net完成300轮、模型选择与独立复核冻结后运行。单案例误差不构成三维GT精度或跨案例成功证明。
