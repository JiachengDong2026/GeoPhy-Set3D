# MRGR 残差反馈分支加入 OARI：20轮探索性筛查

更新：2026-10-19T20:09:08.146719+08:00。状态：`pilot_complete`（以已通过的最终验收为准）。

仅使用166训练案例与36开发案例；不读取test投影、test GT64或GT128。原始冻结模型与正在进行的C/D实验不修改。

## 结构与固定预算

C0与原始第一阶段不变；计算原P(C0)及有符号Y−P(C0)，残差编码与提升后同context0拼接，经原48→32卷积与SiLU，再经过独立OARI，最后由原32→32零初始化卷积产生Δh，context1=context0+Δh，再用原GCA和解码器生成C1。第二个OARI保留原射线关系、描述符和掩码。

三个版本使用同一种子20261011、相同原始参数初始化，各完整运行前20轮、1660次更新，batch2/FP32/AdamW2e-4/裁剪1与全部损失、采样、物理算子不变。保留原cosine T_max300，因此是早期训练前缀，不能与已有300轮模型直接比较。只在预登记20轮内按验证H4最低值选轮。

新增OARI从同种子原OARI初始权重复制为独立可训练参数，不消耗新RNG；原有全部同名张量保持精确一致。零残差版本只将残差输入置零，仍计算P(C0)，掩码/路径/原特征与新增参数均保留。

| 版本 | 参数 | 进度 | 最佳H4 |
|---|---:|---|---:|
| baseline | 344678 | 20/20，pilot_complete | 8.7909% |
| feedback_oari | 355658 | 20/20，pilot_complete | 8.6101% |
| feedback_oari_zero_residual | 355658 | 20/20，pilot_complete | 15.2074% |

## 资源和验收

每15秒查询UUID、进程/用户、显存、利用率和主机资源；使用C/D控制器共同遵守的gpu_locks，在最多一张卡上为本次短测保留优先级。保留锁不占GPU显存，也不影响外部任务；只在无计算进程、已用<100MiB、空闲≥20000MiB、利用率≤5%连续两次并复查后运行，不共享忙卡或发信号。结束后释放锁，原C/D队列继续使用资源。

CPU检查：通过；GPU梯度/跨进程恢复：通过。

CPU验收覆盖真实64³、原参数初始化、非零适配器时旁路新增OARI与原实现逐张量完全相同、初始C1=C0、零残差通道和视角重排；GPU验收覆盖完整自监督损失反传和1+1/2轮跨进程恢复。浮点恢复rtol1e-4/atol1e-6，整数与RNG精确。失败记录保留并阻断训练。

[预登记/SHA](artifacts/feedback_oari_pilot_v1/design.json)；[实时状态](artifacts/feedback_oari_pilot_v1/status.json)；[资源和调度](artifacts/feedback_oari_pilot_v1/events.jsonl)；[CPU验收](artifacts/feedback_oari_pilot_v1/cpu_checks.json)；[GPU验收](artifacts/feedback_oari_pilot_v1/gpu_checks.json)。

## 结果与解释

| 布局 | 版本 | L1 | L2 | 峰比绝对误差 | GT峰处比 | 质心mm | Dice10 | Dice50 | H4 L1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val-A-V4 | baseline | 20.8778% | 25.9028% | 0.1219 | 0.6744 | 4.8834 | 0.8537 | 0.6745 | 8.4884% |
| clustered-V4 | baseline | 74.0730% | 76.3261% | 0.7109 | 0.0010 | 35.1903 | 0.2315 | 0.0000 | 44.7915% |
| Val-A-V4 | feedback_oari | 20.6410% | 26.7592% | 0.1365 | 0.7133 | 4.2027 | 0.8493 | 0.6439 | 8.1208% |
| clustered-V4 | feedback_oari | 70.3911% | 71.9738% | 0.7678 | 0.0047 | 22.7522 | 0.2918 | 0.0000 | 38.5533% |
| Val-A-V4 | feedback_oari_zero_residual | 30.9107% | 34.2312% | 0.3835 | 0.3733 | 7.7759 | 0.7845 | 0.4919 | 14.5591% |
| clustered-V4 | feedback_oari_zero_residual | 89.0150% | 79.9094% | 0.8905 | 0.0011 | 34.9984 | 0.0909 | 0.0000 | 53.1763% |

| 比较（前−后） | 布局 | L1差百分点 | 探索性95%案例区间 | 胜/平/负 |
|---|---|---:|---|---|
| feedback_oari - baseline | Val-A-V4 | -0.2368 | [-0.6103,+0.1553] | 22/0/14 |
| feedback_oari_zero_residual - baseline | Val-A-V4 | +10.0329 | [+8.8232,+11.3340] | 0/0/36 |
| feedback_oari - feedback_oari_zero_residual | Val-A-V4 | -10.2697 | [-11.4563,-9.1165] | 36/0/0 |
| feedback_oari - baseline | clustered-V4 | -3.6819 | [-6.5484,-1.4343] | 28/0/8 |
| feedback_oari_zero_residual - baseline | clustered-V4 | +14.9420 | [+11.2279,+18.7811] | 3/0/33 |
| feedback_oari - feedback_oari_zero_residual | clustered-V4 | -18.6239 | [-22.2072,-15.2996] | 36/0/0 |

尚无同时满足普通/聚簇L1、峰值与定位的充分筛查信号；短程结果不足以否定收敛后效果。

扩大实验的预登记筛查信号要求：真实残差OARI在普通/聚簇两布局均降低平均L1，且不恶化平均峰比绝对误差与质心，并在L1上优于零残差版本。该规则只是下一步资源分配依据，不构成显著性、非劣性或最终性能证明。

聚簇五draw先在同案例内平均后进行36案例配对bootstrap20000次；未经多重校正，不覆盖训练种子方差。所有17布局L1/L2、逐案例中位/P90/最大、峰值与定位、输入/H4一致性以及C0/C1原始标量全部保留。未定义形态指标会明确记为缺失，不静默删例。

[完整统计](artifacts/feedback_oari_pilot_v1/summary.json)；[逐例CSV](artifacts/feedback_oari_pilot_v1/per_case_metrics.csv)；[独立重算](artifacts/feedback_oari_pilot_v1/independent_verification.json)。

| 版本 | 训练秒 | 训练峰MiB | 最佳轮 |
|---|---:|---:|---:|
| baseline | 677.56 | 7917.2 | 20 |
| feedback_oari | 758.90 | 9779.9 | 20 |
| feedback_oari_zero_residual | 591.83 | 8229.8 | 18 |

## 局限与复现

预检查首版CPU结构检查通过，但控制器因报告模块同名导入而退出，尚未启动GPU任务；完整源码、状态与日志保存在`mrgr_feedback_oari_pilot_v1_preflight_revision1_20261019_1711`。更名报告模块后重新登记验收，模型、种子及预算未变；另修正未定义形态指标的JSON表示，保留缺失而不删除案例。

单种子20轮只检验早期训练趋势；没有参数匹配的非注意力新增模块，故不能把与baseline的差异完全归因于OARI机制。正结果仍需匹配预算、更长训练和多种子确认，负结果也不能否定收敛后可能有效。不能使用已用35例测试集作独立确认。

worktree：`/home/djc/projects/GeoPhy-Set3D-feedback-oari-pilot-v1`；分支`experiment/feedback-oari-pilot-v1`；基线`175eaadd624fa2d0812a9a65b903aaf0259f253e`。解释器`/scratch/djc/envs/gas-torch/bin/python`，入口`/home/djc/projects/GeoPhy-Set3D-feedback-oari-pilot-v1/scripts/feedback_oari_pilot_v1/controller.py`；权重/数组/日志位于`/scratch/djc/GeoPhy-Set3D/mrgr_feedback_oari_pilot_v1`，不提交Git。原始主仓库和C/D源码保持不变。

## 仓库同步说明

本文件是便于GitHub浏览的报告副本，证据与图片使用仓库相对链接。原始验收时报告逐字保存在同目录证据中的 `source_report.md`；原验收及源码SHA继续指向原执行位置，保留历史溯源含义。大体积权重和预测数组仍在原scratch目录。汇总与导出校验见[本轮结果索引](mrgr_development_results_20261020_zh.md)。
