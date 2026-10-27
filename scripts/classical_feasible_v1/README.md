# 本轮训练入口与证据

工作分支：`experiment/classical_feasible_formal_v1`。大型结果位于
`/scratch/djc/GeoPhy-Set3D/classical_feasible_v1`，输入使用迁移后的 `/data/djc` 数据。

本轮已有训练由持久化 controller 管理。以下两个命令只刷新报告，不会启动训练：

```bash
/scratch/djc/envs/gas-torch/bin/python scripts/classical_feasible_v1/status_report.py
/scratch/djc/envs/gas-torch/bin/python scripts/classical_feasible_v1/delivery_report.py
```

`train_unet_workers.py` 从已核验的完整 epoch 边界恢复 U-Net，随后执行原协议的
300 轮物理训练。`continue_at_boundary.py` 负责后续独立验证、模型冻结、595 个预测的
封存和独立 GT64 评分；任一科学验收失败时保留结果并停止，不产生正式测试分数。

FNO 使用 `train_multicase_workers.py` 完成相同的回投影工程提速。其余固定预算试验由
`remaining_pilots.py` 排队，GNOT 使用 `train_gnot_chunked.py` 保留完整全局算子并增大查询分块。
`train_deeponet_stabilization.py` 是独立登记、固定一次的低学习率对照，不改写原 DeepONet 失败。

所有训练入口都保护既有目录及 journal，不能用重新启动 fresh 入口代替中断恢复。
epoch 边界、optimizer step、scheduler 和 RNG 的证据见 `workers_boundary.json`、
`fno_workers_boundary.json` 及不可变的 `workers_boundary.pt`。

独立复核由 `completion_observer.py` 调用 `review_multicase.py` /
`review_deeponet_stabilization.py`，复算完整验证集及测量干预。
正式 U-Net 另由 `review_validation.py` 核验完整训练预算和 NumPy 物理投影。

异常伪目标的 CPU 成对诊断入口是 `repair_teacher_linesearch_v2.py`。
旧 `repair_teacher_linesearch.py` 及初次登记保留为零更新预检查失败的原始证据；
v2 只修正了“ABNORMAL 且只接受一次迭代”的案例选择，并加入同设备 control。
该诊断没有通过目标质量门槛，故没有生成替代训练目标，也没有改变运行中的训练。

最终解释与图表见 `docs/reports/classical_feasible_v1_training_test_zh.md`。
`current_comparison.csv` 的验证 H 是投影误差；`unet_test_summary.csv` 才是封存后的
三维 GT64 测试指标，两者不能混为同一种误差。
