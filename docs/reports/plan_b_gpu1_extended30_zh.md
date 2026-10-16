# 方案 B：GPU 1 延长学习验证

用户授权在 GPU 1 开展方案 B 的训练与测试后，先运行一个预登记的延长学习实验：16 个梯度更新案例、8 个验证案例、30 轮、240 次 AdamW 更新，最长 1800 秒。该实验用于判断首版 B 的 40 次更新验收失败是否与训练不足有关，不属于全量 300 轮正式训练。

## 实验边界

- 物理 GPU 1：RTX 4090，UUID `GPU-cd80a78f-7d90-c291-4680-4895db9aac8b`，训练设备 `cuda:1`。
- 新实验从头初始化，沿用全体 166 个训练案例的测量尺度初始化、种子 20261011、网络结构、损失、Q96 积分与 FP32；梯度更新仍只用固定 16 案例。
- 相比旧的 5 轮 gate，只修改轮数、对应的 cosine T_max、更新预算及设备。学习率轨迹因此不同，两次实验并非同一训练轨迹的断点续训。
- 仅使用 Pool A 训练/验证投影；不读取体场 GT、最终测试集、Pool B 或外部监督权重。
- 原 smoke/gate 的失败结果保留，验收阈值不变。正式训练仍要求同源 smoke 和 gate 均通过；本实验即使通过，也不会把旧 smoke 的失败改成通过。
- A 的冻结源码和 `src/**/*.py` 成员均保持不变，A 在 GPU 0 的训练继续运行。

预登记：[gpu1_extended30_registration.json](../../protocols/plan_b_v1/gpu1_extended30_registration.json)。配置：[scheme_b_gate_gpu1_extended30.yaml](../../configs/research/scheme_b_gate_gpu1_extended30.yaml)。

## 运行与复现

CPU 测试：30 passed、1 skipped；物理 GPU 1 opt-in 梯度测试：1 passed。新配置校验通过，启动前 A、B 源码哈希与各自冻结清单一致。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/train_plan_b.py \
  --config configs/research/scheme_b_gate_gpu1_extended30.yaml
```

训练由独立会话中的 supervisor 管理，逐轮保存 best/last 检查点。启动目录为 `outputs/research/launches/scheme_b_gate_gpu1_extended30_20261015T162356Z/`；原始结果目录为 `outputs/research/scheme_b_gate_gpu1_extended30/`。运行状态以启动目录的 `launch.json` 和结果目录的 `acceptance_result.json` 为准。

训练结束后的测量数值检查与外部测量推理仍属于工程验证，不产生正式体场准确率结论。

## 已完成结果

2026-10-16 北京时间 00:23:56 启动，00:32:17 完成，训练任务墙钟约 **8 分 20 秒**。30 轮、240 次更新完整结束，扩展 gate 的全部既定条件通过。

| 指标 | 结果 |
|---|---:|
| 初始验证留出投影 relative L1 | 2.200097（220.01%） |
| 最佳验证留出投影 relative L1 | **0.411339（41.13%），第 27 轮** |
| 最后一轮验证留出投影 relative L1 | 0.413341（41.33%） |
| 首轮 → 最后一轮训练 loss | 0.781655 → 0.103880 |
| 初始 → 最后一轮验证 loss | 0.521769 → 0.134456 |
| 平均每次更新，含两个案例 | **1.715 秒** |
| 训练计时合计 / 验证计时合计 | 411.56 秒 / 80.07 秒 |
| 峰值 PyTorch 已分配显存 | 0.441 GiB |

更新计时包含数据读取、双分支编码、连续投影、一致性损失、反传及 AdamW，不包含逐轮验证和存盘；墙钟另包含初始化、来源核验及诊断。该速度是本次 GPU 1 的运行测量，不能将与旧共享 GPU 2 耗时的差异直接归因于代码加速。

延长训练让 B 从近似均匀预测进入可学习的空间变化；第 8 轮首次优于零预测，后期改善放缓并有波动。旧 40 更新实验失败不能代表 B 在更长训练后仍无法通过小规模 gate，但本实验也不足以证明全量训练后的精度。验证只有 8 例，不能直接与 A 或历史模型的全 36 例验证指标比较。

训练和选模的 GT/禁用 NPY/外部监督检查点访问次数均为 0。初始化读取全部 166 个训练投影文件，另有 8 个验证文件，共 174 个不同的允许投影文件。最佳检查点 SHA256 为 `453531e4dd7af33e84203fb836332d4d40ca476b7cb30d97c9123c0acc8db2df`。

## 冻结后测试

在最佳检查点上固定一次 encode 的 context，对同一连续场比较积分精度：Q96 相对 Q384 的投影 relative L1 为 **3.5247e-4（0.03525%）**，Q192 相对 Q384 为 **6.9413e-5（0.00694%）**。导出 C64 后用独立 Q384 体素算子投影，与直接连续积分的差异为 **0.007454（0.7454%）**。本次只检查一个工程场，不构成所有案例与几何上的积分误差保证。

物理 GPU 1 上，外部 measurement-only NPZ 的 64³、128³ 重建均成功，场与重投影均有限，质量分数在 [0,1] 内。固定同一 context，导出 64³、128³ 前后的直接重投影逐位相同，确认导出网格不进入连续重投影路径。不同 CUDA 进程的两次导出有极小的舍入差异：75 个投影值不逐位相等，最大绝对差 0.0009765625 ppm·m、relative L1 为 2.4224e-10；最初临时添加的逐位相等断言因此失败，随后单独记录了固定 context 对照和数值误差，没有修改源码或 gate 的验收阈值。128³ 仍只是同一个连续函数的更密采样。

所有 A/B 冻结源码哈希及 `src/**/*.py` 成员在训练前后再次核对通过。

结果链接：[Loss 曲线](../../outputs/research/scheme_b_gate_gpu1_extended30/plots/loss_curve.png)、[CSV](../../outputs/research/scheme_b_gate_gpu1_extended30/plots/loss_curve.csv)、[验收结果](../../outputs/research/scheme_b_gate_gpu1_extended30/acceptance_result.json)、[积分数值检查](../../outputs/research/scheme_b_gate_gpu1_extended30/diagnostics/q_convergence.json)、[导出一致性](../../outputs/research/scheme_b_gate_gpu1_extended30/diagnostics/export_resolution_consistency.json)。

**本轮完成的是 30 轮小规模训练，尚未启动全量 300 轮正式任务。** 此次扩展 gate 已通过，但原 20 更新 smoke 仍未通过既定的“优于零预测”严格条件；按当前同源 smoke 与 gate 均需通过的规则，正式训练尚未晋级。旧失败结果与三个正式配置全部保留。
