# 方案 A 实现优化与速度实测

本报告比较相同真实 Pool A 测量、相同初始权重下的完整 A/B/H 训练更新与验证计算。所有计算保持 FP32，并关闭 TF32。

**设备为共享 RTX 4090，未取得独占卡吞吐结果。** 同期 `nvidia-smi` 负载与进程已保存在原始 JSON 中；不同时间的共享负载会影响速度比。

## 测速协议与适用范围

- A2、K=4、Q=96、lift32、64³ 输出、prior width32；有效 batch=2。
- 每个训练计时包含完整双分支物理展开、A/B/H 外部损失、反向、实际案例梯度平均、clip 与 AdamW 更新。
- CPU collation 与传输包含在计时内，磁盘读取、模型/优化器重置、数值证据拷贝排除。
- `benchmark_plan_a_speed.py` 单独创建外部投影 operator；优化后的真实 runner 复用 `model.operator`。二者的几何缓存归属及命中情况有差异。
- 快测速包含 objective 所请求的逐步 diagnostics，但不包含真实 runner 的指标汇总、标量记录、快照及日志工作；实际 166 案例 epoch 验收需补齐这些开销。
- 至少一次预热与至少两次正式计时；报告中位数及观测最小/最大值。
- 每次从相同初始权重、新 AdamW 开始；包含首次更新的 moment 初始化。几何缓存与 CUDA kernels 在预热后保持热状态。
- 固定两个训练案例和两个验证案例，仅表示该输入下的工程吞吐；不能直接等同 166 案例 epoch、36 案例验证或后期权重下的训练耗时。
- 本测速脚本自身不生成验收；真实 smoke/gate 复验另列。未开展 300 轮长训练或 GT 三维精度评价。

## 实测速度

| 配置 | 训练更新中位数（最小–最大）秒 | 两案例验证中位数（最小–最大）秒 | 训练/验证中位加速 | 峰值训练显存 GiB |
|---|---:|---:|---:|---:|
| 基线 | 16.227（12.370–17.128） | 12.616（11.068–14.450） | 1.00× / 1.00× | 3.67 |
| optimized_speed | 3.577（3.538–3.658） | 0.857（0.854–0.882） | 4.54× / 14.73× | 3.06 |
| optimized_chunk1024 | 3.250（3.207–3.312） | 0.825（0.804–0.832） | 4.99× / 15.30× | 3.05 |
| optimized_chunk1024_no_checkpoint | 2.732（2.552–2.838） | 0.817（0.810–0.829） | 5.94× / 15.44× | 7.69 |
| optimized_chunk1024_batch2 | 2.725（2.707–2.754） | 0.704（0.615–0.709） | 5.95× / 17.93× | 5.42 |
| selected_speed | 2.822（2.300–2.939） | 0.748（0.732–0.815） | 5.75× / 16.86× | 7.69 |

最小/最大极值速度比范围保存在 summary.json 中；它们不是置信区间，也不是配对速度统计。

## FP32 数值一致性

输入张量 hash、实际数据文件 hash 和初始参数必须完全一致；初始参数另做逐元素精确比较。容差在比较工具中固定：输出与 H 预测 `rtol=1e-4, atol=1e-6`，完整 clip 后梯度 `rtol=1e-3, atol=1e-6`，更新后参数 `rtol=1e-4, atol=1e-6`。逐元素条件为 `|candidate-reference| ≤ atol + rtol·|reference|`。

以下 relative L2 使用每组全部张量元素，未抽样；完整梯度包含全部有梯度参数。它们衡量实现一致性，不能证明长训练收敛或科学精度等价。

| 候选 | 阶段 / 张量 | relative L2 | 最大绝对差 | 超差元素 / 总元素 | 检查 |
|---|---|---:|---:|---:|---|
| optimized_speed | train / volumes | 1.072e-07 | 2.384e-07 | 0 / 1048576 | 通过 |
| optimized_speed | train / heldout_predictions | 4.279e-08 | 6.250e-02 | 0 / 32768 | 通过 |
| optimized_speed | train / clipped_gradients | 1.231e-07 | 5.588e-09 | 0 / 481014 | 通过 |
| optimized_speed | train / post_update_state | 4.605e-08 | 1.345e-06 | 1 / 481014 | 未通过 |
| optimized_speed | validation / volumes | 1.051e-07 | 3.874e-07 | 0 / 1048576 | 通过 |
| optimized_speed | validation / heldout_predictions | 4.239e-08 | 6.250e-02 | 0 / 65536 | 通过 |

optimized_speed：输入与数据 hash 一致；初始参数逐元素一致；整体数值检查未通过。

| optimized_chunk1024 | train / volumes | 1.123e-07 | 9.388e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024 | train / heldout_predictions | 4.345e-08 | 4.688e-02 | 0 / 32768 | 通过 |
| optimized_chunk1024 | train / clipped_gradients | 1.114e-07 | 5.588e-09 | 0 / 481014 | 通过 |
| optimized_chunk1024 | train / post_update_state | 3.353e-08 | 9.295e-07 | 0 / 481014 | 通过 |
| optimized_chunk1024 | validation / volumes | 1.091e-07 | 4.172e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024 | validation / heldout_predictions | 4.264e-08 | 6.250e-02 | 0 / 65536 | 通过 |

optimized_chunk1024：输入与数据 hash 一致；初始参数逐元素一致；整体数值检查通过。

| optimized_chunk1024_no_checkpoint | train / volumes | 1.124e-07 | 2.086e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024_no_checkpoint | train / heldout_predictions | 4.361e-08 | 6.250e-02 | 0 / 32768 | 通过 |
| optimized_chunk1024_no_checkpoint | train / clipped_gradients | 1.100e-07 | 5.588e-09 | 0 / 481014 | 通过 |
| optimized_chunk1024_no_checkpoint | train / post_update_state | 4.216e-08 | 1.153e-06 | 0 / 481014 | 通过 |
| optimized_chunk1024_no_checkpoint | validation / volumes | 1.088e-07 | 3.576e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024_no_checkpoint | validation / heldout_predictions | 4.252e-08 | 6.250e-02 | 0 / 65536 | 通过 |

optimized_chunk1024_no_checkpoint：输入与数据 hash 一致；初始参数逐元素一致；整体数值检查通过。

| optimized_chunk1024_batch2 | train / volumes | 1.123e-07 | 2.384e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024_batch2 | train / heldout_predictions | 4.384e-08 | 4.688e-02 | 0 / 32768 | 通过 |
| optimized_chunk1024_batch2 | train / clipped_gradients | 1.864e-07 | 9.313e-09 | 0 / 481014 | 通过 |
| optimized_chunk1024_batch2 | train / post_update_state | 5.821e-08 | 1.800e-06 | 1 / 481014 | 未通过 |
| optimized_chunk1024_batch2 | validation / volumes | 1.090e-07 | 4.470e-07 | 0 / 1048576 | 通过 |
| optimized_chunk1024_batch2 | validation / heldout_predictions | 4.228e-08 | 6.250e-02 | 0 / 65536 | 通过 |

optimized_chunk1024_batch2：输入与数据 hash 一致；初始参数逐元素一致；整体数值检查未通过。

| selected_speed | train / volumes | 1.123e-07 | 2.235e-07 | 0 / 1048576 | 通过 |
| selected_speed | train / heldout_predictions | 4.472e-08 | 6.250e-02 | 0 / 32768 | 通过 |
| selected_speed | train / clipped_gradients | 1.152e-07 | 4.191e-09 | 0 / 481014 | 通过 |
| selected_speed | train / post_update_state | 2.328e-08 | 5.364e-07 | 0 / 481014 | 通过 |
| selected_speed | validation / volumes | 1.088e-07 | 4.768e-07 | 0 / 1048576 | 通过 |
| selected_speed | validation / heldout_predictions | 4.247e-08 | 6.250e-02 | 0 / 65536 | 通过 |

selected_speed：输入与数据 hash 一致；初始参数逐元素一致；整体数值检查通过。

## 基线复测

基线复测训练中位数 25.117 秒，验证中位数 15.048 秒；原基线/复测中位比为 0.65× / 0.84×，反映同期共享环境波动，不能全部归因于代码。

## 完整真实数据一轮实测

使用真实 runner 完成 166 个训练案例 / 83 次更新，以及全部 36 个验证案例。训练 304.99 秒，验证 29.94 秒，合计 **5.58 分钟/轮**；峰值已分配显存 7.75 GiB。

按本次单轮耗时线性外推：300 轮约 **27.91 小时**，三个种子单卡串行约 **83.73 小时**。这是单次共享负载下的预算估计，尚未实测独占 GPU 或完整 300 轮。

该轮从新权重、新 AdamW 和空模型几何缓存开始；操作系统文件缓存未清空，计时前 hash 读取可能预热文件。计时包含实际数据读取、完整双分支更新和 runner 诊断/指标汇总，外层投影复用 model.operator。模型设置、采样顺序、源代码/data hash、梯度检查和同期 GPU 负载均留存。

外推排除了 setup 和正式 runner checkpoint/freeze 序列化；未来视角采样、缓存淘汰、模型权重及共享负载都会改变耗时，不能以两案例热缓存结果替代该实测。

原始证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/full_epoch_speed.json`。

## 优化后真实 smoke / gate 复验

复验读取既有运行记录与 hash，不反序列化 checkpoint 权重或打开 GT。其通过条件与工程逐元素数值对比条件分别报告。

| 运行 | 完成轮数 / 更新数 | 初始 / 最佳 H 相对 L1 | 验收 | 当前源码匹配 | checkpoint 字节 hash |
|---|---:|---:|---|---|---|
| smoke | 20 / 20 | 103.447910 / 0.982463 | 通过 | 一致 | 一致 |
| gate | 5 / 40 | 71.118650 / 0.289250 | 通过 | 一致 | 一致 |

各次禁读审计、测量依赖性检查、完整验收条目、源码/data/config/checkpoint hash 见 summary.json；原始运行目录如下。

- smoke：`/home/djc/projects/GeoPhy-Set3D/outputs/research/scheme_a_A2_smoke_perf_v2`。
- gate：`/home/djc/projects/GeoPhy-Set3D/outputs/research/scheme_a_A2_gate16_perf_v2`。

## 配置与证据

- 基线：microbatch=1，accum=2，checkpoint_prior=True，ray_chunk=256，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/baseline_speed.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/baseline_speed.numerics.pt`。
- optimized_speed：microbatch=1，accum=2，checkpoint_prior=True，ray_chunk=256，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_speed.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_speed.numerics.pt`。
- optimized_chunk1024：microbatch=1，accum=2，checkpoint_prior=True，ray_chunk=1024，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024.numerics.pt`。
- optimized_chunk1024_no_checkpoint：microbatch=1，accum=2，checkpoint_prior=False，ray_chunk=1024，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024_no_checkpoint.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024_no_checkpoint.numerics.pt`。
- optimized_chunk1024_batch2：microbatch=2，accum=1，checkpoint_prior=True，ray_chunk=1024，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024_batch2.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/optimized_chunk1024_batch2.numerics.pt`。
- selected_speed：microbatch=1，accum=2，checkpoint_prior=False，ray_chunk=1024，diagnostics=on；禁读审计通过。
  原始报告：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/selected_speed.json`；数值证据：`/home/djc/projects/GeoPhy-Set3D/outputs/research/plan_a_perf_v2/selected_speed.numerics.pt`。

完整时间样本、逐张量误差、环境、hash、源文件 hash 和审计见 `artifacts/plan_a_perf_v2/summary.json`。旧 `plan_a_v1` 验收报告保持原样；修改后的源码不能复用旧源码 hash 的正式资格。
