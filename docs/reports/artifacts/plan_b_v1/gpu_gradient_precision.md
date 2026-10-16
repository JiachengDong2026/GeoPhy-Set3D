# 方案 B GPU checkpoint 梯度精度诊断

设备为物理 GPU 2（UUID `GPU-d3da1ea7-9229-8f8a-5054-854962d4b6b7`），PyTorch 2.11.0+cu128。不使用正在运行方案 A 的 GPU 0。

最初测试使用了 PyTorch 默认的 cuDNN TF32 设置，未匹配训练入口：参考训练同时关闭 `torch.backends.cuda.matmul.allow_tf32` 和 `torch.backends.cudnn.allow_tf32`。仅修改测试 fixture，使两项设置与训练一致，并在退出测试时恢复；未修改模型、物理算子或原有梯度容差。

对每种设置重复三次，比较相同参数和输入的普通模型重复 backward，以及普通模型与 checkpoint 模型的 backward。每次前向输出在两种模型之间逐位相同。下表取全部重复、全部参数张量的最大误差；relative L2 按单个参数张量计算，分母为该张量参考梯度范数。

| matmul TF32 | cuDNN TF32 | 比较 | 最大绝对误差 | 最大 relative L2 | 超过原容差的元素总数 |
|---|---|---|---:|---:|---:|
| 关 | 开（初始测试设置） | 普通模型重复 | 1.261e-5 | 4.562e-6 | 139 |
| 关 | 开 | checkpoint vs 普通 | 1.261e-5 | 4.521e-6 | 38 |
| 开 | 关 | 普通模型重复 | 3.278e-7 | 4.429e-7 | 0 |
| 开 | 关 | checkpoint vs 普通 | 4.172e-7 | 4.575e-7 | 0 |
| 开 | 开 | 普通模型重复 | 9.879e-6 | 2.767e-6 | 146 |
| 开 | 开 | checkpoint vs 普通 | 1.194e-5 | 4.717e-6 | 162 |
| 关 | 关（真实训练设置） | 普通模型重复 | 4.768e-7 | 3.932e-7 | 0 |
| 关 | 关 | checkpoint vs 普通 | 3.576e-7 | 3.949e-7 | 0 |

结果指向 cuDNN TF32 对微小反向舍入差异的放大，而非 checkpoint 独有的梯度错误：未使用 checkpoint 的重复计算也会在 cuDNN TF32 开启时失败。关闭 TF32 后，checkpoint 差异与普通模型自身重复差异处于同一量级。启用 `torch.use_deterministic_algorithms(True)` 后，PyTorch 明确报告 `grid_sampler_3d_backward_cuda does not have a deterministic implementation`。因此可确认链路包含不确定的 CUDA 三维采样反向；这些对照不能把全部舍入误差进一步唯一分配给某一个内核。

保留既有 GPU 逐元素容差 `atol=1e-6, rtol=1e-4`：真实训练设置下，本次三次重复最大绝对误差低于该绝对容差，全部参数通过，无需放宽。该界限是当前设备、版本与测试输入的部署验证值，不能解释为跨所有 CUDA 环境的数学保证。

完整逐参数记录见 `gpu_gradient_diagnostics.json`；诊断脚本及终端输出分别为 `gpu_gradient_diagnostics.py` 和 `gpu_gradient_diagnostics.txt`。修正 fixture 后，GPU 专项结果为 **1 passed, 21 deselected in 4.36s**，见 `gpu_tests.txt`。
