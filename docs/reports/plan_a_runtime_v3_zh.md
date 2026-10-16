# 方案 A 第二轮执行优化实验

保持 A2、K4/Q96、宽度32、64³ 输出和完整梯度。网络执行候选独立于 v2 正式配置；物理计算与损失保持 FP32。所有时间为共享单张 RTX 4090，固定案例预热后的完整 batch2 更新。

| 候选 | 比较基准 | 更新中位数 秒 | 两案例验证 秒 | 更新加速比 | 峰值显存 GiB | 严格数值核验 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| autotune | baseline | 2.315 | 0.762 | 1.09× | 7.69 | 通过 |
| baseline | baseline | 2.527 | 0.678 | 1.00× | 7.69 | 通过 |
| baseline_repeat | baseline_repeat | 1.513 | 0.385 | 1.00× | 7.69 | 通过 |
| bf16 | baseline | 0.897 | 0.442 | 2.82× | 4.73 | 未通过 |
| bf16_fp32_heads | baseline_repeat | 0.536 | 0.186 | 2.82× | 4.89 | 未通过 |
| compile | baseline | 2.427 | 0.594 | 1.04× | 6.70 | 未通过 |
| compile_autotune | baseline | 2.084 | 0.540 | 1.21× | 6.69 | 通过 |
| compile_autotune_repeat | baseline_repeat | 1.156 | 0.278 | 1.31× | 6.69 | 通过 |
| layout | baseline | 5.943 | 1.188 | 0.43× | 7.74 | 未通过 |
| layout_autotune | baseline | 2.759 | 1.085 | 0.92× | 7.74 | 通过 |
| tf32 | baseline | 1.064 | 0.385 | 2.38× | 6.72 | 未通过 |

逐元素容差沿用 v2：体素/H预测 rtol=1e−4、atol=1e−6；完整clip后梯度 rtol=1e−3、atol=1e−6；更新后参数 rtol=1e−4、atol=1e−6。初始化与输入必须相同。不因 TF32/BF16 结果放宽阈值。未通过不等于不能学习，也不允许自动替代当前已验收 FP32 配置。

编译和算法搜索的首次开销记录在 train_warmup，表中只列预热后时间。不同候选的最小/最大值、完整数值比较和禁读审计见配套 JSON。短测速不能直接外推 300 轮。

本轮中途 GPU 0 的大任务结束，复测时仍有另一个 Python 进程。baseline 与 baseline_repeat 的耗时差包含明显负载变化，不能视为代码加速。repeat 候选应与 baseline_repeat 比较；其他早期候选仅与早期基准比较，不将跨阶段比值解释为算法收益。

## autotune 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.044e-07 | 0 |
| heldout_predictions | 4.335e-08 | 0 |
| clipped_gradients | 1.861e-07 | 0 |
| post_update_state | 3.340e-08 | 0 |

## baseline 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 0.000e+00 | 0 |
| heldout_predictions | 0.000e+00 | 0 |
| clipped_gradients | 0.000e+00 | 0 |
| post_update_state | 0.000e+00 | 0 |

## baseline_repeat 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 0.000e+00 | 0 |
| heldout_predictions | 0.000e+00 | 0 |
| clipped_gradients | 0.000e+00 | 0 |
| post_update_state | 0.000e+00 | 0 |

## bf16 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 2.379e-02 | 915651 |
| heldout_predictions | 9.300e-03 | 17397 |
| clipped_gradients | 2.091e-02 | 270584 |
| post_update_state | 5.372e-04 | 2252 |

## bf16_fp32_heads 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 2.304e-02 | 913057 |
| heldout_predictions | 8.807e-03 | 17388 |
| clipped_gradients | 2.130e-02 | 270571 |
| post_update_state | 5.364e-04 | 2230 |

## compile 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.043e-07 | 0 |
| heldout_predictions | 4.238e-08 | 0 |
| clipped_gradients | 1.203e-07 | 0 |
| post_update_state | 4.499e-08 | 1 |

## compile_autotune 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.043e-07 | 0 |
| heldout_predictions | 4.265e-08 | 0 |
| clipped_gradients | 1.673e-07 | 0 |
| post_update_state | 2.273e-08 | 0 |

## compile_autotune_repeat 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.049e-07 | 0 |
| heldout_predictions | 4.225e-08 | 0 |
| clipped_gradients | 1.761e-07 | 0 |
| post_update_state | 1.165e-08 | 0 |

## layout 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.046e-07 | 0 |
| heldout_predictions | 4.304e-08 | 0 |
| clipped_gradients | 1.137e-07 | 0 |
| post_update_state | 5.567e-08 | 1 |

## layout_autotune 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.049e-07 | 0 |
| heldout_predictions | 4.266e-08 | 0 |
| clipped_gradients | 1.724e-07 | 0 |
| post_update_state | 5.384e-08 | 0 |

## tf32 数值检查

| 张量 | 相对 L2 | 超差元素数 |
| --- | ---: | ---: |
| volumes | 1.019e-03 | 765586 |
| heldout_predictions | 4.165e-04 | 12289 |
| clipped_gradients | 6.264e-04 | 17305 |
| post_update_state | 1.012e-04 | 241 |

## 独立 FP32 编译/自动算法配置验收

此配置通过独立启动器运行，来源 hash 包含启动器及执行配置实现；与 v2 验收身份不同。

| 验收 | 完成/通过 | 最佳留出投影相对 L1 | 当前源码/执行代码匹配 |
| --- | --- | ---: | --- |
| smoke | True / True | 0.982465 | True |
| gate | True / True | 0.289146 | True |

## 降精度学习探针

使用实际 epoch runner、相同初始化种子与 A/B/H 采样，固定 16 个训练/8 个验证案例，5 轮40次更新。无正式 checkpoint、无正式验收身份，不能据此认定三维精度等价。严格逐元素比较失败仍保留。

| 探针 | 学习检查 | 初始 / 最佳 H 相对 L1 | 更新数 | 禁读尝试 |
| --- | --- | ---: | ---: | ---: |
| probe_tf32 | True | 71.105496 / 0.288472 | 40 | 0 |
| probe_bf16 | True | 71.377363 / 0.285879 | 40 | 0 |

## FP32 组合配置完整一轮

真实166案例、83次更新：训练 143.93 秒，36案例验证 13.98 秒，合计 2.63 分钟/轮。按单轮线性估算，300轮约 13.16 小时，三种子单卡串行约 39.48 小时。

这是共享负载下的一轮测量，包含首次编译/算法搜索（磁盘编译缓存可能已存在），OS文件缓存未清空；外推不是独占卡结果、收敛承诺或完整正式训练，未计正式checkpoint/freeze序列化。
