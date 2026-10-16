# 方案 A 第二轮执行优化：编译、卷积算法与网络精度

本轮按“保持结构的执行优化 → 网络部分降精度”的顺序实测。未缩小先验宽度、未减少 K、未改物理采样数或训练损失，也未启动 300 轮正式训练。旧 v2 源码和配置保持不变。

## 当前可用的 FP32 配置

推荐继续使用 **FP32＋先验模块 `torch.compile`＋cuDNN 自动算法选择**，采用原连续内存布局。它保留 A2 的 K4/Q96、宽度32、lift32/output64、全部梯度与原始初始化。通过独立启动器启用，不修改历史训练入口。

本轮固定真实案例完整 batch2 更新的中位数：普通 FP32 2.527 秒，组合配置 **2.084 秒（1.21 倍，耗时降低 17.6%）**；峰值已分配显存约 **6.69 GiB**，基准约 7.69 GiB。这是共享单张 RTX 4090 的预热后实测；编译和算法搜索的首次成本另列在报告，不用短测速直接推算整轮。前后复测、全部候选和最终完整轮次见[自动实测报告](reports/plan_a_runtime_v3_zh.md)。

实验中途同卡的大任务结束，此后仍有另一个 Python 进程，不能视为独占。同阶段复测普通 FP32 为 **1.513 秒**，组合配置为 **1.156 秒（1.31 倍，耗时降低23.6%）**，严格数值核验再次通过。不能将先前2.527秒直接与后期1.156秒相除作为优化收益；GPU负载采样和每次时间均保留。

同一后期负载下，另用真实 epoch runner 完整执行 **166 train / 36 val**：83 次真实 AdamW 更新的训练阶段 **143.93 秒**，验证 **13.98 秒**，合计 **157.91 秒，即2.63分钟/轮**，峰值已分配显存 **6.75 GiB**。初始几何缓存为空，没有预热更新；首次编译/算法搜索计入阶段时间，但磁盘编译缓存已存在，OS文件缓存也没有清空。逐次梯度均有限，数据审计仅访问202个Pool A train/val投影文件，禁读尝试0。

按本轮线性估算：单种子300轮约 **13.16小时**，三个种子单卡串行约 **39.48小时（1.64天）**，不含正式checkpoint/freeze序列化。它是当前较轻共享负载下的成本估计，不是独占卡或完整300轮实测；先前v2约28小时的估算是在更重的共享负载下得到，下降不能全归因于本轮代码优化。

| 候选 | 完整更新中位数 | 严格数值核验 | 处理 |
| --- | ---: | --- | --- |
| 仅 channels_last_3d | 5.943 秒 | 未通过 | 不采用 |
| 仅 cuDNN 自动选择 | 2.315 秒 | 通过 | 保留证据 |
| 仅编译先验 | 2.427 秒 | 未通过，一个更新后参数超差 | 不单独采用 |
| channels_last_3d＋自动选择 | 2.759 秒 | 通过 | 未带来加速 |
| 编译先验＋自动选择 | 2.084 秒 | 通过 | 当前 FP32 推荐 |
| 上述组合＋TF32 卷积 | 1.064 秒 | 未通过 | 独立降精度候选 |
| 上述组合＋BF16 网络 | 0.897 秒 | 未通过 | 独立降精度候选 |

“未通过”指沿用 v2 的严格 FP32 工程逐元素容差，不等于不能学习。所有失败原值保留，不调整阈值；也不声称当前执行优化保持位级一致或长期训练轨迹完全一致。比较输入、初始化、完整梯度和更新后参数，而不是只比较平均 loss。

## 验收与版本隔离

新 FP32 组合配置已完成真实 smoke：2 train/2 val、20 次更新，最佳 H 相对 L1 **0.982465**；以及 gate16：16 train/8 val、5 轮40次更新，最佳 H 相对 L1 **0.289146**。全部原验收条件通过，GT、其他 NPY、外来检查点禁读尝试均为 0。旧 v2 gate 的 0.289250 与本轮接近，但这些只是小规模投影学习结果，不能证明三维精度等价。

新的配置使用独立 `scheme_a_runtime_fp32_v3` 标识，原训练入口会拒绝。启动器将自身与执行实现的文件哈希纳入来源记录，正式预检也核对执行设置。v3 不能复用 v2 smoke/gate，默认 v2 训练也不能复用 v3 的源码身份。运行目录、检查点与旧记录分开保存。

[三个种子的只读正式预检](reports/artifacts/plan_a_runtime_v3/formal_preflight.json)均通过；它只读配置、投影文件校验与验收记录，不构造模型或执行正式更新。

## 降精度实验的边界

TF32 只允许 cuDNN 卷积使用，矩阵乘法 TF32 仍关闭。BF16 只包围二维编码、三维解码和先验调用，返回值恢复 FP32，再进入几何提升、物理状态更新、投影、解析梯度和损失。原参数与参数梯度仍为 FP32。恢复数据类型不会消除已经产生的舍入误差。

在初始权重的训练体素比较中，TF32 相对 L2 差约 **0.102%**，BF16 约 **2.38%**；完整裁剪后参数梯度差约 **0.063% / 2.09%**。这类差异不能按“小数位不同”直接忽略。相同 16/8 案例的 40 次更新学习探针另行记录，探针不生成正式冻结身份或精度等价结论。保留 FP32 输出头的 BF16 补充候选也单独记录。

两次降精度学习探针均通过有限梯度、测量依赖和留出误差改善等小规模检查：TF32 最佳 H 相对 L1 **0.288472**，BF16 **0.285879**。不能据此判断哪种精度的正式三维效果更好。补充 BF16＋FP32 输出头的体素差约 **2.30%**，未通过严格比较，改善有限，表明差异不只来自末端输出的量化；该补充候选没有继续做学习探针。

因此，本轮可用训练启动器只接受 FP32 组合配置。TF32/BF16 没有进入正式配置，也没有通过改变宽度或步数来弥补速度。

本轮优化覆盖训练执行；已有重建/评价入口可按原 FP32 eager 路径加载相同键名的权重，未自动开启编译，因此不能将本页训练执行的加速直接算成已有推理接口的加速。

相关回归与新增运行边界检查 **76 项通过**（[CPU 74项](reports/artifacts/plan_a_runtime_v3/pytest_core.log)、[CUDA 2项](reports/artifacts/plan_a_runtime_v3/pytest_cuda.log)）。测试包括完整小模型的梯度与状态一致性、BF16 到物理算子的 FP32 边界、原启动器拒绝 v3 配置以及不同执行配置的正式验收拒绝。

## 使用方法

在项目根目录使用 `/scratch/djc/envs/gas-torch/bin/python`。编译线程设置建议如下，避免并发占用过多 CPU：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
```

已完成 smoke/gate 目录不可覆盖；三个正式配置已准备好，但本轮未执行：

- [seed20261011](../configs/research/scheme_a_a2_formal_runtime_v3_seed20261011.yaml)
- [seed20261012](../configs/research/scheme_a_a2_formal_runtime_v3_seed20261012.yaml)
- [seed20261013](../configs/research/scheme_a_a2_formal_runtime_v3_seed20261013.yaml)

只读预检命令（不会开始训练）：

```bash
python scripts/train_plan_a_runtime.py \
  --config configs/research/scheme_a_a2_formal_runtime_v3_seed20261011.yaml \
  --preflight-only
```

复测当前 FP32 组合配置（使用全新输出文件名）：

```bash
python scripts/benchmark_plan_a_speed.py \
  --config configs/research/scheme_a_a2_formal_perf_v2_seed20261011.yaml \
  --device cuda:0 --microbatch 1 --ray-chunk 1024 --checkpoint-prior off \
  --experimental-runtime --compile-prior --cudnn-autotune \
  --output outputs/research/plan_a_runtime_v3/reproduction.json
```

学习探针入口为 [probe_plan_a_runtime.py](../scripts/probe_plan_a_runtime.py)，只允许 2/2 案例20更新或16/8案例40更新。源码与实验边界见[实验登记](../protocols/plan_a_runtime_v3/protocol.json)及[执行实现冻结清单](../protocols/plan_a_runtime_v3/execution_frozen.json)。
