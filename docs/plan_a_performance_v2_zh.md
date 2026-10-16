# 方案 A：性能优化 v2 与实测

> 后续执行优化、卷积算法选择及 TF32/BF16 实验已完成，见[第二轮优化说明](plan_a_runtime_v3_zh.md)。本页保留 v2 原始测量及当时的后续建议，v3 通过独立启动器使用。

在保持 A2 结构、481,014 个参数、K=4、每射线 Q=96、32³ 初始化及 64³ 输出的条件下，当前实现已明显缩短训练时间。采用 FP32、禁用 TF32，保留完整高阶梯度与原 A/B/H 损失。本轮只做工程优化、测速和小规模验收，未启动 300 轮正式训练。

## 实测结果

测试设备为一张 **RTX 4090（cuda:0）**，同卡存在其他任务。未测独占卡，以下数值不能视为独占卡承诺。原始结果、环境、负载采样与数值比较见[自动测速报告](reports/plan_a_perf_v2_zh.md)。

| 测量项目 | 优化前 | 当前选定配置 | 比值 |
| --- | ---: | ---: | ---: |
| 固定真实案例，每次有效 batch=2 的完整训练更新，中位数 | 16.23 秒 | 2.82 秒 | 5.75 倍 |
| 固定两个真实验证案例，中位数 | 12.62 秒 | 0.75 秒 | 16.86 倍 |
| 上述训练峰值已分配显存 | 3.67 GiB | 7.69 GiB | 以显存换速度 |

优化前源码在稍后复测为 25.12 秒/更新、15.05 秒/两个验证案例，说明共享负载波动明显。因此不将两次基线中较慢的一次作为主要加速倍数。

另用真实训练 runner 完整执行 **166 个训练案例、83 次 AdamW 更新、36 个验证案例**，包含实际几何变化、数据加载和完整诊断指标汇总。训练 304.99 秒，验证 29.94 秒，合计 **334.93 秒，约 5.58 分钟/轮**；训练峰值已分配显存 7.75 GiB。初始几何缓存为空、没有预热更新；OS 文件缓存没有清空，测速前的哈希校验可能预热文件读取。

按这一次完整轮次线性外推：**单个种子 300 轮约 27.91 小时；三个种子在同一张卡串行约 83.73 小时（3.49 天）**。这一估算包含训练与验证 runner，未计正式训练的检查点保存与冻结序列化，也不代表后期权重、后续采样或未来负载下的确定速度。完整一轮测速不保存模型，不能作为收敛或三维精度结论。

## 实现与配置

1. 为固定几何建立默认 512 MiB 的 CUDA LRU，按实际几何、网格、物理算子参数、设备和精度区分缓存；整站传输一次后在 GPU 分块，替代逐块重复拷贝。CPU LRU 同样有界。`device_cache_budget_bytes=0` 可回退原流式传输；LRU 上限约束缓存持有量，自动微分图可能继续持有已淘汰张量。
2. 内外层复用同一算子，A/B/H 几何各准备一次并在双分支投影间复用。
3. 合并归一化投影与精确 `J(C)^T(p(C)-y)`，共享插值，保留残差与浓度的全部求导路径。A-control 继续使用其原残差反馈。
4. 将诊断标量批量传回 CPU，减少逐项同步。
5. 选定 `ray_chunk=1024`、`batch_size=1`、`gradient_accumulation=2`、`checkpoint_prior=false`。关闭先验重计算降低计算量，但提高显存需求。

实现改动的冻结差异见 [source_changes.patch](reports/artifacts/plan_a_perf_v2/source_changes.patch)，最终源码清单见 [source_frozen.json](../protocols/plan_a_perf_v2/source_frozen.json)。旧版源码副本保存在 `outputs/research/plan_a_perf_v2/baseline`，旧实验目录与报告保留。

## 后续瓶颈与优化方向

对已保存的 `selected_speed.trace.json` 做只读分析，通过 `External id` 将 GPU kernel 关联到 CPU 算子，再按记录的卷积输入与权重形状归属模块，得到[卷积模块耗时分解](reports/artifacts/plan_a_perf_v2/network_bottleneck_breakdown.json)。这些比例的分母是本次 trace 的 GPU kernel 时长总和（2.181 秒），不含 memcpy/memset，也不是整次更新的墙钟占比。共享负载与 profiler 本身也会影响耗时。

| 算子归属 | GPU kernel 时间占比 |
| --- | ---: |
| 64³ 共享三维先验卷积，前向＋反向 | 60.7%（前向 10.5%，反向 50.2%） |
| 初始化器三维解码卷积，前向＋反向 | 23.0% |
| 全网 GroupNorm，前向＋反向 | 8.5% |
| 初始化器二维编码卷积，前向＋反向 | 0.07% |
| 其他 kernel，包括物理计算、索引、激活与优化器 | 7.7% |

GroupNorm 和激活没有进一步拆回模块，因此第一行不是整个先验模块的全部耗时。一个有效 batch=2 更新包含 2 案例×2 分支×4 步，共 16 次先验调用；每次有 8 个卷积，其中 6 个为 32→32 的 3×3×3 卷积。共享权重节省参数，并不省掉这些重复计算。初始化器升采样到 64³ 后的三维卷积同样需要关注。

后续建议依次进行独立候选实验，而不直接改变已验收配置：

- 保持结构与 FP32，测试卷积内存布局（`channels_last_3d`）、cuDNN 算法选择和仅对先验模块做 `torch.compile`。形状变化、布局转换及编译成本可能抵消收益，需要测完整更新，不能仅看单个卷积。
- 对网络部分测试 TF32 或 BF16，测量归一化、物理积分、解析梯度和外层损失继续 FP32。须比较完整梯度、更新后参数及小规模学习结果；物理输入恢复 FP32 不能消除网络先前引入的舍入误差。当前正式配置仍保持原精度。
- 若工程优化收益不足，再研究宽度 32→16 或 K=4→2。前者使主要同宽卷积的理论乘加量约变为 1/4，后者使重复先验调用减半，均不代表整网同比加速，且改变模型容量或算法步数，必须作为独立消融评价精度。
- 将先验放到 32³ 再上采样修正，能减少体素计算量，但改变空间表达与细节恢复能力，优先级低于上述候选。

本节仅定位瓶颈并列出候选；未实施或测量这些新增候选，不据此承诺新的加速倍数。当前 v2 源码和验收身份保持不变。

## 正确性与验收证据

[相关回归测试](reports/artifacts/plan_a_perf_v2/pytest_final.log) **72 项通过**。新增测试覆盖融合算子的输出、梯度、二阶梯度、目标梯度、无梯度执行、无效射线以及 CUDA 缓存命中、淘汰和几何变更。[算子数值检查](reports/artifacts/plan_a_perf_v2/operator_numerics.json)在 CPU FP64 与 CUDA FP32 下通过。

最终选定配置与旧源码采用相同真实输入、相同初始化权重和 AdamW 状态比较：训练体素相对 L2 差约 1.12e−7、全参数裁剪后梯度约 1.15e−7、更新后参数约 2.33e−8，全部满足预先固定阈值。它们不是位级一致，也不能保证长期训练轨迹完全相同。

所有候选结果均保留。两个未选用的候选各有一个更新后参数超过固定逐元素容差；对应梯度差非常小，AdamW 的 `eps=1e−8` 会放大小梯度的差异。未放宽阈值，也不以平均误差掩盖失败项。逐项误差和候选状态见自动报告。

完整轮次审计读取 202 个 Pool A train/val 投影文件，GT64/GT128、其他 NPY、外来检查点禁读尝试均为 0。新版本的真实 smoke20 与 gate16 均已通过，来源校验与检查点哈希见自动报告：

| 验收 | 训练 / 验证案例 | 轮数 / 更新数 | 初始 → 最佳留出投影相对 L1 | 状态 |
| --- | ---: | ---: | ---: | --- |
| smoke | 2 / 2 | 20 / 20 | 103.44791 → 0.982463 | 通过 |
| gate16 | 16 / 8 | 5 / 40 | 71.11865 → 0.289250 | 通过 |

两次验收的测量依赖性、有限梯度、输出变化与全部其他条件均通过，禁读尝试为 0。历史 v1 通过记录不能替代当前源码的验收。小规模验收不等于正式三维效果验证。

## 复现入口

在仓库根目录使用 `/scratch/djc/envs/gas-torch/bin/python`，推荐设置 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。测速脚本会实际执行优化器更新，但不会保存正式模型。

```bash
python scripts/benchmark_plan_a_speed.py \
  --config configs/research/scheme_a_a2_formal_perf_v2_seed20261011.yaml \
  --device cuda:0 --microbatch 1 --ray-chunk 1024 --checkpoint-prior off \
  --output outputs/research/plan_a_perf_v2/reproduction_speed.json

python scripts/benchmark_plan_a_epoch.py \
  --config configs/research/scheme_a_a2_formal_perf_v2_seed20261011.yaml \
  --device cuda:0 --microbatch 1 --ray-chunk 1024 --checkpoint-prior off \
  --output outputs/research/plan_a_perf_v2/reproduction_epoch.json
```

小规模训练配置为 [smoke](../configs/research/scheme_a_smoke_perf_v2.yaml) 和 [gate16](../configs/research/scheme_a_gate_perf_v2.yaml)。已完成目录不覆盖，重跑须通过训练入口的 `--run-name` 指定新名称。

保留的三个正式配置分别为 [seed20261011](../configs/research/scheme_a_a2_formal_perf_v2_seed20261011.yaml)、[seed20261012](../configs/research/scheme_a_a2_formal_perf_v2_seed20261012.yaml)、[seed20261013](../configs/research/scheme_a_a2_formal_perf_v2_seed20261013.yaml)，均为 300 轮，绑定新版本 smoke/gate 证据。[只读正式配置预检](reports/artifacts/plan_a_perf_v2/formal_preflight.json)均通过，核对源码、数据及验收来源，未构造模型或执行更新。本轮未执行这些正式任务。
