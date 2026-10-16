# 方案 B：FP32 运行优化 v2

在物理 GPU 1（RTX 4090，24 GiB）实测，保留原网络、FP32 精度、全部有效射线、Q96 连续积分、64³ 双分支一致性损失及有效 batch=2，完整 166 个训练案例、36 个验证案例的一轮耗时从 **160.83 秒降至 44.60 秒**，整体加速 **3.61 倍**。按首轮训练和验证计时外推，单种子 300 轮由 **13.40 小时降至 3.72 小时**；实际安排建议预留约 4 小时，最终取决于后续负载与存盘开销。

## 改动与耗时来源

本版不修改网络层数、宽度、位置编码、初始化、优化器、学习率或科学损失。只修改三个运行参数：

| 参数 | 原版 | v2 |
|---|---:|---:|
| `checkpoint_query` | true | false |
| `ray_chunk` | 256 | 2048 |
| `query_chunk` | 32768 | 196608 |

原版为限制显存使用，对完整查询/MLP/积分块做梯度检查点，反向时重算，并把查询拆成许多小块。GPU profiling 的一个真实案例中，MLP 的 `aten::addmm` 占约 39% 的 self CUDA 时间，三线性插值反向约占 12%；同时有大量 Python 调度与小 kernel 调用。关闭重算和扩大工作块，主要减少了这些重复计算及调度。此处 profiler 时间含采样开销，不作为正式测速结果。

内存代价明显：峰值已分配显存从约 0.44 GiB 升至 **13.18 GiB**，峰值 reserved 为 **13.79 GiB**。本轮包含输入 A、B 各 5 个视角的组合，没有 OOM；部署时需为该运行模式保留约 15 GiB 可用显存。两版 matmul/cuDNN TF32 均关闭，没有改用 BF16 或减少积分采样。

## 完整轮次对照

两次实验均从相同随机种子、完整训练投影尺度初始化出发，训练顺序、A/B/H 采样和验证案例一致。每次完成 **83 次真实 AdamW 更新**，再验证全部 36 例；没有 warmup 更新，没有保存模型权重。计时使用原 `run_epoch`，包含读取、传输、全部六路连续投影、64³ 一致性、反传、梯度裁剪、更新和完整诊断。

| 指标 | 原版 | v2 |
|---|---:|---:|
| 训练 / 验证案例 | 166 / 36 | 166 / 36 |
| 平均每次更新 | 1.78595 秒 | **0.47116 秒** |
| 每轮训练 | 148.23 秒 | **39.11 秒** |
| 每轮验证 | 12.60 秒 | **5.50 秒** |
| 每轮合计 | 160.83 秒 | **44.60 秒** |
| 300 轮外推，单种子 | 13.40 小时 | **3.72 小时** |
| 首轮平均训练 loss | 0.3646793264 | 0.3646793250 |
| 首轮结束验证 loss | 0.2207147599 | 0.2207149565 |
| 首轮结束验证投影 relative L1 | 0.6840089008 | 0.6840096069 |

两版首轮结束的验证投影 relative L1 绝对差约 **7.06e-7**。本表的约 68.4% 是全量数据首轮指标，与之前 16 例训练、8 例验证、30 轮得到的 41.13% 使用不同的训练预算和验证成员，不应直接比较其高低。

此外，在相同的两个真实训练案例和相同的已训练权重上，逐项测试关闭重算及不同 chunk。2048 射线工作块的 loss 与原版一致，全部参数梯度最大绝对差 **1.58e-7**、相对 L2 差 **1.85e-6**。固定案例 warmed 更新约为 0.48 秒，完整轮次测量与该结果一致。

这些检查支持运行优化的数值一致性，不能替代 300 轮收敛与重建准确率评价。300 轮时间是按单个完整首轮的外推，尚未计入逐轮检查点落盘及所有启动/终止诊断。没有强制清空操作系统文件缓存；“cold”指模型几何缓存和内核首次运行，完整训练投影初始化可能预热文件缓存。

## 代码、配置与复现

新入口为 `scripts/train_plan_b_perf.py`，适配器为 `scripts/plan_b_perf_v2.py`。适配器将原训练器加载进独立模块命名空间，替换运行协议及三个运行参数，并将自身和新入口纳入来源哈希。原 A、B 冻结源码文件与 `src/**/*.py` 成员均经前后核验，未改动。B 历史检查点仍使用原源身份。

新协议为 `scheme_b_continuous_measurement_only_perf_v2`，预登记见 `protocols/plan_b_perf_v2/registration.json`。已提供 `scheme_b_perf_v2_smoke.yaml`、`scheme_b_perf_v2_gate.yaml` 以及三个 300 轮正式设计配置。新运行模式保留原训练器的验收规则，不会把旧的 smoke 失败改成通过，也不会将原运行模式的证据直接当成新协议的正式资格。本次完成性能验证，没有启动新的全量 300 轮训练。

后续用户已授权正式启动，记录见[GPU 1 正式训练启动说明](reports/plan_b_perf_v2_formal_launch_zh.md)。后续在保留原严格验收结果的同时，新增预登记的指南 G6/G7 资格记录，重新执行同源 smoke/gate，再启动单种子 300 轮；下文性能数据仍为先前完成的对照实验。

```bash
# 独立训练配置校验
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/train_plan_b_perf.py \
  --config configs/research/scheme_b_perf_v2_gate.yaml --validate-only

# 完整首轮测速；每次使用新的 JSON 路径
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/benchmark_plan_b_epoch.py \
  --variant fast --device cuda:1 --output outputs/research/plan_b_perf_v2/new_full_epoch_fast.json

# 相同配置的原版对照
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/benchmark_plan_b_epoch.py \
  --variant reference --device cuda:1 --output outputs/research/plan_b_perf_v2/new_full_epoch_reference.json
```

CPU 回归及运行隔离/来源测试 **32 passed、1 skipped**；跳过的是显式 opt-in 的原 GPU 检查点测试，本次另有 GPU 1 上完整参数梯度对照及两版真实全量轮次检查。两版全量测速的 GT、禁用 NPY、外部监督检查点访问计数均为 0；每版只读取 166 个训练投影与 36 个验证投影。

原始证据：[原版完整轮次](../outputs/research/plan_b_perf_v2/full_epoch_reference.json)、[优化版完整轮次](../outputs/research/plan_b_perf_v2/full_epoch_fast.json)、[工作块及梯度对照](../outputs/research/plan_b_perf_v2/initial_benchmark.json)、[原版 profile](../outputs/research/plan_b_perf_v2/reference_profile.txt)。
