# 方案 B 优化版：GPU 1 正式训练启动记录

用户在完整首轮性能对照之后授权开始训练。本次启动单种子 `20261011`、全量 **166 个训练案例 / 36 个验证案例、300 轮、24,900 次更新**，使用物理 GPU 1，RTX 4090，UUID `GPU-cd80a78f-7d90-c291-4680-4895db9aac8b`。

## 配置与冻结

配置：[scheme_b_perf_v2_formal_seed20261011_gpu1.yaml](../../configs/research/scheme_b_perf_v2_formal_seed20261011_gpu1.yaml)。模型和物理均为 FP32，Q96；关闭查询梯度重算，射线工作块 2048、查询工作块 196608；完整双分支损失、有效 batch=2、AdamW 2e-4、cosine T_max=300、无早停。正式模型重新随机初始化，并使用完整训练投影的固定尺度 bias 校准，不从 gate 或历史模型权重热启动。

正式源码 SHA256：`654b1746945028af5b6951f5e08d25b07f42216543a388e55fbf0ebdad2d2879`，见 [source_frozen.json](../../protocols/plan_b_perf_v2/source_frozen.json)。这包含原 B 模型、原训练器、优化运行适配器、新入口及正式资格政策。原 A、B 的冻结源码均保持原身份。

本次启动预登记见 [formal_launch_registration.json](../../protocols/plan_b_perf_v2/formal_launch_registration.json)。此前 `source_checked.json` 保留的是性能测量阶段的历史适配器身份；本次新增正式资格核对，模型、物理、损失和运行参数没有变化。

## G6 / G7 口径

原指南第 11 节的 G6 要求 2 个案例、20 次更新下无 OOM/NaN、损失下降、输出随输入变化；“优于零场”属于 G7。原训练器曾把该条件也施加到 smoke。本次在新的优化协议中，预登记独立的 [qualification_policy.json](../../protocols/plan_b_perf_v2/qualification_policy.json)，按指南分开核对，保留其余既定稳定性、梯度与禁读检查。

不改写任何 `acceptance_result.json` 或旧失败记录。新的 smoke/gate 各自额外生成 `qualification_result.json`，记录必须通过的条件、诊断项、严格验收原值、政策哈希及原始证据哈希。G6 的零场比较只作诊断；G7 仍必须优于零场。正式入口重新推导资格并检查记录一致、来源/政策/数据/几何相同，存在禁读、未完成或证据变更时拒绝启动。

新版本重新执行 smoke 20 更新和 gate 16 训练 / 8 验证、30 轮、240 更新，然后启动正式训练。CPU 回归与资格检查测试为 **35 passed、1 skipped**，覆盖 G6 保留严格失败、G7 零场门槛、审计及源身份变更拒绝。

## 运行与恢复

独立会话 supervisor：`scripts/supervise_plan_b_training.py`。启动目录为 `outputs/research/launches/scheme_b_perf_v2_formal_seed20261011_20261015T170706Z/`；其中 `launch.json` 记录阶段、PID、配置哈希、开始/结束时间及阶段结果，`smoke.log`、`gate.log`、`formal.log` 分别记录日志。每阶段启动前检查冻结来源及 GPU 1 身份。

正式结果目录：`outputs/research/scheme_b_perf_v2_formal_seed20261011/`。每个完整 epoch 保存 `checkpoints/last.pt`，验证投影 relative L1 改善时保存 `best.pt`；状态包括模型、优化器、调度器、Python/NumPy/Torch/CUDA/loader RNG。会话断开后后台任务继续；若进程失败，supervisor 记录失败，不启动重复训练。

只有原进程已经退出时才使用下列恢复命令，源码、配置和验收证据需保持冻结身份：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/train_plan_b_perf.py \
  --config configs/research/scheme_b_perf_v2_formal_seed20261011_gpu1.yaml --resume
```

完整首轮性能对照给出单种子 300 轮约 3.72 小时，包含训练与验证、不含全部存盘及额外诊断；实际 ETA 以正式任务日志为准。本轮只启动一个种子；另外两个种子保留为正式设计配置。


## 已确认的启动结果

同源 G6 于 2026-10-15 17:07:27 UTC 完成，G7 于 17:10:00 UTC 完成，两者指南资格均通过。G6 原始严格验收仍为 false，唯一未满足项为零场比较，独立资格记录保留该诊断值；G7 原始严格验收与独立资格均为 true，最佳验证投影 relative L1 为 0.4113390551。两项的 GT、禁用 NPY、外部监督检查点访问次数均为 0。

正式阶段于 **2026-10-15 17:10:00 UTC（北京时间 2026-10-16 01:10:00）** 启动，训练 PID `312655`，supervisor PID `310544`。进程最新状态以启动目录的 `launch.json` 和 `formal.log` 为准。
