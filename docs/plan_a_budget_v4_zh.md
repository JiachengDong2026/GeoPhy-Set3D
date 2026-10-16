# 方案 A 五小时训练预算版本 v4

当前推荐独立变体：**32³上下文先验＋64³局部分支、BF16网络、FP32物理、1536 MiB几何缓存、4096射线分块**。原A2/v2/v3保留作对照。

在共享单张RTX 4090上，用本版已验收模型连续执行完整166 train / 36 val，每轮83次真实AdamW更新，包含验证、诊断、scheduler和检查点：三轮分别为 **63.57 / 38.16 / 37.67秒**。300轮按三轮均值外推约 **3.87小时/种子**；按首轮＋299个较慢稳定轮次估算约 **3.19小时**。峰值已分配显存约 **3.77 GiB**。共享GPU负载变化会影响实际时间。

K4、Q96、64³物理状态与输出、全量案例、每轮验证、原始A/B/H测量损失及完整物理梯度均保留。先验卷积MAC降到原版约17%，参数略增至488,919；主要收益来自计算网格与执行方式，不能只按参数量判断速度。

本版真实smoke20与gate16均通过，最佳留出投影相对L1分别 **0.847822 / 0.283138**；三个正式种子的只读预检通过。较窄候选smoke失败（1.296774）已排除，完整失败记录保留，验收阈值未变。

BF16与新结构都属于独立变体；小规模投影学习不能证明正式三维精度等价。本轮没有启动300轮正式训练。完整配置、复现命令、筛选记录和限制见[实测报告](reports/plan_a_budget_v4_zh.md)，数值汇总见[summary.json](reports/artifacts/plan_a_budget_v4/summary.json)。

正式配置为`configs/research/scheme_a_budget_v4_formal_seed20261011.yaml`（另有12、13种子），使用`scripts/train_plan_a_budget.py`。原训练入口拒绝本版配置；正式训练核对本版smoke/gate的结构、精度、缓存、模型、源码与数据身份。三个种子单卡串行需上述单种子时间的三倍。

新结构检查点已完成严格重载与measurement-only NPZ推理检查，使用`scripts/reconstruct_plan_a_budget.py`。正式冻结后的验证预测/评分入口为`scripts/evaluate_plan_a_budget.py`，保留原有GT评价边界；本轮未进行GT评价。

## 正式训练启动记录（2026-10-15）

已于2026-10-15 14:10:42 UTC启动单张GPU 0上的300轮正式训练，种子20261011。启动前重新通过本版验收与来源预检。使用上述最终多尺度宽32版本、BF16网络/FP32物理、完整166/36案例，每轮83次更新，共24,900次。

后台任务的实时进程/结束状态见[launch.json](../outputs/research/launches/scheme_a_budget_v4_wide_seed20261011_20261015T141042Z/launch.json)，逐轮日志见[train.log](../outputs/research/launches/scheme_a_budget_v4_wide_seed20261011_20261015T141042Z/train.log)，训练输出在`outputs/research/scheme_a_budget_v4_wide_seed20261011/`。每完整轮次保存`checkpoints/last.pt`，并保存验证测量最佳`best.pt`；发生中断可通过同一配置加`--resume`恢复。

本节为后续正式启动记录，上文的“本轮没有启动300轮”描述此前优化与小规模验收阶段。当前仅启动本种子，其他两个种子保留配置。
