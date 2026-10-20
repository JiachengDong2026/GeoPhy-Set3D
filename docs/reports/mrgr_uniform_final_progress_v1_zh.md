# MRGR-UniformFusion v1 执行进展（非最终结果）

更新时间：2026-10-20T19:33:14.770360+08:00（北京时间）；调度状态：`stage_complete_awaiting_user_decision`。

任务尚未全部完成时，不将本文当作正式评价报告。原有35例测试报告和已验收结果保留。本轮数据审计未发现额外独立案例；后续35例评价属于补充结果。

固定主模型：MRGR-UniformFusion-v1，342725参数，seed20261011第285轮，权重SHA256 `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。两历史均匀融合种子核验通过；新主线种子20261020/20261021，与独立支线反馈OARI匹配。

## 工程与统计检查

- cpu：通过
- gpu：通过
- statistics_synthetic：通过

## 训练队列

| 任务 | 轮次 | 实际更新 | 最佳H4 L1 | GPU UUID | 状态 |
|---|---:|---:|---:|---|---|
| uniform_seed20261020 | 300/300 | 24900 | 3.4286% |  | complete |
| uniform_feedback_oari_seed20261020 | 300/300 | 24900 | 3.2548% |  | complete |
| uniform_seed20261021 | 300/300 | 24900 | 3.5660% |  | complete |
| uniform_feedback_oari_seed20261021 | 300/300 | 24900 | 3.4653% |  | complete |

## 后续消融暂缓

按最新用户要求，先完成当前四个实验的300轮训练及35例三维配对评价，再讨论均匀融合＋反馈OARI是否作为最终模型。以下7项保持登记但暂不启动，评价完成后也不会自动恢复；此前取消的3个第二种子仍取消。

- `uniform_no_feedback_seed20261011`
- `uniform_no_feedback_seed20261019`
- `uniform_zero_residual_seed20261011`
- `uniform_zero_residual_seed20261019`
- `uniform_no_ray_relations_seed20261011`
- `uniform_no_consistency_seed20261011`
- `uniform_single_scale_seed20261011`

本阶段只有四项纳入训练完成门槛；各自H4选轮和预测封存后再评分。原模型、训练预算和统计校正分母不变。反馈OARI不自动晋级，阶段完成不代表所有消融或最终模型决策完成。

每正式训练300轮/24900更新，FP32、原始预算不变；OARI先20轮验证H4工程筛查，门槛在运行前登记。训练期禁止体积GT与测试投影；全部纳入任务训练完成后才生成和封存新增预测。

[执行协议](../mrgr_uniform_final_v1_protocol_zh.md)；代码入口：`scripts/uniform_final_v1/controller.py`；运行目录：`/scratch/djc/GeoPhy-Set3D/mrgr_uniform_final_v1`。

完整GPU进程、UUID锁、资源快照、异常与恢复记录保存在运行目录`scheduling/`。完成评分和独立验收后自动生成正式报告并尝试按限定文件清单同步GitHub；任何失败都会阻断后续阶段并保留日志。

正式评价及独立验收已完成，结果以[最终总报告](mrgr_uniform_final_v1_zh.md)为准；GitHub同步状态另见运行目录`github_sync.json`。
