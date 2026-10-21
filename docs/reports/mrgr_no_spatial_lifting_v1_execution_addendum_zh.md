# GeometryLifting 消融：执行与数据边界补充

本页补充[冻结分析生成的正式报告](mrgr_no_spatial_lifting_v1_zh.md)，不改变模型、选轮、指标或统计结果。

训练期间一次会话中断后发现原控制器和两个训练进程均已退出；实际退出信号与退出码不可得。最后完整提交分别为seed20261011第268轮（22244次更新）和seed20261019第260轮（21580次更新）。原始环境、状态、指标、GT访问审计、last/best检查点均在恢复前保存副本及哈希。

独立恢复登记锁定新增调度代码，原115份训练源码和7份分析源码未修改。恢复在原GPU UUID上通过排他锁和双次空闲检查启动，使用原worker恢复模型、最佳模型、optimizer、scheduler及Python/NumPy/Torch/CUDA/loader RNG。未提交的中途epoch按原恢复规则重放；最终每种子恰有连续300个epoch记录、24900次更新，无新增种子或训练因素。本次中断属于执行异常，应随结果披露。

另一次控制器错误发生在首个训练任务完成后：完成状态含有seed字段，原progress()汇总重复传入seed，触发TypeError。第二个训练worker继续完成300轮；两份检查点均已保存。独立v2入口只修复状态合并并衔接原冻结评价流程，未重新训练、改变检查点选择或统计。错误堆栈和新增源码登记均保留。

评分和独立统计完成后，还发生同名report模块导入冲突。独立v3报告入口只复用已完成结果，核验并保留该失败记录，未重跑训练、推理、评分或统计。正式报告与原始验收已披露全部三项异常。[补充最终验收](artifacts/mrgr_no_spatial_lifting_v1/acceptance_recovery_v1.json)额外绑定[独立完整性复查](artifacts/mrgr_no_spatial_lifting_v1/post_acceptance_integrity_review.json)。原报告、原验收和原始日志均保留，不覆盖历史证据。

训练进程只读取登记的train/val投影，未读取测试数组、指标或GT。与此同时，用户另行授权对旧冻结Uniform Fusion模型开展OARI机制诊断；该独立任务复用了历史测试标量指标并读取少量测试投影，未读取GT64或GT128，也未读取当前GeometryLifting新模型测试结果。该并行授权不等于整个会话从未接触历史测试指标；其结果未用于改变当前结构、训练、选轮或分析协议。

两完整对照和两消融均采用Uniform Fusion。新增预测统一封存时间早于第一次获准GT64评分；GT128始终禁止。本次是已使用测试集上的补充机制分析，不能单独证明可泛化几何规律。

[恢复登记](artifacts/mrgr_no_spatial_lifting_v1/recovery_execution_v1.json) · [状态汇总恢复](artifacts/mrgr_no_spatial_lifting_v1/post_training_recovery_v2.json) · [报告恢复](artifacts/mrgr_no_spatial_lifting_v1/report_recovery_v3.json) · [并行OARI边界](artifacts/mrgr_no_spatial_lifting_v1/parallel_OARI_diagnostic_boundary.json)
