# 最终普通交互模型交付复核与执行记录

模型冻结、经典补充评价和论文表格已完成；独立测试状态为 `BLOCKED_NO_INDEPENDENT_DATA`。具体模型、科学结论及边界以[冻结报告](mrgr_plain_final_model_freeze_v1_zh.md)、[经典比较](mrgr_plain_final_classic_comparison_v1_zh.md)、[论文表](mrgr_plain_final_tables_v1_zh.md)、[结果说明](mrgr_plain_final_results_v1_zh.md)及[独立数据审计](mrgr_independent_test_readiness_v1_zh.md)为准。

正式推理只运行一次，使用GPU 0的UUID锁及15秒双快照，2026-10-23 14:26:41至14:27:58（Asia/Shanghai）完成。仅新增385组主模型预测；经典预测2380组及主模型210组复用，既有路径、哈希与来源不变。没有正式训练、经典重建或干预其他用户进程。后续评分/统计运行CPU，GPU已释放。[调度任务](artifacts/mrgr_plain_final_v1/scheduling/inference.json)、[资源快照](artifacts/mrgr_plain_final_v1/resource_snapshots.jsonl)。

3,570场全部先封存后评分，旧结果可对照的2,800场正式体指标最大差为0。主要配对统计的200个向量以原始draw差值、重采样计数权重和独立排序分位数实现复核，最大差为1.1368683772161603e-13。[评分访问记录](artifacts/mrgr_plain_final_v1/GT64_access.json)、[统计复核](artifacts/mrgr_plain_final_v1/independent_statistics_verification.json)。这里的复核是算法实现交叉验证，并非新的独立测试集，也不是独立人员盲评。

注册预检曾出现三次工程中止，均在本轮推理及GT64评分之前：第一次将训练数据的相对哈希路径误作绝对路径；第二次把return_stages中的辅助残差张量也套用了C0/C1体场检查；第三次源码快照未兼容既有外部worktree路径。修复限于路径解析、输出检查对象和快照存储，不改模型、数值算法或实验定义。三次目录、日志、当时协议及失败记录分别保留在scratch的`mrgr_plain_final_v1_registration_attempt1/2/3`，轻量失败证据见[恢复记录](artifacts/mrgr_plain_final_v1/recovery)。第四次注册成功后，推理、评分、统计、报告与交付审计均成功，没有静默覆盖失败产物。[最终复核](artifacts/mrgr_plain_final_v1/completion_review.json)。

内存计时补充：`torch.cuda.max_memory_allocated`统计活动张量分配的峰值，包含计时进程中仍驻留的对象，**不包含仅由分配器保留、尚未分配给张量的reserved缓存**。论文成本表沿用原实测值；不能把该峰值或其模型间差值解释成模块的独立显存工作集。网络计时与经典求解计时边界不同，不据此宣称加速倍数。经典方法直接优化输入观测，因此其输入投影误差可能更低；在这个欠定重建任务中，这并不保证三维体场或留出射线误差更低，报告保留了这两类证据。

全部大型checkpoint和预测数组留在`/scratch/djc/GeoPhy-Set3D/mrgr_plain_final_v1`及原有来源目录。Git只纳入本轮脚本、协议、中文报告、标量CSV/JSON、来源路径及哈希。[交付文件哈希清单](artifacts/mrgr_plain_final_v1/delivery_manifest.json)覆盖运行证据；README新增当前入口，历史报告保持原样。

尚需用户决策仅限下一阶段：组织真正新增且去重的案例/GT64，确定目标域、采样数量依据及盲评保管方，然后另行预注册。当前35个案例不能承担独立确认。
