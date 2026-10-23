# 最终主模型冻结与三种深度学习基线：执行状态

本轮已完成最终 Plain 模型冻结、三模型独立实现、CPU/GPU 验证、固定小规模训练和四方法验证评价入口的实际集成。**正式比较尚未完成**：DeepSets/GNOT 双种子共4项已登记并进入持久 GPU 队列；Senseiver 保留历史科学 NO-GO，不重复同一路线300轮。本文及导出状态为2026-10-24（北京时间）快照，运行中的 epoch 以 scratch 状态为准。

## 1. 主模型与历史成果保护

当前主模型是 `uniform_plain_set_interaction_seed20261011`，342209参数，best epoch298。checkpoint SHA-256：`0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c`；seed20261019参考权重为 `fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d`。原冻结 commit 为 `ee6ac52e07034aee1b23b5779de54818d9b43d27`。参见[完整冻结清单](final_model_freeze_manifest.md)。

代表性验证输入和 C0/C1 输出均已封存；集成前及集成后 CPU 重算均与锚点逐位一致，两个完整权重哈希不变。新增算法通过独立工厂构建，未修改已有 `src` 文件、模型注册表、数据集、loss、物理投影、历史训练/评价代码。经典算法11项回归通过；其已有预测和报告未覆盖。未重训/微调主模型。

## 2. 实现、来源与信息条件

| 模型 | 参数 | 架构 | 状态 |
|---|---:|---|---|
| 最终 Plain | 342209 | Encoder+固定GeoLift+Plain交互+多尺度MRGR+UniformFusion+PRF | 冻结复用 |
| Ray-DeepSets-v1 | 104577 | 203→128逐射线phi、masked mean、latent与96维坐标Fourier的MLP解码 | CPU/GPU/pilot通过 |
| Ray-Senseiver-v1 | 254913 | 原ray适配结构、64×64潜变量、交叉/自注意力、分块坐标query | CPU/GPU/pilot通过，科学NO-GO |
| Ray-GNOT-v1 | 527109 | width64/2层/4头/2专家，异构归一化线性注意力、全局query统计、XYZ门控专家 | CPU/GPU/pilot通过 |

三个模型均输出 FP32 `[B,1,64,64,64]`、XYZ顺序、`[0,0.6]³ m`节点上的CH4质量分数，sigmoid范围[0,1]。203维token含合法积分值与 entry/exit/direction/path/Fourier几何，保留有效零积分，正确排除无效NaN与padding。H只进物理loss/selector，不进重建网络。积分输入并非点传感器浓度。

GeoLift无可训练参数，但周围的Encoder/Decoder属于主模型学习表示。因此三基线不共享它们；历史306625参数的GeoLift-DeepSets只作为受控架构实验留存，不能替代本轮标准DeepSets。详见[训练前公平性审计及共同协议](deep_learning_baselines_plan_zh.md)。

Senseiver复用固定BSD3官方来源与原结构，明确标注射线积分适配；GNOT保留论文核心算子/几何门控。GNOT官方核验版本无明确LICENSE，未vendor官方源码，当前为公开数学结构的独立实现。分别见 [DeepSets](baseline_deepsets_implementation_v1_zh.md)、[Senseiver](baseline_senseiver_implementation_v1_zh.md)、[GNOT](baseline_gnot_implementation_v1_zh.md) 来源与适配说明。

## 3. 实际验证与计算成本

CPU合并验证：模型/旧Senseiver回归20项、统一objective与统计9项、经典算法11项，合计 **40 passed，另3 subtests passed**。统一测试包含真实完整64³输出接原六路K96投影的自监督loss反向、所有参数有限梯度、H无梯度且不进入网络、GT/重叠站位拒绝、不同批量/观测数量、掩码、排列不变性及checkpoint重载。

独占GPU真实训练batch2，三者均通过完整A/B/H objective、backward、clip、AdamW更新、输出形状/有限性/范围、测量变化响应与checkpoint逐位往返验证：

| 模型 | smoke总秒数 | batch2单次推理秒 | smoke峰值分配GiB | 参数FP32字节 |
|---|---:|---:|---:|---:|
| DeepSets | 2.063 | 0.009604 | 0.526 | 418308 |
| Senseiver | 2.544 | 0.015460 | 0.633 | 1019652 |
| GNOT | 4.193 | 0.131720 | 1.906 | 2108436 |

这里的推理是一次同步前向，未做独立预热/重复benchmark；smoke总时间还含loss、更新、保存重载等操作，峰值含反向，不能冒充正式训练总成本。

恢复对照使用原完整epoch提交引擎、workers4，实际三个新进程比较连续2轮与1轮+重启续1轮：model/best/AdamW/cosine/各RNG/loader/选模与指标通过登记容差。整数/RNG精确一致，float rtol1e-4/atol1e-6。此项验证的是**epoch边界跨进程恢复**，不是epoch中途强杀试验；中断epoch重播仍沿用原引擎语义。

## 4. 固定小规模pilot

三个模型各seed20261011，fresh初始化，4train/2val，5epoch/10updates，原batch2/AdamW/clip/loss，cosine仍T_max300。没有GT训练、监督warm-start、拟合输出先验或超参数搜索。

| 模型 | val H4 relative L1（e1→e5） | val总loss（e1→e5） | epoch5输出均值 | 解释 |
|---|---|---|---:|---|
| DeepSets | 269.9657→266.2335 | 57.0350→56.2461 | 0.48636 | 默认sigmoid尺度仍很大；工程可运行，未证明有效学习 |
| Senseiver | 206.8669→85.0279 | 43.7687→17.9422 | 0.12401 | 10更新改善不能推翻历史长期塌缩/修复失败 |
| GNOT | 73.4060→1.9578 | 15.4970→0.4167 | 0.0018123 | 数值有限、初期下降；仍需检查近零/模板塌缩 |

这两例验证子集的H4值不等于完整36例selector或三维误差，不能用于正式模型优劣结论。三者无异常NaN/Inf或严格常量输出，但短pilot不足以保证条件重建能力。原Senseiver科学记录为 `REPAIR_FAILED_NO_GO`，包括原自监督113epoch/9379updates近零塌缩，以及有限初始化/loss/读出修复失败；监督旧checkpoint有GT，不能warm-start或计入本轮自监督成绩。

## 5. 正式训练真实状态

当前有效计划为 scratch `formal_queue_plan_v2.json`；**只有** `training_registered_v2/` 下4项协议可作为本轮当前正式任务。早先 `training/` 中4份未启动登记已保留并标记superseded，正式更新为0，不应手动启动。

| 实验 | 种子 | 快照状态 | 预算 |
|---|---:|---|---|
| DeepSets | 20261011 | 等待独占空闲GPU | 300epoch/24900updates |
| DeepSets | 20261019 | 等待独占空闲GPU | 同上 |
| GNOT | 20261011 | GPU0正式训练中，已提交完整epoch，实时进度见状态CSV | 同上 |
| GNOT | 20261019 | GPU3冲突后已复核，排队等待GPU0恢复 | 同上 |
| Senseiver | 两个种子 | 历史科学NO-GO阻塞，未启动正式训练 | 不重复同一路线300轮 |

GNOT正式前两轮完整166train/36val每轮约139/137秒，H4约1.0016/1.0010，已接近零场基准1，**存在塌缩风险，尚无有效性结论**。短期改善不能证明模型利用观测。当前不按前期效果调整网络、loss、seed、输出bias或提前截短登记预算。完整训练成本和结果待完成后报告，不把上述pilot作为正式训练完成。

所有正式设置逐字段对齐冻结主模型：数据划分/归一化/A-B-H采样、300轮、24900updates、batch2、AdamW2e-4/wd1e-4、cosine300、FP32/TF32off、clip1、loss及H4最低值选模。不承诺不同架构共享模块初始化逐参数相同；实际初态state hash已登记并在CUDA构建时核验。

## 6. 资源、锁与异常

重新检查四张RTX4090后，只有GPU0持续空闲，GPU1–3为其他用户任务。使用全局UUID flock、15秒双空闲快照、锁FD继承、独立run/controller锁及资源快照，不依赖旧截图。原4个控制器允许空闲卡动态并行；之后发生一次真实冲突：GPU3在其他用户多段训练间隙通过双空闲快照，本任务seed19开始后，对方下一段PID1377578启动。冲突监控只停止本任务的进程组，没有终止对方进程。

原队列按故障协议整体停止，也中断了GPU0上GNOT seed11尚未提交的epoch4。已检查日志/PID归属/哈希并保存 `inspected_resource_recovery.json`：seed11的权威last.pt完整提交epoch3/249updates，seed19没有完整epoch提交。使用新增独立 `recover_queue.py` 在**当前进程中仅过滤GPU调度到UUID为GPU0的卡**，不修改冻结训练worker、模型或配置；按原last.pt恢复RNG/AdamW/cosine并重播未提交epoch，之后依次执行剩余原4项。GPU1–3的短暂空档不再视为可抢用资源。原队列启动记录保留，当前持久恢复队列见 `inspected_recovery_launch.json` / `queue_status.json`。

24900指正式完整epoch提交的更新数，中断epoch内已执行但未提交的步骤会重播，具体被丢弃步数无法从epoch粒度日志精确恢复，不能宣称总计算尝试次数恰好24900。没有更改正式有效更新预算或丢弃已提交checkpoint。

最终控制器使每项任务独立session/进程组，识别自身训练/loader/嵌套检查进程；发现外来冲突仅清理自己的进程组、保留epoch checkpoint并阻塞自动重试。失败重启需检查记录后提供 `--recover-reason`，不静默修改定义。

工程期间保留了旧guard输出根目录错误、`queue.py`遮蔽stdlib导入失败、恢复检查误判自己嵌套进程，以及独立审计发现的异常启动清理边界。均在正式更新前修复并保存记录。第一次恢复检查只停止本任务检查进程，后续v2新目录对照通过；为避免覆盖冻结身份，正式4项与mini均重新登记v2，早期文件仍保留。详见 `artifacts/deep_baselines_v1/engineering_issues.json`、`scheduling/recovery.jsonl` 与完整本地 `scheduling/`。没有改主模型或依据GT调整算法。

## 7. 四方法统一验证评价：已实际跑通，但非正式比较

固定使用官方Val G1–G5清单的case_index0、1及draw0，每方法10条。四方法全部40份预测封存后，才只读这两个验证案例GT64；GT阶段不读checkpoint/测量，inference阶段不读GT。最终v2保存了逐预测hash、输入/H哈希、逐draw/逐case CSV、各组统计、paired CSV、参数与推理成本。GT128、Pool B和test均未读。

| 方法 | G1三维relative L1 | G5三维relative L1 | 训练量说明 |
|---|---:|---:|---|
| 冻结Plain | 0.071440 | 0.119444 | 历史300epoch主模型 |
| DeepSets pilot | 359.122291 | 358.977439 | 本轮仅10updates |
| Senseiver pilot | 90.382168 | 90.433255 | 本轮仅10updates |
| GNOT pilot | 2.239854 | 2.140213 | 本轮仅10updates |

**该表只证明错误指标可计算并暴露尺度/学习风险，训练量不对等，不能排名、写入论文正式精度表或据此宣称几何鲁棒性。** 值为比值而非百分比。只有2案例/1draw也不支持稳定区间或布局效应结论；完整值在 `mini_integration_summary.csv`，完整正式比较表的未运行指标为空。

评价入口支持固定36×5×5=900 Val清单：先draw均值后case宏平均、L1/L2/MAE/RMSE/质量与峰值幅度误差、观测/H测量误差、G5−G1、grand-mean G5/G1、P90/P95/max/worst、10000次case bootstrap seed2026101801，并检查四方法的record/input/H哈希对齐。种子分开，不将draw当独立案例。[入口与命令](deep_baselines_evaluation_interface_v1_zh.md)。`formal=True`仅标完整900条Val范围，**不是独立测试身份**。

独立统计复核另用bincount权重重算真实保存的mean/median/P90/P95/max/CI和配对统计，而非调用同一汇总函数。最终v2共2990个统计字段，最大差异0.0；31个输入JSON均有哈希，只读保存的JSON、没有再读GT/预测/测量/checkpoint数组。机器证据在 `artifacts/deep_baselines_v1/independent_review/`。

## 8. 测试边界与未完成工作

本轮**未执行正式测试**，也未做完整900条Val正式比较。三种基线尚未全部得到有效且完整预算的冻结checkpoint，Senseiver科学门槛未通过，不能宣布具备最终三算法公平比较条件。原35例参与过模型开发，沿用探索性标签；独立数据审计仍为 `BLOCKED_NO_INDEPENDENT_DATA`，不以重新命名作为独立测试。

下一步是保持源码/配置冻结，完成已排队的4项训练并验收完整预算与H4选模；如最终仍塌缩，按失败基线如实报告，不自动扩大搜索。完成后可预先登记各选定checkpoint，对完整36例Val900清单评分。Senseiver只保留当前适配失败与监督可学性诊断的边界，若后续要改变任务或结构必须另行定义独立研究，不能将其补成当前成功自监督对照。

## 9. 复核入口

完整本地根目录：`/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/`。代码、配置、日志、source/checkpoint/prediction hashes和CSV的小体积导出位于 [Git证据目录](artifacts/deep_baselines_v1/)。权重、全部64³预测及完整GPU进程快照保留本地，未把GT数组放入Git。

当前队列：`queue_status.json` / `inspected_recovery_queue.log`（原 `formal_queue_v2.log` 保留故障记录）；单任务：`training_registered_v2/<arm>_seed<seed>/status.json`、`metrics.jsonl`、`checkpoints/last.pt`及`best.pt`。last.pt是完整epoch提交权威，禁止用test GT选择checkpoint。进程和GPU状态是快照，不能将等待误写为完成。

正式汇总CSV：[formal_comparison_status.csv](artifacts/deep_baselines_v1/formal_comparison_status.csv)；工程指标：[mini_integration_summary.csv](artifacts/deep_baselines_v1/mini_integration_summary.csv)。待训练完成后，运行 `scripts/deep_baselines_v1/deliver.py` 可重新导出状态；不要覆盖原始scratch实验记录或修改登记哈希对应的数值源码。
