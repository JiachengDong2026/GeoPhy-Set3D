# 普通集合交互最终模型冻结与既有经典结果补充比较 v1

本阶段不训练、不调参、不实现深度基线、不绘方法架构图。按用户指定冻结seed20261011 epoch298普通集合交互主模型（342209参数，SHA256 0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c），seed20261019 epoch279仅参考。两者均从既有300轮验证投影H4严格最小值选轮；不比较测试误差选种子。当前架构已经历历史35案例探索，不能声称模型开发从未使用测试信息，区分训练/选轮无GT与研发独立性。

先核验两checkpoint、训练协议、300轮/24900更新、配置、源码、数据和访问日志；任一不一致停止。工厂必须调用scripts/oari_targeted_v1/variants.py:model_for(config,'uniform_plain_set_interaction',device)，不能单凭继承配置中的OARI name建模。源码快照和哈希、规范配置、划分、归一化、K96投影器、C0/C1、环境与加载命令全部登记。模型eval/inference_mode、FP32/TF32off，无optimizer、不改权重；CPU/独占GPU轻量推理检查，参数buffer canonical hash前后相同，已有预测可容差核对。保存原始文件路径，不重命名旧结果伪装新实验。

经典主表NNLS-NL/Tikhonov/TV共用正式非线性P_diff：质量分数三线性采样后转换摩尔分数，96中点积分。SART只作补充：线性节点摩尔分数CSR算子不同，统一P_diff仅用于其评价投影。核验历史final_lock、data_identity、源/恢复修订、配置/停止/校准规则、完整595键、input/H4站位顺序、原始观测/geometry/mask/hash、curve迭代及输出shape/finite/[0,1]/hash和历史全局封存。匹配则复用2380经典预测，不重建。数值失败不删，预算耗尽保留。完整普通交互已存210组预测逐条复用，其余385组补推理，共595组、C0/C1各595场，不新增第二种子17布局推理（第二种子仅已有普通/聚簇稳健性参考）。

正式35案例及17布局完全沿用历史：普通V4、spread-V4各draw0，V3/V6/clustered-V4各draw0..4。每组输入/geometry/mask/raw/归一化tensor hash均登记。先冻结模型/配置/分析源码和统计规则；全部新预测、旧主模型预测、经典预测统一hash核验封存（2975组、3570场）后才由独立CPU评分进程读取本35份授权GT64。GT128及Pool B数组始终禁止；训练/验证smoke不读GT。此前历史指标只作复算一致性核验，不改变任何方法。

评价直接调用正式volume_metrics/safe_shape/relative_projection_metrics，全量3D/形态指标和输入/H4投影指标保留，包括空值有效例数。经典与已有主模型volume/morphology均从封存数组重新评分，与旧同口径指标逐项核验；主模型全部测量诊断补算，经典冻结测量诊断按身份复用并作固定案例投影复核。不得将相同案例draw视独立样本。

唯一主要端点普通V4 C1 relativeL1。五类布局各案例先聚合draw均值再统计35案例；逐draw也保存。配对Δ=主模型−经典，负值有利（核心消融单独用ablation−full，正值为变差，两表明确）。直接沿用正式20000次案例bootstrap、seed20261019、百分位95%；四经典构成当前固定对照family，保存Bonferroni四比较区间，并额外保存历史20对照保守区间。主表三个，SART补充，不能利用主表省略改变family。均值/中位数/P90/max和有效数、配对均值/中位数/增降不变(|Δ|≤1e-8)、逐案例向量、各均值案例区间保存；辅助指标均探索性。独立案例计数权重与独立排序插值复核bootstrap。重用35案例所得一切区间不赋予独立确认性。

核心消融和复杂度不重跑，核验最新六项acceptance、完整checkpoint/日志/统计与benchmark。输出两个种子两布局的主/消融指标与均值/中位数/区间/例数，辅助指标不利结果保留。成本来自真实batch1同GPU验证0/17/34的一次warmup+三次测量、训练实际elapsed和峰值；明确历史benchmark峰值allocated包括进程已有缓存/模型，非独立增量工作集。FLOPs/MACs/能耗未测，标缺失。经典秒数含求解器恢复I/O、几何setup另列；SART CPU，不与网络前向直接计算加速倍数。

独立数据审计仅遍历项目自有raw/processed目录与元数据，读取原始manifest、source_index、frame_manifest和split/访问历史，原始targets只列名/stat不打开。核对case/source-run/GT来源hash、debug/重复/布局/PoolB同case血缘；查找新增数据目录/未登记来源。无法覆盖服务器外未知数据，结论限定审计范围。若没有可证实新case，状态BLOCKED_NO_INDEPENDENT_DATA，不运行独立评价；建议初始至少35个新独立source-run且目标50–100（数据规划，非功效论证），与237旧case做case/run/hash去重，另预注册按案例样本量/精度或功效依据、固定模型/布局/指标/停止与盲评分，不自动启用PoolB/GT128。

资源只用真实空闲GPU，UUID锁FD、15秒双快照、主机资源门限，遇其他任务等待，保持既有工作流，不发信号给他人。独立scratch目录和Git轻量文本/CSV/hash证据，不提交权重/数组。运行源及协议先冻结，异常保存后审计恢复不静默覆盖；README新增当前入口，历史报告不改。最终验证完成情况/阻塞、commit和远端hash必须实际读取。
