# GeoPhy-Set3D

2026-10-20 **均匀融合与反馈OARI双种子三维配对评价已完成**：两种子Control和两种子Treatment的35例/17布局评价、四种经典算法历史对比及独立验收归档。7个骨干消融暂缓，是否将反馈OARI作为最终模型等待审阅，未自动晋级。沿用此前已使用的35例，属于补充评价。[阶段总报告](docs/reports/mrgr_uniform_final_v1_zh.md)、[OARI结果](docs/reports/mrgr_uniform_oari_v1_zh.md)、[执行协议](docs/mrgr_uniform_final_v1_protocol_zh.md)。

2026-10-20 新增反馈OARI、均匀融合与描述符结构删除的 **35例补充测试已完成并通过独立验收**：7个新增冻结模型各测试17布局，复用4个已核验历史模型，共13,090个场独立重算。反馈OARI相对匹配MRGR，普通/聚簇三维L1分别为 **8.5089%→8.3140% / 31.3236%→29.7930%**，峰值误差有所增加。本轮沿用此前已使用的35例，不构成全新独立保留集确认。[完整结果、配对区间与固定切面](docs/reports/mrgr_extension_test35_v1_zh.md)、[验收记录](docs/reports/artifacts/mrgr_extension_test35_v1/acceptance.json)、[导出哈希及完整原始指标入口](docs/reports/artifacts/mrgr_extension_test35_v1/export_manifest.json)。

2026-10-20 追加开发实验已完成：GCA均匀融合/可观测性描述符删除的两种子300轮消融，以及残差反馈OARI的20轮筛查和单种子300轮延长实验。反馈OARI相对本轮匹配MRGR，普通/聚簇三维L1分别为 **9.4688%→9.0465% / 33.7562%→30.5974%**，聚簇质心误差降低，峰值误差略有增加。仅使用166训练/36已用开发例；不构成独立测试结论。[结果索引、完整报告与证据](docs/reports/mrgr_development_results_20261020_zh.md)。

MRGR 论文最终 **35 例测试已完成并通过独立验收**：21 种方法、17 个固定布局，共 12,495 次重建、22,610 个阶段场。原始 MRGR 普通固定 V4 的三维相对 L1 为 **8.5159%**；四种经典对照为 NNLS-NL、Tikhonov、TV、SART，七项因素消融各用两个种子完整训练。全部预测封存后才进行测试 GT64 评分，未读 GT128，评分后未重新训练。完整结果、负结果、统计限制与清晰切面见[35例测试报告](docs/reports/mrgr_paper_final_test35_zh.md)、[预登记规则判读](docs/reports/mrgr_paper_final_interpretation_zh.md)、[验收记录](outputs/research/mrgr_paper_final_v1/final_acceptance.json)。Git 中保留代码、标量与图表证据；数据、权重及大体积预测数组保留本地。

新增方案 A（集合初始化＋显式物理展开）独立实现与复现入口，支持 A0/A1/A2/A3 及参数匹配 A-control；配置、训练/续训、推理与冻结评价见 [中文实现与验收说明](docs/plan_a_implementation_zh.md)。工程验收和正式训练状态以该说明及实际运行产物为准，下面的历史阶段记录继续保留。

方案 A 性能优化 v2：固定案例完整更新实测约 5.75 倍加速；真实 166/36 train/val 一轮约 5.58 分钟，300 轮线性估算约 28 小时/种子。以上为共享单张 RTX 4090 的测量；优化方式、数值核验和适用限制见 [性能优化说明](docs/plan_a_performance_v2_zh.md) 与 [测速报告](docs/reports/plan_a_perf_v2_zh.md)。正式长训练未启动。

追加方案 A 执行优化 v3：独立启动器支持保持 FP32 的先验编译＋cuDNN 自动算法选择，smoke20/gate16 均通过；负载降低后的同阶段复测为普通 FP32 1.51 秒、组合配置 1.16 秒/更新。完整166/36一轮2.63分钟，按当前较轻共享负载估算300轮约13.16小时/种子，不能将相对旧28小时的全部差异归因于优化。TF32/BF16 另列为降精度实验候选，未替换默认精度。配置、来源隔离和完整证据见 [第二轮优化说明](docs/plan_a_runtime_v3_zh.md) 与 [实验报告](docs/reports/plan_a_runtime_v3_zh.md)，未启动正式长训练。

方案 A 五小时预算 v4：多尺度三维先验＋BF16网络/FP32物理＋完整几何缓存，独立smoke/gate通过；共享单张RTX 4090完整轮次实测约38秒，300轮估算约3.2–3.9小时/种子，含每轮验证与检查点。保留K4/Q96/64³与原始损失，三维精度仍待正式验证；见[预算版本说明](docs/plan_a_budget_v4_zh.md)与[实测报告](docs/reports/plan_a_budget_v4_zh.md)。已按用户要求启动seed20261011的300轮任务，实际状态以运行目录为准。

方案 B 首版：32³集合条件特征＋61维连续坐标MLP＋Q96直接连续射线积分，支持训练/续训及64³、128³测量重建。CPU30项及GPU专项通过；16案例40更新的留出relative-L1从2.200降到1.783，仍差于零预测1.0，学习gate未通过，未启动B正式训练。B代码独立于A冻结源码；见[设计与接口说明](docs/plan_b_implementation_zh.md)和[验收报告](docs/reports/plan_b_v1_zh.md)。

当前阶段：[有监督Ray-Senseiver稳定性与正式基线](docs/reports/ray_senseiver_supervised_v1_zh.md)已实际完成稳定性验证、正式训练、冻结及900条对比。八例在2000/2200/2400步连续通过，终点NMSE=.054545、L1=.317464；正式fresh训练完成187轮+下一轮59更新（共15580步），按2小时墙钟停止，收尾保存后7209.69秒，末次更新在7199.39秒内完成。best第187轮Val NMSE=.286811、L1=.638421；三例测量依赖全部有效，错配均增大GT误差。**监督可学已验证，但当前适配不具竞争性强基线精度，不再追加训练/修复。** 冻结900条G1/G5 L1：Senseiver=.647051/.739676、监督MRGR=.055386/.374038、自监督Ours=.088386/.298018、DeepSets=.135463/.618805。保留为有监督独立参考，不能纳入纯自监督排名；不同loss/selector/实际预算不能只归因架构。原始254913参数/完整64³保持，34项适用检查及最终访问/源码/更新审计通过，历史自监督失败与旧FP32 consistency回归失败保留。[协议](docs/ray_senseiver_supervised_v1_protocol_zh.md)、[正式配置](configs/train_ray_senseiver_supervised_v1.yaml)、[训练](scripts/train_ray_senseiver_supervised.py)、[冻结评价](scripts/evaluate_senseiver_supervised.py)、[最终审计](docs/reports/artifacts/ray_senseiver_supervised_v1/integrity.json)、[900配对结果](docs/reports/artifacts/ray_senseiver_supervised_v1/paired_900.csv)。

上一阶段（历史结论，当前授权已保留监督候选）：[Ray-Senseiver训练集GT64有监督可学性诊断](docs/reports/ray_senseiver_supervised_probe_v1_zh.md)已实际完成。原始结构单例500步连续通过（NMSE=.03243、三维relative L1=.26464）；八例2000步终点NMSE=.07147、L1=.39627，较平均模板降低90.19%，3个固定case测量依赖全部达标且错配增大GT误差。但1900步L1=.57105，尚未连续两次通过，按预算归档 **BUDGET_EXHAUSTED_UNRESOLVED**，不登记正式训练候选。保留积极条件学习证据，不宣称架构不可学；暂停当前Senseiver路线，沿用冻结DeepSets作为独立学习基线，不追加修复、补步或长训练。
本轮仅登记train前8例/Pool A及其GT64，原始254913参数、完整64³、单一归一化监督MSE、固定LR、fresh初始化；无GT128/Val/test/Pool B、外来权重或阶段A warm start。新增5项及原模型4项测试通过，历史consistency的1项FP32回归失败继续保留。**有监督诊断不进入纯自监督榜单，权重不得用于后续标称无GT的训练。** [协议](docs/ray_senseiver_supervised_probe_v1_protocol_zh.md)、[配置](configs/ray_senseiver_supervised_probe_v1.yaml)、[入口](scripts/run_senseiver_supervised_probe.py)、[标量/图/访问与退出审计](docs/reports/artifacts/ray_senseiver_supervised_probe_v1/)。

上一阶段：[Ray-Senseiver快速定位与单次修复闭环](docs/reports/ray_senseiver_quick_repair_v1_zh.md)已完成，**修复失败归档，正式训练NO-GO**。固定四站16例冻结latent读出中，岭回归/query头检查MSE=1.32e−8/2.39e−5（常量4.97e−4），两者均可读；随后只新增4096参数的全局latent解码残差，重新fresh对照。Control35轮/140步、Treatment30轮/120步均FAILED_TO_LEARN；共同e30 H L1=1.06066→1.03528，但L2=.99306→.99587，测量依赖3例均未达1e−3，不能作为有效修复。
参数共享初态/完整query/真实更新/退出和访问审计通过，无GT64/128、Val、Pool B或test，无历史/读出/smoke权重warm start。新增7项及原模型4项测试通过；旧consistency幅度回归仍有1项FP32数值失败，原日志和有限float64核验保留，不改旧函数或容差。[协议](docs/ray_senseiver_quick_repair_v1_protocol_zh.md)、[唯一设计](docs/ray_senseiver_quick_repair_v1_design_zh.md)、[训练入口](scripts/run_senseiver_single_repair.py)、[结果与决策](docs/reports/artifacts/ray_senseiver_quick_repair_v1/)。本轮停止；唯一后续选择是暂停当前适配路线、沿用已冻结可学习的Ours/DeepSets作为比较对象，不追加结构/loss搜索或300轮。

上一阶段：[Ray-Senseiver 测量信号通路独立诊断](docs/reports/ray_senseiver_signal_path_v1_zh.md)已完成，正式训练仍 **NO-GO**。只读最新consistency Treatment的epoch0/last35，完成30次完整64³干预前向、6个成功259-query局部梯度图，0次optimizer更新；另保留1个工程异常中止局部图及两次版本化执行器修订。
测量正确进入token，预处理与latent响应非零；末态置零测量的浓度relative MA仅1.77e−5–1.11e−4，错配也几乎不改变原A/H拟合。测量权重列未消失，latent响应较初态弱，sigmoid缩小绝对响应但相对浓度响应系数接近1，尚不能锁定唯一瓶颈。22/24个logit有限差分未通过，原值及访问审计局限完整保留；12项诊断/模型回归通过，参数/buffers/历史hash不变，无GT/Val/Pool B/test。[协议](docs/ray_senseiver_signal_path_v1_protocol_zh.md)、[入口](scripts/diagnose_senseiver_signal_path.py)、[标量及图表](docs/reports/artifacts/ray_senseiver_signal_path_v1/)。
唯一后续任务建议：独立预登记固定同几何、真实测量均值的latent→标量读出可学习性对照，检验现有latent信息是否能被可靠利用；本轮未实现。完成后停止，不训练、不调loss、不扫描增益或继续搜索，不将当前失败适配解释为原Senseiver能力上限。

上一阶段：[Ray-Senseiver consistency 开/关最后一次小预算对照](docs/reports/ray_senseiver_consistency_probe_v1_zh.md)已完成，正式训练仍 **NO-GO**。
两臂relative均为0，仅consistency权重.05/0；重新fresh训练均在35轮/140次更新触发 `FAILED_TO_LEARN`。Treatment H L1/L2=1.06057/0.99306（Control=1.06063/0.99308），SmoothL1较初态改善40.526%，但H L2恶化3.472%。case差异增加，测量依赖虽增加仍3例均未达1e−3；仅保留有限指标改善的 `PARTIAL_LEARNING_EVIDENCE`，不作为有效强基线。
同图loss/全参数及输出梯度等价与唯一差异、17项测试、固定8例访问及最终checkpoint/更新/退出审计通过。只用原8例train，无GT/Val/Pool B/test；历史与冻结Ours/DeepSets保持。**本轮结束逐项loss排查，不追加搜索或长训练**。唯一后续建议：独立检查当前ray-token编码与latent bottleneck能否有效利用测量信息，本轮不实现。[预登记协议](docs/ray_senseiver_consistency_probe_v1_protocol_zh.md)。

上一阶段：[Ray-Senseiver 测量loss梯度拆分与relative-L1开/关对照](docs/reports/ray_senseiver_relative_probe_v1_zh.md)已完成，正式训练决策 **NO-GO**。
两臂均在35轮/140次更新触发 `FAILED_TO_LEARN`。关闭relative项后H L1/L2=1.06070/0.99307（Control=1.00554/0.99931），测量依赖仍不足。
仅SmoothL1改善保留 `PARTIAL_LEARNING_EVIDENCE`，不视为有效基线或GO。近零常量方向诊断支持relative项压低预测的局部机制，但不能外推所有模型；全程只用原8例train，无GT/Val/Pool B/test、无新300轮或搜索。[独立协议](docs/ray_senseiver_relative_probe_v1_protocol_zh.md)。

更早阶段：[Ray-Senseiver 输出初始化两臂探针（中文报告）](docs/reports/ray_senseiver_init_probe_v1_zh.md)已完成，结果 **NO-GO**。
原近零塌缩 run 在113轮/9379次更新安全停止并完整归档；新 Control/Treatment 均在35轮/140次更新触发预登记 FAILED_TO_LEARN。
测量尺度初始化将初态mean纠正至0.001558，但Treatment末轮训练H L1=1.00554、测量依赖不足；未达到严格塌缩阈值。
全程无GT、Pool B或test，冻结Ours/DeepSets；本轮未启动新300轮或额外搜索。[协议与复现入口](docs/ray_senseiver_init_probe_v1_protocol_zh.md)。

基于物理约束的三维气体羽流重建研究项目。

当前深度学习比较阶段已完成：统一最小比较接口与 **GeoLift-DeepSets-Sum 首轮自监督同条件对照**。新baseline完整300轮/24,900次更新，306,625参数，原v1 frozen参考不重训；同原900条V4清单先保存预测再独立Val GT64评价。Ours的五档3D L1均36/36 case胜出，G1–G4 macro降低35.33%，G5从DeepSets=0.618805降至Ours=0.298018（降低51.84%，Δ95%CI=[−0.343692,−0.296426]）。L2与held-out方向一致；普通几何峰值误差优势不确定。Ours参数多8.30%、实测前向median约3.12倍，G5仍有残余高误差。单seed/共享表示基础比较，不归因单个模块，不代表超过Senseiver等强基线；未读GT128/Pool B/最终test，未改Ours或loss，历史结论保留。见 [中文报告](docs/reports/deepsets_selfsup_v1_zh.md)、[训练前协议](docs/deepsets_selfsup_v1_protocol_zh.md)、[统一接口](docs/deep_learning_comparison_interface_zh.md)、[配置](configs/train_deepsets_selfsup_v1.yaml)、[训练](scripts/train_deepsets_selfsup.py)、[独立评价](scripts/evaluate_deep_comparison.py)、[900配对明细](docs/reports/artifacts/deepsets_selfsup_v1/paired_900.csv)。唯一优先下一阶段建议：独立预注册并接入Senseiver同条件任务适配；本轮不自动执行。

此前诊断阶段已完成：original frozen SelfSup-MRGR v1 的 **C0/C1 五档几何配对诊断**。共1,818次冻结前向（normal900、zero900、固定signflip/wrong_pairing各9），保存预测后独立GT64评价，正常C1复现原900条全部十指标。G1–G5每组36/36 case的三维L1均改善；G5从C0=0.398727降至C1=0.298018（改善25.26%，Δ95%CI=[−0.118003,−0.084900]），主要符合“初始重建已差、修正仍有帮助”。逐布局有5条L1恶化，辅助L2也并非逐case一致改善。zero和有限shortcut支持signed residual信息依赖，但不足以优先实施坏几何抑制门控；唯一下一阶段建议是独立确认固定V4的几何互补单站替换。未训练或实现新模块，历史GO/NO-GO保留，256项回归通过。见 [中文诊断报告](docs/reports/selfsup_v1_c0c1_geometry_diagnostic_zh.md)、[冻结协议](docs/selfsup_v1_c0c1_geometry_protocol_zh.md)、[独立配置](configs/diagnose_selfsup_v1_c0c1.yaml)、[诊断入口](scripts/diagnose_selfsup_v1_c0c1.py)、[900配对明细](docs/reports/artifacts/selfsup_v1_c0c1_geometry/paired_900.csv)。C0是联合训练中间输出，内部比较不等同有无MRGR训练消融。

上一正式训练阶段已完成：显式 Geometry Augmentation matched experiment **NO-GO**。两臂fresh各300epochs/24,900steps，旧Control审计拒绝复用；新50个确认V4布局、36Val案例、两臂5,760次冻结推理已完成。确认G5 L1从0.26747降至0.16291（改善39.09%、36/36改善、delta95%CI=[−0.12429,−0.08661]），但G1–G4 macro恶化11.13%，未满足3%非劣界限且超过PARTIAL的10%上限。完整回归252项通过，训练/selector禁读尝试0；先measurement-only冻结审计，再独立posthoc GT64。见 [正式报告](docs/reports/geometry_augmentation_formal_v1_zh.md)、[预注册协议](docs/geometry_augmentation_formal_v1_protocol_zh.md)。不追加比例/阈值搜索或训练；该阶段结束时建议单独预注册model-side geometry awareness。

历史敏感性阶段：原始冻结 SelfSup v1 的五档 V4 Geometry Sensitivity Analysis 已完成：36 Val×25几何，共900次真实推理。G1–G5 relative L1=0.08839/0.08589/0.08951/0.09925/0.29802；G5−G1 case95%CI=[0.17768,0.24472]、36/36案例变差，但组均值不严格单调。中心分数与误差Spearman为−0.817，去掉极端组仍为−0.647。测量拟合与voxel误差实际强相关（Pearson/Spearman=0.902/0.952），不支持预设“弱相关”。见 [中文总结](docs/reports/geometry_sensitivity_summary.md)、[优化前冻结协议](docs/geometry_sensitivity_v1_protocol_zh.md)、[逐case×geometry CSV](docs/reports/artifacts/geometry_sensitivity_v1/geometry_sensitivity_results.csv)。25个8³线性chi稀疏算子谱只作粗网格诊断，不能当完整64³可观测性或唯一恢复证明；exploratory hull-area score明确使用Val GT误差选择，不改旧分数/边界。

后续判断：几何是可操作的显著瓶颈之一；本轮没有容量/优化/损失干预，不能证明它比网络容量更主要。已独立实现50% natural/25% moderate/25% hard sampler，物化49,800对前瞻站位计划并登记 [准备协议](docs/geometry_augmentation_preparation_v1_zh.md)；九种mode组合的无GT forward/backward smoke已实际通过，初态/参数/缓冲不变、optimizer steps=0；该入口没有正式训练分支。随后正式增强训练已按上面的独立预注册完成；curriculum/hard weighting、V3/V6新五档、Pool B、最终test均未启动，原v1架构、损失和自然Val measurement selector保留。

历史阶段保留：两个冻结模型的三档V3/V4/V6验证Stage C GO见 [报告](docs/reports/geometry_strata_frozen_v1_zh.md)；fresh同q A/B匹配300轮Stage D **NO-GO**见 [报告](docs/reports/geometry_stratified_selfsup_matched_v1_zh.md)，它主要改变A/B联合分布，并未检验本次显式极端增强。RVTC关闭，经典算法有限预算结果见 [原报告](docs/reports/classical_v4_comparison_v1_zh.md)，不称理论下限。历史数据、网络、checkpoint、数值训练代码、配置、实验结论及用户未跟踪绘图脚本保留。

## 开发环境

已有 Conda 环境：`gas-torch`，位于 `/scratch/djc/envs/gas-torch`。

```bash
source /data/apps/miniconda3/etc/profile.d/conda.sh
conda activate gas-torch
cd /home/djc/projects/GeoPhy-Set3D
```

环境采用 Python 3.12.15、PyTorch 2.11.0 和 CUDA 12.8，已验证 RTX 4090 上的 GPU 运算。
`requirements.txt` 保存初始化时的完整 pip 依赖版本，包含训练、科学计算和 Jupyter 工具。
Notebook 可选择 `Python (gas-torch)` 内核。

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `src/geophy_set3d/data/` | 观测数据读取、预处理和数据集 |
| `src/geophy_set3d/geometry/` | 三维坐标、空间离散与观测几何 |
| `src/geophy_set3d/physics/` | 物理方程、边界条件和约束项 |
| `src/geophy_set3d/models/` | 模型组件 |
| `src/geophy_set3d/reconstruction/` | 重建与优化流程 |
| `src/geophy_set3d/visualization/` | 三维场和实验结果可视化 |
| `configs/` | 数据、物理参数和实验配置 |
| `scripts/` | 数据处理、训练、重建和评估入口 |
| `notebooks/` | 数据探索与方法验证 |
| `docs/` | 项目规划、研究笔记、参考资料和报告 |
| `data/raw/` | 原始数据入口 |
| `data/processed/` | 处理后数据入口 |
| `outputs/` | 模型权重、运行日志和图像输出 |

实际数据存放在 `/data/djc/GeoPhy-Set3D/data/`，项目的 `data/raw` 和 `data/processed` 是指向该位置的符号链接。
已有 Fluent 数据集可通过 `data/raw/Fluent/static_final_frame_partial_v1/` 访问，目录说明见 `docs/data-layout.md`。
原路径 `/data/djc/Fluent` 已保留为兼容链接。

实验输出存放在 `/data/djc/GeoPhy-Set3D/outputs/`；项目的 `outputs/checkpoints`、`outputs/logs` 和 `outputs/figures` 分别链接到对应目录。
`configs/example.yaml` 已配置数据集入口和外部输出路径。
实际数据、实验输出和本机符号链接不纳入 Git。

## 数据集生成

生效配置为 `configs/dataset_generation_v1.yaml`，执行入口为 `scripts/build_projection_pool.py`。
唯一数据源是既有静态子集的 `manifest.jsonl`；只读取 frame 60 / 2.0 s 的三维 GT，忽略原 `inputs/`。
派生数据写入 `/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/`，源数据保持只读。

```bash
python scripts/build_projection_pool.py --phase debug --resume
# 检查 debug/audits/figures/geometry_and_projection.png，并在 debug/debug_acceptance.json 记录视觉验收。
python scripts/build_projection_pool.py --phase mvp --resume
```

Debug 为 5 个训练样本、16/8 个站位、K=128；MVP 为全部准入样本、64/32 个站位、K=256。
每个站位包含 64×64 条独立 TDLAS 激光束；扫描半径约 6 m，水平 ±5°、垂直 ±4°。
原始投影保存为 float32 `ppm·m`；共享真实几何、valid-ray mask、64³ target、源 GT 引用和元数据一并保存。
在线视角组合、噪声和标定误差留给训练流程生成。

划分按 case / trajectory / 相同几何工况 / GT 内容重复项分组，再平衡可用工况字段。
保留的方法验证样本固定到训练集，投影器验证和 Debug 也仅使用训练集。
冻结后的配置、哈希、统计量和审计见数据集根目录；执行报告见 `docs/reports/dataset_generation_v1.md`。
`configs/example.yaml` 仍仅为早期配置结构示例。

## 训练数据接口与 64³ 投影审计

训练数据配置为 `configs/training_data_v1.yaml`。`ProjectionPoolDataset` 只从冻结的
`source_index.jsonl` 确定 membership，通过 mmap 读取选择的 station，默认不读取原始 128³ GT。
训练只允许 `split="train", geometry_pool="train"`（Pool A）；`val/test + train` 表示羽流泛化，
`val/test + ood` 表示独立几何评估。所有 split 共用训练集有效 ray 的 p99.5 归一化尺度。

```python
from torch.utils.data import DataLoader
import torch
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate

dataset = ProjectionPoolDataset(
    "/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1",
    split="train", geometry_pool="train", base_seed=20261011,
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                    collate_fn=projection_pool_collate,
                    generator=torch.Generator().manual_seed(20261011))
dataset.set_epoch(0)  # 每个 epoch 创建 iterator 前设置；支持 persistent_workers。
batch = next(iter(loader))
```

直接导入时需设置 `PYTHONPATH=src`。每个样本返回 3–6 个随机 input station 和 1 个不重叠的
held-out station，保留原始 `ppm·m`、固定尺度归一化投影、valid-ray mask、路径长度和冻结几何。
GT64 是 `[1,64,64,64]` XYZ 质量分数；`load_gt128=True` 才加载原始 GT。
Collate 按 batch 最大 view 数补零，返回 `view_mask` / `hold_view_mask`；station ID padding 为 -1。
`view_mask` 表示实际 view，`valid_ray` 表示实际射线穿过域，两者必须分别使用。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/validate_training_interface.py
python scripts/validate_frozen_generation.py
python scripts/audit_c64_projection_mismatch.py --samples 96 --split train
```

审计对全部 166 个训练 case 的相同 Pool A 计算 `P_diff(C128,96)` 和 `P_diff(C64,96)`，
与冻结的 `P_ref(C128,256)` 比较，正式指标仅使用 valid rays。默认采用 CPU / float64 / 8 线程。
输出在 `/data/djc/GeoPhy-Set3D/outputs/audits/`，项目 `outputs/audits` 是该目录的本机链接。
`validate_frozen_generation.py` 将原有几何、物理、收敛和完整性 QA 的结果写到外部，
适合检查冻结数据；原生成入口的 `--audit-only` 会刷新冻结文件，本轮不使用。
接口、复现规则、审计数值和测试结果见 `docs/reports/training_interface_v1.md`。

## GeoLift-Mean-Res3D / Supervised B0

站点采样只依赖 seed、epoch、sample index 和 case_id；num_workers=0/2/4 得到相同 station IDs
及顺序，支持 persistent workers。B0 使用共享 2D encoder、显式 XYZ→扫描角 lifting、
可见性 masked mean 与残差 3D decoder，输出 `[B,1,64,64,64]` CH4 质量分数。
训练仅使用 train/Pool A 的 voxel SmoothL1 + 0.1×relativeL1；固定四视图 val 选择 best checkpoint。

在已安装 torch/CUDA、NumPy、PyYAML、Matplotlib 的环境中，从仓库根目录执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/smoke_supervised_b0.py
python scripts/overfit_supervised_b0.py --cases 4 --epochs 300
python scripts/train_supervised_b0.py
python scripts/evaluate_supervised_b0.py
# 需要恢复训练时：
python scripts/train_supervised_b0.py --resume /data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b0/last.pt
```

完整训练入口要求过拟合 gate 通过；实验已存在时使用 `--resume` 或更改 experiment.name。
默认 FP32、batch_size=2、AdamW、cosine decay、最多150 epochs、val patience=20。
配置在 `configs/train_supervised_b0.yaml`；日志、权重和图片全部保存到外部
`/data/djc/GeoPhy-Set3D/outputs/{logs,checkpoints,figures}/`，不提交数据和权重。
评估入口仅运行 val/Pool A，包含 zero、train-mean、错配几何、跨 case measurement、zero measurement
和同步 view permutation。test 不参与模型选择。


本轮实测：150 epochs，best 为第148个 epoch；36-case val 的 mean per-case
MAE=0.0004593、RMSE=0.0023858、relative L1=0.13289、relative L2=0.15161。
wrong geometry / wrong measurement / zero measurement 的 relative L1 为0.87744 / 1.92355 / 1.00134，
正常输入明显更好；同步 view permutation 最大差4.92e−7。31项测试通过，冻结数据 inventory 未变。
结构、完整 per-case/pooled 指标、GPU显存、过拟合、完整训练和限制见
[Supervised B0报告](docs/reports/supervised_b0_v1.md)。

## B0 Generalization Evaluation Protocol v1

泛化画像保持 B0 网络、正式150-epoch配置和旧实验输出冻结，只在新的
`/data/djc/GeoPhy-Set3D/outputs/evaluations/supervised_b0_generalization_v1/` 保存结果。
协议在 `configs/evaluate_b0_generalization_v1.yaml`：先精确复现 Val-A-V4，再评估同一 acquisition
family 的未见 Pool B station poses、V=2/3/4/5/6/8 的五 deterministic draws、
V4/V6 的 spread/random/clustered、metadata 分层和 worst cases。所有 model evaluation 使用 val。

```bash
python scripts/evaluate_b0_generalization.py --stage reference
python scripts/train_b0_multiseed.py --seed 20261012
python scripts/train_b0_multiseed.py --seed 20261013
python scripts/run_b0_horizon_probe.py
python scripts/evaluate_b0_generalization.py --stage all
python scripts/summarize_b0_generalization.py
```

三个训练 seed 使用相同 reference val station IDs；原seed20261011直接引用已有checkpoint。
Horizon probe从头初始化，以cosine T_max=220诊断更长训练预算，不改写原checkpoint/scheduler。
新run支持`--resume`；不运行test、不训练Pool B、不实现B1。
主图导出PNG/PDF，同时保留CSV、逐case/draw JSONL和source/dataset/reference完整性记录。


本轮泛化画像已完成：reference精确复现，两个新增seed完成150 epochs，独立horizon diagnostic完成220 epochs。
三seed Val-A relative L1=0.13856±0.00563，Val-B=0.13155±0.00502；measurement/geometry shortcut结论跨seed复现。
220-epoch diagnostic relative L1=0.11634，比reference改善1.655个百分点，且best仍在220上限，
因此建议先确定公平的baseline训练预算，暂不进入B1。正式B0 reference未替换，38项测试通过，
source/dataset/model/checkpoint/旧实验完整性全部通过。
完整protocol、view count、coverage、分层、worst cases、multi-seed及horizon结论见
[B0 Generalization报告](docs/reports/supervised_b0_generalization_v1.md)。

## B0/B1 architecture comparison：固定300 epochs

本阶段按负责人指定冻结300-epoch协议，不再搜索训练horizon。
新增 `configs/train_supervised_b0_300.yaml` / `configs/train_supervised_b1_300.yaml`，
不覆盖B0-150 reference或220 diagnostic。两者均FP32、batch2、AdamW lr2e−4/weight_decay1e−4、
gradient clip1、cosine T_max300，禁用early stopping；train只用Pool A/V3–6/hold0，
固定Val-A-V4 minimum supervised loss选best。新配置默认GPU3，可在新的实验启动前修改device。

B1 `GeoLift-GCA-Res3D` 复用B0的encoder/lifting/decoder，仅替换fusion：
voxel级lifted feature与归一化α/β、scanner距离和station azimuth共同生成轻量view score，
在visibility AND view_mask上做安全masked softmax。支持 `model(batch, return_attention=True)`。
B0 306,625参数；B1 308,578参数，新增fusion仅1,953参数。

```bash
python scripts/compare_b0_b1_300.py --prepare
python scripts/train_supervised.py --config configs/train_supervised_b0_300.yaml --gate-config configs/train_supervised_b0.yaml
python scripts/overfit_supervised_b1.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b1_300.yaml
python scripts/compare_b0_b1_300.py --device cuda:3
# 恢复只允许同实验、同config的last checkpoint：
python scripts/train_supervised.py --config configs/train_supervised_b1_300.yaml --resume /data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b1_gca_300/last.pt
```

比较入口要求两个run均完成300 epochs，使用冻结best checkpoint，只评估Val-A/Val-B、
Pool A的V3/V4/V6各5个deterministic draws、clustered V4五draw、wrong-geometry/zero-measurement，
以及固定少量val案例的attention/entropy/permutation/padding诊断。
始终不做test模型评估，Pool B仅用于冻结checkpoint的评估。
输出在 `/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_vs_b1_300/`，
包含summary/CSV/effective protocol、逐case/draw结果、四张主图、完整性与test证据。
完整报告由完成后的比较入口生成：`docs/reports/supervised_b1_gca_v1.md`。
单seed结构比较不等于跨seed稳定性或统计显著性结论。

本轮两个run均完成300 epochs，best分别为第298/290 epoch。
B0/B1的Val-A relative L1=0.108330/0.103431，relative L2=0.122842/0.112474；
Val-B relative L1=0.104500/0.100378，relative L2=0.125769/0.111750。
B1主表L1改善0.489941个百分点（相对4.5227%），V3改善1.115847个百分点，clustered V4改善4.461874个百分点。
五项基本GO检查全部通过，但主表L1收益偏弱，单seed不足以支持跨seed稳定或显著优势；暂不建议直接进入B2。
46项测试全部通过，frozen dataset、历史结果、B0网络源文件和数值训练代码hash均未变化。
完整设计、主表、shortcut、attention与复现证据见 [B1 GCA报告](docs/reports/supervised_b1_gca_v1.md)。

## B2：Geometry-Conditioned Set Interaction

B2 `GeoLift-GCSI-Res3D` 在每个32³ voxel内加入一层view self-attention：
geometry embedding4→16→16，feature32+geometry16投影到token32，4 heads、FFN32→64→32、pre-norm residual。
interaction输出继续使用原B1 GCA pooling，encoder/lifting/decoder与supervised loss保持一致。
没有view-index positional embedding或pairwise geometry bias，不在voxel空间做attention。
B2共319,042参数，比B1增加10,464；B=2/V6 FP32前向反向已验证可在RTX4090运行。

只训练新的B2-300/seed20261011，复用被冻结的B0/B1 checkpoints与结果：

```bash
python scripts/compare_b0_b1_b2_300.py --prepare
python scripts/overfit_supervised_b2.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b2_300.yaml
python scripts/compare_b0_b1_b2_300.py --device cuda:3
```

继续使用固定300-epoch architecture-comparison protocol，不使用test，Pool B仅用于冻结best后的评估。
比较输出写到 `/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_b1_b2_300/`，
包含三模型主表、V3/V4/V6、clustered、spread V4、成对improvement表、interaction矩阵、四张图与完整性记录。
新报告在完成后的比较入口生成：`docs/reports/supervised_b2_gcsi_v1.md`。
B2判断优先关注V3与clustered收益；本轮不实现B3或追加seed/超参数搜索。

B2已完成300 epochs，best为第288轮，Val-A total supervised loss=0.009892290。
Val-A relative L1/L2=0.098907/0.106767，Val-B=0.096799/0.114124；
相对B1，Val-A L1改善4.3741%，V3从0.152994降至0.147769（改善3.4151%）。
clustered V4 L1仅从0.653769降至0.652042（改善0.2642%），Val-B L2反而恶化2.1247%。
冻结的五项L1工程GO检查通过，建议保留B2并进入B3；单seed结果不能证明已学会去冗余。
56项测试、trained permutation/padding、mask/all-invalid与受保护文件完整性检查全部通过。
完整主表、成对improvement、shortcut和interaction诊断见 [B2 GCSI报告](docs/reports/supervised_b2_gcsi_v1.md)。

## B3 前置 Gate：64³ Representation Capacity Audit

直接优化64³ concentration volume，检查naive GT64的projection mismatch是否能够通过
measurement-compatible representation消除；本阶段不训练网络或实现B3。
固定12个evenly-spaced train cases，共用16个spread Pool A fit stations及16个不重叠held-out stations。
目标为saved P_ref(C_GT128,K256)，复用未修改的P_diff/K96、frozen train p99.5归一化，
sigmoid volume参数化、Adam lr0.01、最多2000步，只按fit rays的SmoothL1优化和选择iterate。
held-out只在优化后评估，不参与loss、early stop或iterate selection；未使用test或Pool B。

```bash
python scripts/audit_64_representation_capacity.py --device cuda:3
# 已完成结果只重新生成报告，不重复优化：
python scripts/audit_64_representation_capacity.py --report-only
```

输出在 `/data/djc/GeoPhy-Set3D/outputs/audits/representation_capacity_64_v1/`：
protocol、selected cases/stations、operator/naive64/optimized64的raw valid-ray metrics、
optimized volumes、voxel plausibility diagnostics、一个zero-init sanity、两张图和完整性记录。
结果是有限优化得到的fitting result，不是exact theoretical representation floor。
工程Gate看fit/held-out projection errors；voxel similarity仅作diagnostic。

本轮12个case全部完成，工程分类为 **FAIL / NO-GO for strong 64³ projection-domain constraint**。
fit relative L1 mean/median/max=0.624358%/0.495559%/1.535897%；
held-out=1.577390%/1.040984%/4.552113%，naive held-out mean为1.664341%。
fit recovered_fraction mean/median=0.695069/0.693206；所有case的fit误差下降，但3个case的held-out恶化。
结论限定于本次有限优化，不证明64³存在理论capacity floor；本轮不进入强B3 projection constraint。
63项测试通过，dataset、B0/B1/B2 checkpoints、projector源码与旧实验未修改，未使用test或Pool B。
完整protocol、raw metrics、zero-init和plausibility诊断见 [64³ Capacity Audit报告](docs/reports/representation_capacity_64_v1.md)。

## OARI：冻结 Final architecture candidate v1

B3 measurement consistency及matched control仍为独立NO-GO结果，见 [B3报告](docs/reports/supervised_b3_measurement_consistency_v1.md)。
网络创新阶段已完成 **GeoLift-OARI-Res3D / OARI GO**：仅替换B2 set interaction，保留encoder、32³ lifting、GCA pooling、decoder与监督训练协议。
真实3D station-to-voxel ray relations进入per-head attention logits，5维local directional observability descriptor进入tokens和bias MLP。
总参数319,558，相对B2仅增加516（0.161734%）；一层d32/h4/FFN64，pairwise bias为8→16→4、SiLU、final layer zero-init。

一次4-train-case fixed-V4/120-epoch gate通过，正式训练从随机初始化跑满300 epochs，best为第278轮。
与冻结B2完全相同的5个deterministic draws中：clustered V4 L1从0.652042降至0.565045（改善8.699705个百分点、13.3422%）；
V3从0.147769降至0.136302（改善1.146646个百分点、7.7597%）。五个clustered draw均改善。
Val-A L1/L2=0.093694/0.102252，Val-B=0.090660/0.101354；二者均优于B2。
spread V4 L2略恶化；clustered绝对误差仍较高，单seed结果不证明解决ill-posedness或跨seed稳定性。
115项测试通过；trained permutation/padding最大差3.28e−7/2.09e−7；mask、all-invalid、两个新增分支的梯度与连接检查正常。
wrong relation只错配interaction内关系，L1从0.093694升至0.117097。
dataset、B0/B1/B2 checkpoints、B3 artifacts及projector保持不变；test未使用，Pool B evaluation-only。

```bash
python scripts/overfit_supervised_oari.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_oari_300.yaml
python scripts/compare_b2_oari_300.py --prepare
python scripts/compare_b2_oari_300.py --device cuda:3
```

已有gate禁止重复覆盖；正式训练仅允许使用自身last.pt与同config恢复。
结果、逐case/draw数据、geometry-stratified描述性分析、六个attention样例和五张图保存在
`/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_oari_300/`。
完整方法与边界见 [OARI报告](docs/reports/supervised_oari_v1.md)。本轮冻结OARI为Final architecture candidate v1，保留B2作为冻结比较基准。

## MRGR：冻结 Final architecture candidate v2

**GeoLift-OARI-MRGR-Res3D / MRGR GO** 已完成单次 measurement-residual-guided refinement。
共享 OARI 产生 C0，使用未修改的 K96 projector 计算 raw ppm·m signed residual，固定除以
32284.894580078173，再经 16-channel residual encoder、同一个 GeometryLifting 和 zero-init adapter
修正 context；第二次调用同一个 GCA 与 decoder 产生 C1。训练仅监督 C1，没有新增 projection loss。
总参数 **332,070**，较 OARI 增加 12,512（3.915%）。

一次冻结 OARI、4-case fixed-V4、60-epoch gate 通过：C0 L1 保持 0.036839，C1 降至 0.029469。
正式模型重新随机初始化，按相同 seed/protocol 跑满 300 epochs，best 为第 295 轮。
以下为 mean per-case relative L1；V3/random V4/V6/clustered 为与 OARI 完全配对的 5 draws：

| 条件 | 冻结 OARI | MRGR C0 | MRGR C1 |
|---|---:|---:|---:|
| Val-A fixed V4 | 0.093694 | 0.111228 | **0.067980** |
| Val-B fixed V4 | 0.090660 | 0.106385 | **0.060948** |
| V3 | 0.136302 | 0.159126 | **0.098309** |
| random V4 | 0.097139 | 0.111392 | **0.066923** |
| V6 | 0.080065 | 0.094160 | **0.053401** |
| clustered V4 | 0.565045 | 0.804391 | **0.495210** |
| spread V4 draw0 | 0.089940 | 0.106313 | **0.058919** |

MRGR 的 C0 是联合训练中只由 C1 loss 间接监督的中间输出；其误差高于冻结 OARI，
C0→C1 改善与 residual shortcut 提供 refinement 贡献证据。
Val-A normal/zero/wrong-pairing/signflip L1 为 0.067980/0.109388/0.106254/0.180266。
Val-A observed reprojection L1 从 0.067022 降至 0.032218；held-out 从 0.071169 降至 0.040345。
134 项测试通过，trained FP32 permutation/padding 最大差为 2.98e−7/2.38e−7。
峰值 allocated memory 4.342 GiB，平均 epoch 16.277 s（非受控 benchmark）。
所有旧 OARI/B0/B1/B2/B3 产物、dataset、projector 保持不变；Pool B 仅在训练后评估，test 未使用。
单 seed 结果不证明跨 seed 稳定性或 inverse-problem uniqueness；clustered 绝对误差仍高。

```bash
python scripts/overfit_supervised_mrgr.py
python scripts/train_supervised_mrgr.py --config configs/train_supervised_oari_mrgr_300.yaml
python scripts/compare_oari_mrgr_300.py
```

已有 gate 禁止重复覆盖；正式训练仅允许使用自身 last.pt 与同 config 恢复。
完整方法、20 节报告见 [MRGR 报告](docs/reports/supervised_oari_mrgr_v1.md)。
所有 CSV/JSON、42 项终端汇报、六组 diagnostics、五张图及 v2 冻结记录在
`/data/djc/GeoPhy-Set3D/outputs/comparisons/oari_vs_mrgr_300/`。
OARI v1 继续作为冻结基准保留；本轮不增加第三个模块。

## Matched residual control 与 GT-free MRGR v1

Stage A 已完成唯一一次 from-scratch、300-epoch matched control：与 full MRGR 同为
332,070 参数，同 seed 初始 tensors、sampler 和训练预算；refinement 输入为
`[0, valid_ray, normalized_path]`，训练不调用 projector。control/full MRGR 的 Val-A、V3、
clustered V4 relative L1 分别为 0.094630/0.067980、0.137334/0.098309、0.583203/0.495210。
该单 seed 配对结果支持 signed measurement residual 提供额外信息；监督结构实验已关闭。
完整结果见 [matched control 报告](docs/reports/mrgr_residual_control_v1.md)。

Stage B 完成 **Self-supervised training without volumetric ground truth / SelfSup-MRGR v1 GO**。
复用冻结 MRGR，从随机初始化训练；只读 TDLAS projections 和 acquisition geometry。
train A3–5/B3–5/H2、validation A4/B4/H4 均两两不相交，H 不输入网络。
固定 loss 为 `L_hold + 0.5 L_cross + 0.1 L_obs + 0.05 L_cons`，K96/all rays，
无 TV、sparsity 或新增模块。唯一 8-case/80-epoch gate 的 held-out L1 为 0.520501，
优于 zero-volume 基线 1.0，输出未塌缩。

正式模型重新随机初始化，166 train/36 Val-A、FP32/effective batch2 跑满 300 epochs；
best 第 278 轮，仅由 held-out measurement L1 选择。
训练/验证 GT 打开尝试、外部 checkpoint 打开尝试均为 **0**；只读取 202 个 Pool A projection 文件。
Val-A held-out measurement L1/L2=0.032752/0.042441，冻结后 Pool B=0.033222/0.043871。
best checkpoint SHA256：`1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026`。
只有完成 300 epochs 并冻结 hash 后，独立 post-hoc script 才读取 GT，方法与 checkpoint 随后保持冻结。

以下 voxel relative L1 **仅用于 post-training evaluation**，不用于训练或选模；七种条件的
case/station IDs/draws 与冻结监督模型完全配对。

| 条件 | Supervised MRGR | SelfSup MRGR |
|---|---:|---:|
| Val-A fixed V4 | 0.067980 | 0.096193 |
| Val-B fixed V4 | 0.060948 | 0.090946 |
| V3 | 0.098309 | 0.118364 |
| random V4 | 0.066923 | 0.095649 |
| V6 | 0.053401 | 0.088058 |
| clustered V4 | 0.495210 | 0.393402 |
| spread V4 draw0 | 0.058919 | 0.087648 |

GO 表示达到预先声明的工程可行性条件。Val-A/Val-B 相比监督模型仍有 41.50%/49.22% 的 L1 误差增幅；
clustered V4 误差降低 20.56%，但单 seed 不证明跨 seed 稳定性、监督性能持平或三维解唯一。
158 项历史与新增测试通过；旧 OARI/MRGR/B0–B3、dataset、GT、projector 均保持不变，test 未使用。
完整 20 节方法、L1/L2/MAE/RMSE、测量泛化与边界见 [SelfSup 报告](docs/reports/selfsup_mrgr_v1.md)。

```bash
python scripts/train_supervised_mrgr_zero_control.py
python scripts/compare_mrgr_residual_control.py
python scripts/train_selfsup_mrgr.py --smoke
python scripts/train_selfsup_mrgr.py --gate
python scripts/train_selfsup_mrgr.py
python scripts/evaluate_selfsup_with_gt.py
```

以上运行已完成，禁止重复 gate 或覆盖正式实验；正式恢复仅允许自身 `last.pt`。
数值源协议 commit 为 `3f76c3f`。全部逐 case/draw、read audit、完整性检查与 1–39 项终端汇报分别保存在
`/data/djc/GeoPhy-Set3D/outputs/comparisons/mrgr_residual_control_v1/` 和
`/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1/`。

### SelfSup-MRGR v2: rich-view teacher consistency

MRGR remains frozen (332,070 parameters). v2 changes only view construction and the
training objective: the same network reconstructs disjoint sparse A/B and their rich
union T; sparse volumes match detached T while all three retain measurement losses.
Use the independent frozen config; no v1/supervised warm start, EMA, prior or sweep.

```bash
python scripts/train_selfsup_mrgr_v2.py --prepare
python scripts/train_selfsup_mrgr_v2.py --smoke
python scripts/train_selfsup_mrgr_v2.py --gate
python scripts/train_selfsup_mrgr_v2.py
python scripts/evaluate_selfsup_v2_with_gt.py
```

Freeze `training_source_manifest.json` in the v2 comparison directory before the
single gate. Run formal training only after the gate passes. GT evaluation verifies
full 300-epoch completion, sparse measurement-only checkpoint selection and the
frozen SHA256; Val-A/Pool-B measurement evaluation precedes any voxel evaluation.
A failed gate stops the experiment without GT, tuning or another gate. Results and
all 48 terminal checklist fields are recorded in
`/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v2/` and
`docs/reports/selfsup_mrgr_v2.md`.

The v2 run stopped at the engineering gate (NO-GO): student–teacher discrepancy
did not decrease under the predeclared first10/last10 criterion. Formal training
and GT post-hoc evaluation were not executed. See
[the v2 report](docs/reports/selfsup_mrgr_v2.md). To reproduce the read-only failure
diagnostics and artifact report, run `python scripts/report_selfsup_v2_gate_failure.py`;
this loads only the existing v2 gate checkpoint and does not retrain.

The [v2 result snapshot](docs/reports/artifacts/selfsup_mrgr_v2/README.md) contains the frozen gate metrics, read audits, failure diagnostics,
protocol, test log and complete terminal checklist for GitHub review.

[中文 RVTC 失败诊断](docs/reports/selfsup_mrgr_v2_failure_diagnosis.md)区分原工程判据、独立测量教师质量和共享参数梯度风险。
本轮只做无GT、无更新诊断，保留原v2 NO-GO；关/开一致性对照仅为待登记草案。

[中文 RVTC 严格匹配有限对照](docs/reports/selfsup_rvtc_matched_80.md)报告随后获授权的 Control/Treatment 各80轮、320步实验。

[冻结 clustered V4 单站位诊断](docs/reports/clustered_v4_one_swap.md)使用原36例Val与5个draw，保持V=4，配对比较Original、几何互补Targeted和均匀Random替换。SelfSup v1三维L1的Targeted−Original为−0.250984（95% case区间[−0.292007,−0.213704]），Targeted−Random为−0.028574（[−0.042523,−0.015141]）；监督MRGR三维L1方向一致。仅冻结推理，未训练；不证明唯一恢复、跨seed稳定性或原clustered输入能力提升。原v2 NO-GO保留。
相同富视角测量训练下，原0.1体积一致性使 Val-A held-out L1 增加0.031395（配对95%CI：0.016833–0.046475）；保留v1及原v2 NO-GO，不自动启动后续实验。
