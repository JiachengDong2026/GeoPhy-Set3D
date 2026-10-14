# DeepSets 自监督同条件对照 v1：训练前协议

本轮建立最小统一比较接口，并训练一次 GeoLift-DeepSets-Sum。原 frozen SelfSup-MRGR v1 作为历史参考，保留原 GO/NO-GO 与 C0/C1 诊断；不重训、不调参、不实现 Senseiver。

## 模型与原论文适配

Zaheer 等《Deep Sets》(2017), https://arxiv.org/abs/1703.06114，采用共享元素映射、求和与后置映射的置换不变形式。本文适配在每个 lifting 网格点对站位求和：`F(x)=sum_i m_i(x)*phi(Y_i,g_i;x)/4`，再经原 ResidualDecoder3D 输出 sigmoid 64³质量分数。φ复用原 SharedEncoder2D 和 GeometryLifting；ρ是空间卷积 decoder。保留测量图、valid mask、path、真实几何和方向；不学习station编号embedding，不除以实际V，不含OARI/GCA/MRGR。这是任务适配基础参考，并非原论文三维重建官方实现；不将原论文表示定理直接解释为本有限架构的三维唯一恢复保证。

真实view与局部visibility共同排除padding；invalid rays的测量/path安全置零；全不可见聚合为0，不约束decoder输出0。支持可变V和同步置换。独立模型及校验入口，不解除原v1保护，不改变历史B0。

## 训练与工程检查

166 train / 36 Val，Pool A；自然双输入A3–5/B3–5/H2两两不交；固定Val A4/B4/H4，原sampler/IDs/归一化。H只参与目标投影。完整复用 `training.selfsup.objective/project_input/run_epoch/selection_metric`，包括SmoothL1与relative项、全部归约和epsilon：`L_hold+0.5 L_cross+0.1 L_obs+0.05 L_cons`。P_diff/K96/all valid rays。

FP32、seed20261011、effective batch2、AdamW lr2e−4/wd1e−4、clip1、cosine T_max300；完整300 epochs，每轮166 cases/83 steps，总24,900。无early stop、augmentation、新loss、结构或超参数搜索。只由原自然Val两预测平均held-out relative L1选best。共享GPU可换GPU；OOM可独立登记batch1/accum2，保持24,900步。只能从本run的last.pt恢复完整model/optimizer/scheduler/RNG。

正式前独立临时模型进行最大A5/B5/H2真实forward/backward/step、有限梯度、shape/range、置换/padding/全invalid、H不入model、A/B/H不交、GT与foreign checkpoint保护检查。正式训练重新seed初始化，不复用smoke权重。保存初态hash、实际有序case/batch/station身份和digest、更新次数、环境、GTReadGuard审计与恢复记录。历史原v1缺少逐步参数/optimizer记录：可验证同协议/已有digest，不声称严格随机化配对或逐步数值匹配。

## 评价、统计与成本

训练完成300与best SHA256冻结后，独立measurement-only评价并保存全部预测，再独立进程读对应Val GT64。禁GT128、Pool B、final test。主清单完整复用最新已验证的900行原五档V4：36 case×25 layouts，input IDs及顺序、共同H4、mask、测量和scale相同。Ours复用已验证正常C1十指标，不重复900前向；校验输入、checkpoint与来源hash。辅助自然V3/V4/V6本轮不纳入，避免额外身份与预算扩展。

十指标：3D relative L1/L2、MAE、RMSE、积分proxy相对误差、峰值绝对误差、observed与held-out L1/L2。先每case/group均值五布局，再以36case配对。Δ=Ours−DeepSets；bootstrap seed2026101801、10,000次case重采样、percentile95%CI。主终点五组三维L1；其余辅助，全报；不计算p值/不作多重检验显著性断言，CI逐项未校正。持平阈值1e−8。报告均值/median/CI/胜负平、case均值P90/P95/max/worst ID、G1–G4宏平均、G5绝对误差、G5−G1及总体组均值之比G5/G1。900行/voxel/ray不是独立样本。

参数量、FP32参数字节与checkpoint字节、训练墙钟及峰值显存分开报。推理采用同GPU/输入/batch1/FP32，固定case0/18/35、draw0、G1/G3/G5共9组输入，每输入warmup10+同步重复30；Ours包含内部projector。共享GPU时序交替，报告median/P90、当前GPU占用，不称独占benchmark。计时不读GT。

## 图示、解释与交付

固定主要图：五档误差曲线、配对差CI、训练measurement曲线、固定case0/18/35×draw0×G1/G3/G5的GT/Ours/DeepSets对比。y为高度，y32=32×0.6/63=0.30476m；浓度共享每case显示字段最大值色标，绝对误差另设统一色标；不按误差挑案例/切片。

允许DeepSets优于Ours。完整报告未塌缩、预算、信息/标签/selector、每组效果、L2/measurement/峰值/积分一致性、成本及末期是否继续改善，不追加训练。单seed、共享encoder/lifting/decoder基础对照，同时改变interaction/fusion/refinement，不能归因单个模块，也不证明超过Senseiver/DeepONet/GNOT。完成后只给一个优先后续任务，不自动执行。

外部目录 `/data/djc/GeoPhy-Set3D/outputs/comparisons/deepsets_selfsup_v1`；新checkpoint/logs用同独立run名。Git仅代码/config/协议/标量/图/审计；大预测、GT与checkpoint不提交。训练前注册hash/清单与历史metadata见 `docs/reports/artifacts/deepsets_selfsup_v1/protocol.json`。旧源码只允许模型注册新增分支和最终README导航，数值历史源码不变。按协议、实现、结果顺序提交，通过7897代理同步GitHub。
