# OARI 四项定向实验：普通集合交互、关系输入第二种子与冻结token复核

本报告比较同一已使用35测试案例上的探索性结果，不是独立测试确认。所有完整对照与训练变体都采用 **Uniform Fusion**；正式模型、历史checkpoint、原预测与历史协议均未改写。

## 结论与论文建议

普通交互与OARI的平均L1优势未在两种子、两布局中形成统一强证据；不能据此声称OARI专门结构具有稳定独立增益。若普通交互成本更低，可列为简化候选；尚不足以直接宣布其等效或全面替代OARI，本阶段保留正式模型身份。

辅助终点核对：relative_l2的四个配对比较中，均值方向支持OARI/普通交互/数值持平为3/1/0；absolute_peak_ratio_error的四个配对比较中，均值方向支持OARI/普通交互/数值持平为3/1/0；centroid_error_mm的四个配对比较中，均值方向支持OARI/普通交互/数值持平为3/1/0；Dice50的四个配对比较中，均值方向支持OARI/普通交互/数值持平为0/4/0。这些方向计数只是四个已登记比较的描述，不是四个独立样本，也不替代各自区间。
种子20261011的普通交互/OARI平均耗时比0.760，峰值分配显存比0.234；种子20261019的普通交互/OARI平均耗时比0.821，峰值分配显存比0.234。是否优先简化须同时考虑上面的L1、辅助终点与成本；本轮没有预登记等效界限，不据单项优势宣称全面替代。

普通集合交互仍接收原4维几何描述符及正确GeometryLifting，因此它检验OARI专门设计的增益，不是去掉全部几何信息的对照。冻结token实验检验已训练模型的当前依赖，不能直接等同重新训练后的结构贡献。

## 实现差异与工程检查

独立变体uniform_plain_set_interaction仅替换C0前交互：GeometryConditionedSetInteraction接收features、geometry、visibility和view_mask，跳过OARI射线关系、局部描述符、可观测性token与pairwise bias。原Encoder、GeometryLifting、几何描述符、多尺度decoder、Uniform Fusion及Physics Residual Feedback保留。原MRGR.forward通过C3 MRO调用新的PlainSetStage0，反馈代码逐字未改。
从官方工厂构造同种子新鲜模型后，形状兼容的交互子层保留其新鲜初始化；48→32 token_projection独立初始化，不加载56→32不兼容参数，构造恢复RNG。复制不涉及任何训练checkpoint。共享前置模块与解码/反馈的初始化逐张量一致。训练后共享模块自然可能学出不同权重，未要求其训练后输出相同。
普通交互共342209参数，对照342725，减少516（约0.151%）。这是完整交互设计替换，参数量并未严格匹配，不能把所有差值归为某一几何通道的唯一因果贡献。no_ray_relations仍为342725参数，完全沿用首仿射输入前三关系通道置零定义，保留描述符与learned bias。
CPU门槛在两个种子验证共享初始化、第一次GeometryLifting和几何描述符逐元素相同；mask中的NaN隔离、padding不变、排列等变、全无效交互输出为零、完整模型有限/[0,1]、接口与梯度通过。普通attention没有伪造OARI bias。GPU smoke使用独立新鲜模型与单个正式双输入batch，完整损失反传和两次优化通过；其状态没有用于正式训练。

## 训练、数据与调度

三项训练均完整300轮、24900次提交更新，166训练/36验证案例，batch2、AdamW lr2e-4/wd1e-4、cosine300、clip1、FP32、TF32关闭；完全复用原正式训练引擎和损失系数。资源中断时未提交轮的部分计算不保留，按既定epoch-commit机制重放，因此24900指最终轨迹的提交更新数。仅以完整训练期间验证投影H4严格最小值选轮，不按GT选checkpoint，不提前按结果停轮。
启动前GPU1/2被其他用户使用，只在通过UUID锁、间隔15秒双快照、进程/显存/利用率/主机检查的卡上启动任务。沿用初始2并发、首两训练各≥3轮且主机资源充足后最多4并发规则；按每项单GPU规则调度，锁FD由worker继承。GPU1折间短暂无CUDA上下文，第三训练取得锁并启动后，另一用户已启动但延迟初始化的下一折任务建立CUDA上下文，发生短暂共卡。检测后仅中断/清理本方第三训练及其数据worker，保留完整第4轮last.pt与全部日志；排除GPU1/2的外部折间空档，待原两训练完成后在GPU0/3按原引擎恢复相同随机数、优化器和调度器状态，继续至完整300轮，没有终止其他用户进程。实际时间线、进程/锁、恢复校验见recovery及scheduling；中断轮重放属于既定epoch-commit恢复，未选择性丢弃已提交轮。
三训练模型各普通draw0 35份＋聚簇draw0..4 175份，共630新增预测。seed20261019冻结obs_token仅两布局draw0共70新增预测，另有两个固定工程no-op前向。复用完整两种子、no-ray种子11各210份，以及此前obs_token种子11的70份，共700历史预测。共1400不同预测文件、2800 C0/C1字段。
新冻结模型为正式uniform_fusion_seed20261019，严格验证checkpoint哈希与官方工厂加载；直接载入原OARI v2 Intervention实现，移除描述符首仿射的五维输入贡献，保留learned bias/SiLU，未训练。先搜索可核验的seed19输出，缺失才补算；两布局首案例no-op与历史C0/C1在登记容差内一致。所有输入、参数和checkpoint哈希保持不变，始终eval/无梯度。
本轮用户明确授权在既有35案例范围评估本阶段结果。三项训练完成、选定checkpoint冻结且全部新增预测统一封存后，才首次读取本阶段GT64；只加载35份GT64，其SHA256与既有记录一致，GT128始终禁止。

| 模型 | epoch | 参数 | SHA256 |
|---|---:|---:|---|
| uniform_fusion_seed20261011 | 285 | 342725 | `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc` |
| uniform_fusion_seed20261019 | 286 | 342725 | `19bb5f698349c976358be782e9d43301afa9e1eec4a89ddb4e1ceeb8d9bd1424` |
| uniform_no_ray_relations_seed20261011 | 283 | 342725 | `f6afaa13bdcb851ee5e7ebee6086116be482157a3a12546c1164a22ff6e207f4` |
| uniform_no_ray_relations_seed20261019 | 274 | 342725 | `6b8d4f1a19edd9ff17ac68a758cdb17eda5cd52a421571e3c2fd4611a9f6de06` |
| uniform_plain_set_interaction_seed20261011 | 298 | 342209 | `0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c` |
| uniform_plain_set_interaction_seed20261019 | 279 | 342209 | `fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d` |

## 统计与指标

差值=处理误差−同种子完整模型误差，正值表示误差增加；绝对比例差×100即百分点。两种子分别报告，每个区间统计单位均35案例。聚簇正式训练比较先在每案例内平均5个draw，再进行20000次配对bootstrap，固定seed20261026，百分位95%区间估计平均差；draw0单独分析，不与五draw平均混用。区间未经多重校正，均为探索性案例区间，不能解释为训练种子总体不确定性。SD使用ddof1；|Δ|≤1e-8仅表示数值基本不变，不是性能等效界限。
直接使用原正式评估的volume_metrics/safe_shape，原始[0,1]质量分数、XYZ端点网格0..0.6m、float64，无每案例重新归一化。相对L1分母max(Σabs(GT),1e-6)，相对L2平方分母max(ΣGT²,1e-12)再开方；峰幅比=max(pred)/max(GT)，峰比例绝对误差=abs(比值−1)，峰值绝对误差为质量分数差。峰位置距离mm、并列峰采用首个XYZ节点。Dice50双方共用0.5×GTmax阈值。保留梯形积分质心centroid_error_mm与普通节点质心centroid_distance_mm，两者不混用。Dice差正值表示改善，峰幅比本身无单调优劣方向。

## 普通集合交互与OARI：正式训练配对

| 种子 | 布局 | draw | OARI C1 L1(%) | 普通交互(%) | ΔL1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 |
|---|---|---|---:|---:|---:|---|---:|---|
| 20261011 | Val-A-V4 | 0 | 8.5928 | 8.4930 | -0.0998 | [-0.2655, +0.0780] | -0.1760 | 16/19/0 |
| 20261011 | clustered-V4 | 0,1,2,3,4 | 28.0246 | 29.8537 | +1.8291 | [+0.9036, +2.6961] | +2.2516 | 25/10/0 |
| 20261019 | Val-A-V4 | 0 | 8.5527 | 8.7764 | +0.2237 | [+0.0040, +0.4442] | +0.0974 | 23/12/0 |
| 20261019 | clustered-V4 | 0,1,2,3,4 | 29.9165 | 29.7780 | -0.1384 | [-1.2925, +1.0298] | +0.5452 | 21/14/0 |

Val-A-V4：替换OARI后，两种子C1 L1分别降低0.0998和增加0.2237个百分点；两个种子没有形成一致的均值变化方向，证据强度以各自案例区间为限。
clustered-V4：替换OARI后，两种子C1 L1分别增加1.8291和降低0.1384个百分点；两个种子没有形成一致的均值变化方向，证据强度以各自案例区间为限。

普通交互替换后的方向是否一致，须分别看上表四个对比；不能以其中最有利一行替代全局判断。下方同时列出C0与所有辅助指标，并与成本一起评估。

## 射线关系输入：第二种子是否复现

| 种子 | 布局 | draw | no-ray−完整C1 L1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 |
|---|---|---|---:|---|---:|---|
| 20261011 | Val-A-V4 | 0 | +0.1467 | [-0.0883, +0.4505] | -0.0030 | 17/18/0 |
| 20261011 | clustered-V4 | 0,1,2,3,4 | +2.6472 | [+1.6804, +3.6888] | +2.8380 | 29/6/0 |
| 20261019 | Val-A-V4 | 0 | -0.0401 | [-0.2377, +0.1447] | +0.0647 | 19/16/0 |
| 20261019 | clustered-V4 | 0,1,2,3,4 | +1.0652 | [+0.1761, +2.0664] | +0.3842 | 22/13/0 |

聚簇五draw平均下，两个种子的no-ray差值方向分别为增加/增加。两个种子的均值均支持保留射线关系输入；其收益的描述性方向在第二种子复现，但证据强度仍以各自案例区间为限。两个固定种子不足以确认训练种子总体的稳定性。

## 冻结obs_token：两个种子仅draw0对齐

| 种子 | 布局 | ΔC1 L1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 | C1 D均值(%) |
|---|---|---:|---|---:|---|---:|
| 20261011 | Val-A-V4 | +0.0217 | [-0.0286, +0.0705] | +0.0242 | 19/16/0 | 1.0058 |
| 20261011 | clustered-V4 | +1.8151 | [+1.2701, +2.3631] | +2.1723 | 30/5/0 | 9.1960 |
| 20261019 | Val-A-V4 | +0.4358 | [+0.2990, +0.5895] | +0.3711 | 31/4/0 | 2.6257 |
| 20261019 | clustered-V4 | +2.3129 | [+1.0647, +3.5855] | +2.0217 | 28/7/0 | 11.7946 |

第二种子聚簇C1 L1的平均差与第一种子同为正，且其95%案例区间也在零以上，支持在这两个冻结模型中复现有益token依赖。
冻结obs_token仅屏蔽token首仿射输入贡献，保留pairwise描述符路径和全部训练权重；普通集合交互则重新训练且同时移除了专门关系/描述符结构。两者不能用作完全相同的干预解释。即使冻结恶化而普通交互接近，也只能与补偿解释相容，不能排除训练随机性、优化变化或冻结干预的分布偏移。

## 固定硬件推理成本

same exclusiveGPU, validation cases0/17/34 fixedV4, one warmup each +3 measured forwards, batch1 FP32/TF32off return_stagesTrue

| 模型 | 参数 | 平均前向(ms) | 中位数(ms) | SD(ms) | 峰值分配显存(MiB) |
|---|---:|---:|---:|---:|---:|
| uniform_fusion_seed20261011 | 342725 | 25.399 | 25.312 | 0.219 | 1103.40 |
| uniform_fusion_seed20261019 | 342725 | 23.524 | 24.121 | 1.007 | 1103.15 |
| uniform_no_ray_relations_seed20261011 | 342725 | 22.433 | 22.410 | 0.203 | 1103.15 |
| uniform_no_ray_relations_seed20261019 | 342725 | 22.236 | 22.167 | 0.202 | 1103.15 |
| uniform_plain_set_interaction_seed20261011 | 342209 | 19.302 | 19.300 | 0.112 | 258.30 |
| uniform_plain_set_interaction_seed20261019 | 342209 | 19.316 | 19.381 | 0.161 | 258.30 |

九次计时来自同一GPU上的三个固定验证输入，不视作独立模型性能样本；GPU UUID与逐次计时保存在benchmark.json。普通交互跳过特征值分解和关系构造，实际节省以上表为准；只有成本降低并且误差、峰值、位置、Dice及两个种子都可接受，才有依据优先选择更简单模型。没有预登记等效容差，因此不以“不显著”作为等效证据。

## 全部C0/C1辅助终点（差值原单位）

| 模型 | 布局/draw | 阶段 | 指标 | 完整均值 | 处理均值 | 平均差 | 中位数差 | SD差 | 95%区间 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | relative_l1 | 0.113762 | 0.110976 | -0.002786 | -0.001376 | 0.009713 | [-0.0062, +0.0002] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | relative_l2 | 0.166994 | 0.167967 | +0.000973 | +0.003496 | 0.017668 | [-0.0054, +0.0060] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | peak_ratio | 1.166835 | 1.166926 | +0.000091 | +0.009938 | 0.082929 | [-0.0269, +0.0273] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.188009 | 0.206121 | +0.018111 | +0.021591 | 0.076146 | [-0.0071, +0.0429] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | peak_absolute_error | 0.112581 | 0.125116 | +0.012535 | +0.012800 | 0.043654 | [-0.0020, +0.0267] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.385500 | 73.790330 | +4.404830 | +0.000000 | 99.085846 | [-27.9266, +37.0930] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | centroid_error_mm | 3.279464 | 2.850438 | -0.429027 | -0.230253 | 1.665685 | [-0.9829, +0.1010] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.451457 | 2.992252 | -0.459205 | -0.240507 | 1.724729 | [-1.0341, +0.0903] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C0 | Dice50 | 0.821413 | 0.828164 | +0.006751 | -0.004979 | 0.046274 | [-0.0077, +0.0223] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | relative_l1 | 0.085928 | 0.084930 | -0.000998 | -0.001760 | 0.005253 | [-0.0027, +0.0008] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | relative_l2 | 0.157314 | 0.157424 | +0.000110 | -0.001508 | 0.007633 | [-0.0023, +0.0027] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | peak_ratio | 1.173953 | 1.164070 | -0.009883 | -0.001511 | 0.073711 | [-0.0340, +0.0140] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.201320 | 0.207250 | +0.005930 | +0.016588 | 0.074146 | [-0.0186, +0.0299] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | peak_absolute_error | 0.120536 | 0.126209 | +0.005673 | +0.009797 | 0.042850 | [-0.0084, +0.0196] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | peak_location_error_mm | 67.568955 | 86.509828 | +18.940872 | +0.000000 | 81.806516 | [-6.2404, +47.0142] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | centroid_error_mm | 1.429295 | 1.532961 | +0.103666 | +0.114830 | 0.485585 | [-0.0461, +0.2731] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.617395 | 1.716616 | +0.099222 | +0.085166 | 0.517016 | [-0.0583, +0.2819] |
| uniform_plain_set_interaction_seed20261011 | Val-A-V4/0 | C1 | Dice50 | 0.820338 | 0.826584 | +0.006246 | +0.000000 | 0.022903 | [-0.0011, +0.0138] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | relative_l1 | 0.396765 | 0.465608 | +0.068843 | +0.059004 | 0.067195 | [+0.0468, +0.0905] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | relative_l2 | 0.568319 | 0.588168 | +0.019849 | +0.028196 | 0.055981 | [+0.0015, +0.0380] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | peak_ratio | 0.865758 | 0.910494 | +0.044736 | +0.049745 | 0.209528 | [-0.0203, +0.1146] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.197897 | 0.165291 | -0.032605 | -0.029805 | 0.168445 | [-0.0891, +0.0208] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | peak_absolute_error | 0.117784 | 0.100208 | -0.017576 | -0.020335 | 0.096618 | [-0.0495, +0.0136] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | peak_location_error_mm | 121.683882 | 108.650435 | -13.033447 | -4.460201 | 82.324182 | [-40.1902, +13.1177] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | centroid_error_mm | 16.346618 | 17.261192 | +0.914574 | -2.770193 | 12.968980 | [-3.2778, +5.1677] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | centroid_distance_mm | 16.569914 | 17.300577 | +0.730663 | -2.776107 | 12.966223 | [-3.4603, +4.9925] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C0 | Dice50 | 0.294281 | 0.411485 | +0.117204 | +0.093590 | 0.133079 | [+0.0770, +0.1632] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | relative_l1 | 0.328316 | 0.359553 | +0.031236 | +0.056438 | 0.062312 | [+0.0102, +0.0509] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | relative_l2 | 0.500036 | 0.511667 | +0.011632 | +0.012666 | 0.062238 | [-0.0087, +0.0315] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | peak_ratio | 0.806027 | 0.866005 | +0.059978 | +0.059829 | 0.189116 | [-0.0018, +0.1218] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.234753 | 0.203910 | -0.030843 | +0.005088 | 0.189591 | [-0.0946, +0.0298] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | peak_absolute_error | 0.139727 | 0.123114 | -0.016613 | +0.003471 | 0.105444 | [-0.0517, +0.0174] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | peak_location_error_mm | 111.175341 | 131.881571 | +20.706230 | -1.577507 | 127.772966 | [-22.1799, +61.9129] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | centroid_error_mm | 12.645216 | 13.032334 | +0.387118 | -0.231802 | 8.129810 | [-2.1999, +3.1266] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | centroid_distance_mm | 12.988187 | 13.195310 | +0.207123 | -0.496086 | 8.188873 | [-2.3934, +2.9697] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0 | C1 | Dice50 | 0.316040 | 0.419609 | +0.103569 | +0.041626 | 0.178327 | [+0.0472, +0.1635] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | relative_l1 | 0.330008 | 0.361955 | +0.031946 | +0.046433 | 0.038497 | [+0.0191, +0.0443] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | relative_l2 | 0.486151 | 0.525842 | +0.039691 | +0.047908 | 0.048281 | [+0.0240, +0.0553] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_ratio | 0.951225 | 0.985237 | +0.034012 | +0.031548 | 0.098565 | [+0.0017, +0.0665] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | absolute_peak_ratio_error | 0.176864 | 0.180675 | +0.003811 | +0.000700 | 0.074283 | [-0.0199, +0.0290] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_absolute_error | 0.106410 | 0.111124 | +0.004714 | +0.000414 | 0.046292 | [-0.0097, +0.0207] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_location_error_mm | 104.865243 | 107.771564 | +2.906321 | +0.585691 | 57.425352 | [-15.6527, +21.4973] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | centroid_error_mm | 11.064913 | 14.090023 | +3.025111 | +3.526873 | 8.435709 | [+0.1581, +5.6681] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | centroid_distance_mm | 11.123938 | 14.043765 | +2.919826 | +3.358190 | 8.421065 | [+0.0406, +5.5364] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | Dice50 | 0.427069 | 0.408350 | -0.018719 | -0.012472 | 0.102208 | [-0.0516, +0.0148] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | relative_l1 | 0.280246 | 0.298537 | +0.018291 | +0.022516 | 0.027267 | [+0.0090, +0.0270] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | relative_l2 | 0.450057 | 0.466281 | +0.016224 | +0.015643 | 0.041610 | [+0.0030, +0.0299] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_ratio | 0.925057 | 0.928311 | +0.003254 | -0.006140 | 0.079032 | [-0.0220, +0.0297] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | absolute_peak_ratio_error | 0.175157 | 0.181195 | +0.006038 | +0.016243 | 0.073231 | [-0.0187, +0.0293] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_absolute_error | 0.106419 | 0.112638 | +0.006219 | +0.009711 | 0.043055 | [-0.0081, +0.0201] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_location_error_mm | 97.788347 | 122.244949 | +24.456602 | +11.290667 | 67.715898 | [+1.7275, +46.3642] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | centroid_error_mm | 8.701677 | 9.224690 | +0.523013 | -0.471182 | 7.097023 | [-1.5955, +3.0149] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | centroid_distance_mm | 8.978677 | 9.339507 | +0.360830 | -0.582634 | 7.128253 | [-1.7678, +2.8547] |
| uniform_plain_set_interaction_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | Dice50 | 0.413572 | 0.428620 | +0.015049 | +0.033771 | 0.092183 | [-0.0153, +0.0440] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | relative_l1 | 0.115614 | 0.116697 | +0.001083 | +0.002724 | 0.016421 | [-0.0048, +0.0060] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | relative_l2 | 0.166950 | 0.164521 | -0.002429 | +0.003551 | 0.037553 | [-0.0162, +0.0083] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | peak_ratio | 1.170468 | 1.177957 | +0.007489 | +0.008875 | 0.054449 | [-0.0106, +0.0251] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.199461 | 0.211656 | +0.012194 | +0.011095 | 0.053566 | [-0.0054, +0.0295] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | peak_absolute_error | 0.120080 | 0.126971 | +0.006891 | +0.006055 | 0.031972 | [-0.0036, +0.0172] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.210533 | 69.870334 | +0.659801 | +0.000000 | 134.194276 | [-43.0555, +43.6725] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | centroid_error_mm | 3.081028 | 2.844978 | -0.236050 | +0.395310 | 3.142038 | [-1.4345, +0.5950] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.202078 | 2.959131 | -0.242947 | +0.413051 | 3.123634 | [-1.4380, +0.5832] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C0 | Dice50 | 0.833542 | 0.832403 | -0.001139 | -0.010936 | 0.073582 | [-0.0242, +0.0241] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | relative_l1 | 0.085527 | 0.087764 | +0.002237 | +0.000974 | 0.006773 | [+0.0000, +0.0044] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | relative_l2 | 0.160457 | 0.157977 | -0.002480 | +0.000212 | 0.023211 | [-0.0111, +0.0038] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | peak_ratio | 1.171963 | 1.172336 | +0.000372 | +0.007155 | 0.059060 | [-0.0190, +0.0196] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.203002 | 0.205509 | +0.002507 | +0.007155 | 0.058638 | [-0.0164, +0.0219] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | peak_absolute_error | 0.122740 | 0.123847 | +0.001107 | +0.003838 | 0.034220 | [-0.0099, +0.0125] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | peak_location_error_mm | 74.135790 | 64.719371 | -9.416420 | +0.000000 | 115.444043 | [-47.0006, +28.1874] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | centroid_error_mm | 1.438567 | 1.543556 | +0.104989 | +0.125150 | 0.621979 | [-0.1023, +0.3094] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.615044 | 1.741299 | +0.126255 | +0.172975 | 0.620767 | [-0.0735, +0.3346] |
| uniform_plain_set_interaction_seed20261019 | Val-A-V4/0 | C1 | Dice50 | 0.816591 | 0.822113 | +0.005522 | +0.000877 | 0.051094 | [-0.0116, +0.0219] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | relative_l1 | 0.409517 | 0.419100 | +0.009583 | +0.018022 | 0.078419 | [-0.0158, +0.0353] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | relative_l2 | 0.567246 | 0.566443 | -0.000802 | +0.005076 | 0.075030 | [-0.0247, +0.0248] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | peak_ratio | 0.783330 | 0.794385 | +0.011055 | -0.013693 | 0.174723 | [-0.0444, +0.0695] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.235335 | 0.279367 | +0.044031 | +0.026966 | 0.154269 | [-0.0065, +0.0939] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | peak_absolute_error | 0.144732 | 0.167705 | +0.022974 | +0.016476 | 0.086015 | [-0.0053, +0.0507] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | peak_location_error_mm | 95.163840 | 118.271035 | +23.107195 | +4.955443 | 83.711865 | [-4.1401, +50.3821] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | centroid_error_mm | 17.423615 | 20.209747 | +2.786132 | +2.735364 | 10.496895 | [-0.4728, +6.3593] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | centroid_distance_mm | 17.626782 | 20.335266 | +2.708484 | +2.778401 | 10.669668 | [-0.6108, +6.3441] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C0 | Dice50 | 0.186929 | 0.352422 | +0.165493 | +0.153701 | 0.180587 | [+0.1097, +0.2271] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | relative_l1 | 0.347525 | 0.352232 | +0.004706 | +0.001506 | 0.061682 | [-0.0153, +0.0251] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | relative_l2 | 0.499801 | 0.495302 | -0.004499 | +0.007287 | 0.069833 | [-0.0274, +0.0181] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | peak_ratio | 0.822123 | 0.817005 | -0.005119 | -0.057926 | 0.163142 | [-0.0580, +0.0484] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.212159 | 0.253294 | +0.041136 | +0.057926 | 0.132964 | [-0.0021, +0.0850] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | peak_absolute_error | 0.130642 | 0.151832 | +0.021190 | +0.039532 | 0.079998 | [-0.0050, +0.0471] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | peak_location_error_mm | 90.175277 | 122.026684 | +31.851407 | +5.978267 | 74.602644 | [+8.2033, +56.6574] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | centroid_error_mm | 11.072322 | 10.015994 | -1.056328 | -2.094026 | 6.199260 | [-3.0159, +1.0129] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | centroid_distance_mm | 11.462345 | 10.179229 | -1.283116 | -2.620226 | 6.395994 | [-3.3217, +0.8399] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0 | C1 | Dice50 | 0.298867 | 0.380768 | +0.081901 | +0.066046 | 0.168031 | [+0.0292, +0.1391] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | relative_l1 | 0.355517 | 0.332617 | -0.022900 | -0.012406 | 0.034111 | [-0.0343, -0.0121] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | relative_l2 | 0.491721 | 0.479543 | -0.012178 | -0.006813 | 0.046825 | [-0.0275, +0.0031] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_ratio | 0.834147 | 0.875010 | +0.040863 | +0.043359 | 0.110754 | [+0.0065, +0.0782] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | absolute_peak_ratio_error | 0.226670 | 0.206269 | -0.020401 | -0.026440 | 0.102717 | [-0.0528, +0.0139] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_absolute_error | 0.142889 | 0.127303 | -0.015586 | -0.017689 | 0.060421 | [-0.0354, +0.0035] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_location_error_mm | 123.211972 | 116.947925 | -6.264047 | -3.359871 | 69.950669 | [-29.4999, +15.7334] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | centroid_error_mm | 14.693713 | 14.984064 | +0.290351 | +0.977714 | 5.647472 | [-1.7176, +1.9479] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | centroid_distance_mm | 14.862678 | 15.095779 | +0.233100 | +0.842863 | 5.624519 | [-1.7694, +1.8894] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | Dice50 | 0.378553 | 0.438754 | +0.060202 | +0.054667 | 0.096304 | [+0.0294, +0.0912] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | relative_l1 | 0.299165 | 0.297780 | -0.001384 | +0.005452 | 0.035422 | [-0.0129, +0.0103] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | relative_l2 | 0.441267 | 0.445318 | +0.004051 | +0.000883 | 0.051176 | [-0.0126, +0.0207] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_ratio | 0.855189 | 0.904803 | +0.049615 | +0.041902 | 0.099818 | [+0.0178, +0.0829] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | absolute_peak_ratio_error | 0.214456 | 0.187494 | -0.026962 | -0.003375 | 0.081175 | [-0.0531, +0.0000] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_absolute_error | 0.135264 | 0.115788 | -0.019476 | -0.002325 | 0.049070 | [-0.0357, -0.0037] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_location_error_mm | 109.185966 | 110.041748 | +0.855782 | -1.634270 | 59.572652 | [-17.7841, +21.3071] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | centroid_error_mm | 9.900563 | 8.123534 | -1.777029 | -0.850982 | 5.318308 | [-3.7698, -0.3454] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | centroid_distance_mm | 10.151552 | 8.319822 | -1.831731 | -0.992382 | 5.413051 | [-3.8647, -0.3850] |
| uniform_plain_set_interaction_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | Dice50 | 0.410368 | 0.438361 | +0.027993 | +0.015452 | 0.087453 | [+0.0004, +0.0575] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | relative_l1 | 0.113762 | 0.113255 | -0.000506 | -0.001340 | 0.014200 | [-0.0050, +0.0044] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | relative_l2 | 0.166994 | 0.162928 | -0.004066 | -0.004517 | 0.020714 | [-0.0105, +0.0032] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | peak_ratio | 1.166835 | 1.167381 | +0.000546 | -0.000935 | 0.054047 | [-0.0175, +0.0180] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.188009 | 0.183313 | -0.004696 | -0.004293 | 0.053840 | [-0.0223, +0.0128] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | peak_absolute_error | 0.112581 | 0.110502 | -0.002079 | -0.002804 | 0.032740 | [-0.0126, +0.0087] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.385500 | 82.580411 | +13.194911 | +0.000000 | 114.371454 | [-23.9123, +51.0656] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | centroid_error_mm | 3.279464 | 2.865919 | -0.413545 | -0.055015 | 1.726975 | [-1.0299, +0.1126] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.451457 | 2.992534 | -0.458923 | -0.030997 | 1.748632 | [-1.0843, +0.0737] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C0 | Dice50 | 0.821413 | 0.834536 | +0.013123 | +0.011895 | 0.040320 | [+0.0000, +0.0263] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | relative_l1 | 0.085928 | 0.087395 | +0.001467 | -0.000030 | 0.008348 | [-0.0009, +0.0045] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | relative_l2 | 0.157314 | 0.160114 | +0.002799 | +0.000252 | 0.014198 | [-0.0008, +0.0082] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | peak_ratio | 1.173953 | 1.167292 | -0.006661 | -0.001060 | 0.055515 | [-0.0255, +0.0108] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.201320 | 0.198370 | -0.002950 | -0.006975 | 0.055631 | [-0.0204, +0.0155] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | peak_absolute_error | 0.120536 | 0.119985 | -0.000551 | -0.003418 | 0.035476 | [-0.0115, +0.0114] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | peak_location_error_mm | 67.568955 | 78.101406 | +10.532450 | +0.000000 | 107.085129 | [-23.8614, +45.5632] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | centroid_error_mm | 1.429295 | 1.466749 | +0.037454 | +0.099152 | 0.568272 | [-0.1637, +0.2129] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.617395 | 1.640636 | +0.023241 | +0.113855 | 0.606694 | [-0.1898, +0.2116] |
| uniform_no_ray_relations_seed20261011 | Val-A-V4/0 | C1 | Dice50 | 0.820338 | 0.823247 | +0.002909 | +0.001228 | 0.028206 | [-0.0067, +0.0117] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | relative_l1 | 0.396765 | 0.523517 | +0.126751 | +0.119688 | 0.075540 | [+0.1014, +0.1508] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | relative_l2 | 0.568319 | 0.666403 | +0.098084 | +0.082933 | 0.084114 | [+0.0709, +0.1255] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | peak_ratio | 0.865758 | 0.769499 | -0.096259 | -0.097608 | 0.133706 | [-0.1396, -0.0527] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.197897 | 0.264694 | +0.066797 | +0.072706 | 0.125087 | [+0.0251, +0.1072] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | peak_absolute_error | 0.117784 | 0.160233 | +0.042449 | +0.043388 | 0.076883 | [+0.0169, +0.0674] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | peak_location_error_mm | 121.683882 | 124.841956 | +3.158073 | +2.255191 | 117.391302 | [-35.6261, +41.2570] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | centroid_error_mm | 16.346618 | 16.458688 | +0.112070 | -1.039376 | 10.348490 | [-3.1682, +3.5547] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | centroid_distance_mm | 16.569914 | 16.527578 | -0.042335 | -1.336903 | 10.569909 | [-3.3846, +3.4844] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C0 | Dice50 | 0.294281 | 0.180810 | -0.113470 | -0.070273 | 0.146845 | [-0.1616, -0.0666] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | relative_l1 | 0.328316 | 0.391785 | +0.063469 | +0.061747 | 0.051243 | [+0.0466, +0.0802] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | relative_l2 | 0.500036 | 0.564933 | +0.064897 | +0.071778 | 0.067602 | [+0.0426, +0.0870] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | peak_ratio | 0.806027 | 0.753035 | -0.052992 | -0.071478 | 0.178600 | [-0.1106, +0.0072] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.234753 | 0.257805 | +0.023052 | +0.014224 | 0.174003 | [-0.0352, +0.0794] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | peak_absolute_error | 0.139727 | 0.159407 | +0.019680 | +0.009704 | 0.105961 | [-0.0152, +0.0543] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | peak_location_error_mm | 111.175341 | 107.372872 | -3.802469 | -1.005690 | 101.769597 | [-36.8887, +30.0703] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | centroid_error_mm | 12.645216 | 14.119039 | +1.473823 | +0.152458 | 7.059613 | [-0.7523, +3.8586] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | centroid_distance_mm | 12.988187 | 14.343236 | +1.355049 | +0.270112 | 7.109135 | [-0.8945, +3.7539] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0 | C1 | Dice50 | 0.316040 | 0.203650 | -0.112389 | -0.033097 | 0.180603 | [-0.1737, -0.0561] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | relative_l1 | 0.330008 | 0.409789 | +0.079780 | +0.078619 | 0.046029 | [+0.0650, +0.0949] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | relative_l2 | 0.486151 | 0.552833 | +0.066682 | +0.064200 | 0.057728 | [+0.0484, +0.0859] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_ratio | 0.951225 | 0.801207 | -0.150019 | -0.148150 | 0.085656 | [-0.1784, -0.1230] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | absolute_peak_ratio_error | 0.176864 | 0.236056 | +0.059192 | +0.042781 | 0.111619 | [+0.0222, +0.0962] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_absolute_error | 0.106410 | 0.146385 | +0.039975 | +0.025353 | 0.066951 | [+0.0183, +0.0625] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | peak_location_error_mm | 104.865243 | 120.647927 | +15.782684 | +11.149333 | 61.289977 | [-3.6380, +36.2599] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | centroid_error_mm | 11.064913 | 13.752415 | +2.687502 | +1.673820 | 13.247406 | [-1.0243, +7.5254] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | centroid_distance_mm | 11.123938 | 13.873232 | +2.749294 | +1.595962 | 13.453986 | [-1.0064, +7.6787] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C0 | Dice50 | 0.427069 | 0.319621 | -0.107447 | -0.127551 | 0.089778 | [-0.1367, -0.0781] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | relative_l1 | 0.280246 | 0.306718 | +0.026472 | +0.028380 | 0.030784 | [+0.0168, +0.0369] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | relative_l2 | 0.450057 | 0.466314 | +0.016257 | +0.016066 | 0.039747 | [+0.0034, +0.0291] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_ratio | 0.925057 | 0.815886 | -0.109171 | -0.115501 | 0.065895 | [-0.1306, -0.0873] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | absolute_peak_ratio_error | 0.175157 | 0.218007 | +0.042850 | +0.036234 | 0.086191 | [+0.0146, +0.0708] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_absolute_error | 0.106419 | 0.136142 | +0.029722 | +0.021481 | 0.051748 | [+0.0130, +0.0468] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | peak_location_error_mm | 97.788347 | 117.556873 | +19.768526 | +8.565410 | 62.534168 | [-0.9618, +40.0204] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | centroid_error_mm | 8.701677 | 10.377883 | +1.676205 | +1.301341 | 6.146256 | [-0.2450, +3.7779] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | centroid_distance_mm | 8.978677 | 10.618097 | +1.639419 | +1.293640 | 6.131375 | [-0.2813, +3.7351] |
| uniform_no_ray_relations_seed20261011 | clustered-V4/0,1,2,3,4 | C1 | Dice50 | 0.413572 | 0.367205 | -0.046367 | -0.052869 | 0.078698 | [-0.0724, -0.0212] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | relative_l1 | 0.115614 | 0.117124 | +0.001510 | +0.001185 | 0.012753 | [-0.0030, +0.0054] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | relative_l2 | 0.166950 | 0.162980 | -0.003971 | -0.000089 | 0.022741 | [-0.0123, +0.0024] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | peak_ratio | 1.170468 | 1.150760 | -0.019707 | -0.005905 | 0.069271 | [-0.0435, +0.0013] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.199461 | 0.186848 | -0.012614 | -0.001984 | 0.070955 | [-0.0372, +0.0090] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | peak_absolute_error | 0.120080 | 0.113389 | -0.006691 | -0.001172 | 0.042116 | [-0.0213, +0.0062] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.210533 | 53.403919 | -15.806614 | +0.000000 | 118.405408 | [-54.5539, +22.4044] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | centroid_error_mm | 3.081028 | 3.150105 | +0.069078 | +0.630961 | 1.867115 | [-0.6114, +0.6206] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.202078 | 3.292604 | +0.090526 | +0.692779 | 1.881037 | [-0.5927, +0.6448] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C0 | Dice50 | 0.833542 | 0.832850 | -0.000692 | +0.000000 | 0.031950 | [-0.0113, +0.0097] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | relative_l1 | 0.085527 | 0.085125 | -0.000401 | +0.000647 | 0.005848 | [-0.0024, +0.0014] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | relative_l2 | 0.160457 | 0.156698 | -0.003759 | -0.002042 | 0.018016 | [-0.0105, +0.0012] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | peak_ratio | 1.171963 | 1.148703 | -0.023261 | -0.013762 | 0.078703 | [-0.0515, -0.0004] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.203002 | 0.191208 | -0.011794 | -0.007673 | 0.066986 | [-0.0354, +0.0084] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | peak_absolute_error | 0.122740 | 0.116527 | -0.006213 | -0.005235 | 0.039191 | [-0.0198, +0.0057] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | peak_location_error_mm | 74.135790 | 54.194992 | -19.940798 | +0.000000 | 109.956827 | [-55.7622, +15.4908] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | centroid_error_mm | 1.438567 | 1.510465 | +0.071898 | +0.066714 | 0.377013 | [-0.0465, +0.2020] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.615044 | 1.701730 | +0.086687 | +0.057352 | 0.392999 | [-0.0382, +0.2215] |
| uniform_no_ray_relations_seed20261019 | Val-A-V4/0 | C1 | Dice50 | 0.816591 | 0.820692 | +0.004101 | +0.000000 | 0.031299 | [-0.0065, +0.0142] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | relative_l1 | 0.409517 | 0.462972 | +0.053455 | +0.048229 | 0.082307 | [+0.0273, +0.0813] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | relative_l2 | 0.567246 | 0.578774 | +0.011528 | +0.001475 | 0.078922 | [-0.0130, +0.0384] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | peak_ratio | 0.783330 | 0.858696 | +0.075365 | +0.055511 | 0.163361 | [+0.0232, +0.1296] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.235335 | 0.216190 | -0.019146 | -0.021362 | 0.148386 | [-0.0693, +0.0276] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | peak_absolute_error | 0.144732 | 0.133434 | -0.011298 | -0.010965 | 0.089436 | [-0.0414, +0.0171] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | peak_location_error_mm | 95.163840 | 86.419634 | -8.744206 | -10.291017 | 83.434832 | [-35.2922, +19.0908] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | centroid_error_mm | 17.423615 | 18.686805 | +1.263190 | +0.890373 | 11.848788 | [-2.5707, +5.1071] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | centroid_distance_mm | 17.626782 | 18.646063 | +1.019281 | +1.012424 | 11.950692 | [-2.8621, +4.9160] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C0 | Dice50 | 0.186929 | 0.355368 | +0.168439 | +0.200000 | 0.130952 | [+0.1256, +0.2118] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | relative_l1 | 0.347525 | 0.376137 | +0.028612 | +0.018000 | 0.078258 | [+0.0042, +0.0553] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | relative_l2 | 0.499801 | 0.486588 | -0.013213 | -0.015014 | 0.069885 | [-0.0354, +0.0097] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | peak_ratio | 0.822123 | 0.895439 | +0.073316 | +0.089337 | 0.157750 | [+0.0218, +0.1254] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.212159 | 0.162033 | -0.050125 | -0.040587 | 0.133232 | [-0.0955, -0.0078] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | peak_absolute_error | 0.130642 | 0.099237 | -0.031405 | -0.025933 | 0.084575 | [-0.0608, -0.0047] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | peak_location_error_mm | 90.175277 | 91.503434 | +1.328158 | -2.822160 | 94.806250 | [-29.2289, +32.8480] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | centroid_error_mm | 11.072322 | 10.718895 | -0.353427 | -1.542890 | 8.189195 | [-2.9634, +2.3710] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | centroid_distance_mm | 11.462345 | 10.863107 | -0.599238 | -1.770122 | 8.269200 | [-3.2377, +2.1448] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0 | C1 | Dice50 | 0.298867 | 0.413468 | +0.114601 | +0.092443 | 0.143923 | [+0.0698, +0.1631] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | relative_l1 | 0.355517 | 0.368044 | +0.012528 | +0.011095 | 0.049960 | [-0.0031, +0.0297] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | relative_l2 | 0.491721 | 0.501860 | +0.010139 | +0.014732 | 0.040066 | [-0.0027, +0.0233] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_ratio | 0.834147 | 0.917128 | +0.082981 | +0.104264 | 0.116261 | [+0.0451, +0.1210] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | absolute_peak_ratio_error | 0.226670 | 0.182047 | -0.044623 | -0.040700 | 0.097837 | [-0.0763, -0.0123] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_absolute_error | 0.142889 | 0.113563 | -0.029326 | -0.024867 | 0.058966 | [-0.0491, -0.0106] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | peak_location_error_mm | 123.211972 | 115.835986 | -7.375987 | -10.599205 | 65.358810 | [-29.0997, +13.3298] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | centroid_error_mm | 14.693713 | 14.144754 | -0.548959 | +0.137576 | 9.005601 | [-3.7938, +2.0451] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | centroid_distance_mm | 14.862678 | 14.146539 | -0.716140 | -0.154898 | 9.069133 | [-3.9819, +1.8931] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C0 | Dice50 | 0.378553 | 0.442457 | +0.063905 | +0.040756 | 0.079201 | [+0.0390, +0.0902] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | relative_l1 | 0.299165 | 0.309817 | +0.010652 | +0.003842 | 0.028933 | [+0.0018, +0.0207] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | relative_l2 | 0.441267 | 0.448393 | +0.007126 | +0.009804 | 0.035567 | [-0.0046, +0.0185] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_ratio | 0.855189 | 0.950044 | +0.094855 | +0.094273 | 0.092488 | [+0.0644, +0.1255] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | absolute_peak_ratio_error | 0.214456 | 0.166724 | -0.047732 | -0.044515 | 0.086006 | [-0.0755, -0.0195] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_absolute_error | 0.135264 | 0.104161 | -0.031103 | -0.026292 | 0.053379 | [-0.0487, -0.0140] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | peak_location_error_mm | 109.185966 | 100.506078 | -8.679888 | +4.259177 | 64.804555 | [-30.3364, +12.4946] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | centroid_error_mm | 9.900563 | 9.110580 | -0.789984 | +0.292805 | 7.457923 | [-3.6406, +1.1410] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | centroid_distance_mm | 10.151552 | 9.295909 | -0.855644 | +0.184377 | 7.583778 | [-3.7584, +1.1109] |
| uniform_no_ray_relations_seed20261019 | clustered-V4/0,1,2,3,4 | C1 | Dice50 | 0.410368 | 0.449461 | +0.039094 | +0.029227 | 0.070810 | [+0.0167, +0.0627] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | relative_l1 | 0.113762 | 0.113582 | -0.000179 | -0.000231 | 0.003121 | [-0.0012, +0.0008] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | relative_l2 | 0.166994 | 0.167845 | +0.000851 | +0.001215 | 0.002426 | [+0.0000, +0.0016] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | peak_ratio | 1.166835 | 1.168035 | +0.001200 | -0.001309 | 0.012780 | [-0.0027, +0.0056] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.188009 | 0.191408 | +0.003399 | +0.000989 | 0.012366 | [-0.0004, +0.0076] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | peak_absolute_error | 0.112581 | 0.114405 | +0.001824 | +0.000584 | 0.007233 | [-0.0004, +0.0043] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.385500 | 75.689014 | +6.303514 | +0.000000 | 33.827547 | [+0.0050, +18.2679] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | centroid_error_mm | 3.279464 | 2.902260 | -0.377204 | -0.473312 | 0.542146 | [-0.5526, -0.1982] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.451457 | 3.044050 | -0.407407 | -0.565381 | 0.540010 | [-0.5799, -0.2297] |
| obs_token_seed20261011 | Val-A-V4/0 | C0 | Dice50 | 0.821413 | 0.819416 | -0.001997 | +0.000000 | 0.010873 | [-0.0055, +0.0015] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | relative_l1 | 0.085928 | 0.086145 | +0.000217 | +0.000242 | 0.001522 | [-0.0003, +0.0007] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | relative_l2 | 0.157314 | 0.157662 | +0.000347 | -0.000055 | 0.002051 | [-0.0003, +0.0011] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | peak_ratio | 1.173953 | 1.173291 | -0.000662 | -0.000034 | 0.007652 | [-0.0032, +0.0018] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.201320 | 0.203247 | +0.001927 | +0.000928 | 0.007428 | [-0.0004, +0.0044] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | peak_absolute_error | 0.120536 | 0.121641 | +0.001105 | +0.000474 | 0.004670 | [-0.0004, +0.0027] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | peak_location_error_mm | 67.568955 | 67.296847 | -0.272109 | +0.000000 | 1.609818 | [-0.8163, +0.0000] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | centroid_error_mm | 1.429295 | 1.439196 | +0.009901 | +0.019599 | 0.229857 | [-0.0688, +0.0818] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.617395 | 1.623360 | +0.005965 | +0.009360 | 0.237221 | [-0.0748, +0.0804] |
| obs_token_seed20261011 | Val-A-V4/0 | C1 | Dice50 | 0.820338 | 0.819774 | -0.000564 | +0.000000 | 0.007548 | [-0.0029, +0.0021] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | relative_l1 | 0.396765 | 0.439096 | +0.042331 | +0.042063 | 0.030051 | [+0.0329, +0.0524] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | relative_l2 | 0.568319 | 0.585432 | +0.017113 | +0.019477 | 0.026545 | [+0.0084, +0.0257] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | peak_ratio | 0.865758 | 0.844644 | -0.021115 | -0.003120 | 0.066486 | [-0.0449, -0.0016] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.197897 | 0.216400 | +0.018503 | -0.000780 | 0.065074 | [-0.0004, +0.0419] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | peak_absolute_error | 0.117784 | 0.128416 | +0.010633 | -0.000537 | 0.040935 | [-0.0010, +0.0255] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | peak_location_error_mm | 121.683882 | 114.047093 | -7.636789 | +0.000000 | 39.111221 | [-21.6908, +0.0348] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | centroid_error_mm | 16.346618 | 14.331329 | -2.015289 | -2.055216 | 4.589635 | [-3.4899, -0.5007] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | centroid_distance_mm | 16.569914 | 14.407106 | -2.162807 | -2.067243 | 4.696548 | [-3.6676, -0.6161] |
| obs_token_seed20261011 | clustered-V4/0 | C0 | Dice50 | 0.294281 | 0.305498 | +0.011217 | +0.000000 | 0.041240 | [-0.0012, +0.0258] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | relative_l1 | 0.328316 | 0.346467 | +0.018151 | +0.021723 | 0.016850 | [+0.0127, +0.0236] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | relative_l2 | 0.500036 | 0.504732 | +0.004697 | +0.000601 | 0.015127 | [-0.0000, +0.0098] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | peak_ratio | 0.806027 | 0.797134 | -0.008893 | -0.018194 | 0.093542 | [-0.0346, +0.0246] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.234753 | 0.239391 | +0.004638 | +0.013516 | 0.091658 | [-0.0284, +0.0296] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | peak_absolute_error | 0.139727 | 0.143756 | +0.004029 | +0.008090 | 0.048406 | [-0.0131, +0.0177] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | peak_location_error_mm | 111.175341 | 127.767917 | +16.592576 | +0.000000 | 70.235181 | [-5.8137, +40.3075] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | centroid_error_mm | 12.645216 | 11.721103 | -0.924113 | -1.026148 | 2.085520 | [-1.6075, -0.2503] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | centroid_distance_mm | 12.988187 | 12.019550 | -0.968637 | -1.032125 | 2.095423 | [-1.6530, -0.2937] |
| obs_token_seed20261011 | clustered-V4/0 | C1 | Dice50 | 0.316040 | 0.312261 | -0.003778 | +0.000000 | 0.037348 | [-0.0162, +0.0085] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | relative_l1 | 0.115614 | 0.137391 | +0.021777 | +0.021680 | 0.013170 | [+0.0174, +0.0260] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | relative_l2 | 0.166950 | 0.172066 | +0.005116 | +0.006778 | 0.010827 | [+0.0016, +0.0086] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | peak_ratio | 1.170468 | 1.170272 | -0.000196 | -0.001435 | 0.016040 | [-0.0054, +0.0050] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | absolute_peak_ratio_error | 0.199461 | 0.195281 | -0.004181 | -0.003511 | 0.015470 | [-0.0091, +0.0009] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | peak_absolute_error | 0.120080 | 0.117281 | -0.002799 | -0.002338 | 0.009442 | [-0.0058, +0.0003] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | peak_location_error_mm | 69.210533 | 86.328994 | +17.118461 | +0.000000 | 60.119890 | [-0.7219, +38.9716] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | centroid_error_mm | 3.081028 | 2.863083 | -0.217945 | -0.118488 | 0.834865 | [-0.5075, +0.0397] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | centroid_distance_mm | 3.202078 | 2.946996 | -0.255083 | -0.158740 | 0.858072 | [-0.5504, +0.0130] |
| obs_token_seed20261019 | Val-A-V4/0 | C0 | Dice50 | 0.833542 | 0.847301 | +0.013759 | +0.007391 | 0.036450 | [+0.0033, +0.0271] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | relative_l1 | 0.085527 | 0.089884 | +0.004358 | +0.003711 | 0.004479 | [+0.0030, +0.0059] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | relative_l2 | 0.160457 | 0.160679 | +0.000222 | -0.000189 | 0.004903 | [-0.0013, +0.0019] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | peak_ratio | 1.171963 | 1.173238 | +0.001274 | -0.000097 | 0.014949 | [-0.0034, +0.0064] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | absolute_peak_ratio_error | 0.203002 | 0.198276 | -0.004726 | -0.007068 | 0.014218 | [-0.0095, -0.0002] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | peak_absolute_error | 0.122740 | 0.119562 | -0.003178 | -0.003368 | 0.009088 | [-0.0063, -0.0003] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | peak_location_error_mm | 74.135790 | 58.021856 | -16.113934 | +0.000000 | 54.270798 | [-37.3642, +0.0000] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | centroid_error_mm | 1.438567 | 1.465005 | +0.026438 | -0.006436 | 0.345578 | [-0.0839, +0.1409] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | centroid_distance_mm | 1.615044 | 1.623733 | +0.008690 | -0.024099 | 0.352576 | [-0.1037, +0.1258] |
| obs_token_seed20261019 | Val-A-V4/0 | C1 | Dice50 | 0.816591 | 0.816658 | +0.000067 | +0.000000 | 0.018036 | [-0.0061, +0.0058] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | relative_l1 | 0.409517 | 0.452882 | +0.043365 | +0.035491 | 0.043293 | [+0.0301, +0.0582] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | relative_l2 | 0.567246 | 0.581092 | +0.013847 | +0.004852 | 0.030101 | [+0.0043, +0.0239] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | peak_ratio | 0.783330 | 0.787100 | +0.003770 | -0.007338 | 0.070545 | [-0.0196, +0.0264] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | absolute_peak_ratio_error | 0.235335 | 0.237855 | +0.002520 | +0.014702 | 0.065399 | [-0.0184, +0.0248] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | peak_absolute_error | 0.144732 | 0.146767 | +0.002035 | +0.009598 | 0.038888 | [-0.0102, +0.0154] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | peak_location_error_mm | 95.163840 | 96.979149 | +1.815309 | +0.000000 | 19.284710 | [-2.4531, +8.9302] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | centroid_error_mm | 17.423615 | 18.804136 | +1.380521 | +0.892987 | 5.167795 | [-0.2568, +3.0782] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | centroid_distance_mm | 17.626782 | 19.005554 | +1.378772 | +0.764873 | 5.186602 | [-0.2639, +3.0822] |
| obs_token_seed20261019 | clustered-V4/0 | C0 | Dice50 | 0.186929 | 0.195801 | +0.008872 | +0.000000 | 0.055220 | [-0.0086, +0.0278] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | relative_l1 | 0.347525 | 0.370654 | +0.023129 | +0.020217 | 0.038844 | [+0.0106, +0.0359] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | relative_l2 | 0.499801 | 0.505989 | +0.006187 | +0.002105 | 0.029931 | [-0.0033, +0.0161] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | peak_ratio | 0.822123 | 0.777985 | -0.044138 | -0.049761 | 0.050532 | [-0.0603, -0.0276] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | absolute_peak_ratio_error | 0.212159 | 0.251699 | +0.039540 | +0.042266 | 0.051565 | [+0.0226, +0.0562] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | peak_absolute_error | 0.130642 | 0.155095 | +0.024452 | +0.028837 | 0.031622 | [+0.0141, +0.0347] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | peak_location_error_mm | 90.175277 | 108.519888 | +18.344612 | +0.000000 | 53.220384 | [+3.4863, +37.6735] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | centroid_error_mm | 11.072322 | 11.507734 | +0.435412 | +0.262551 | 5.772688 | [-1.2984, +2.4448] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | centroid_distance_mm | 11.462345 | 11.785149 | +0.322804 | -0.022989 | 5.804380 | [-1.4197, +2.3437] |
| obs_token_seed20261019 | clustered-V4/0 | C1 | Dice50 | 0.298867 | 0.276155 | -0.022712 | -0.011023 | 0.044769 | [-0.0385, -0.0093] |

## 审计、限制与交付

所有1400份可与历史核对的C0/C1字段，其全部指标最大绝对差为0。独立实现用案例计数权重、逐draw差先平均及独立排序插值复核bootstrap，最大差为5.4e-14；全部1400预测和全部checkpoint哈希再次核验。
首次登记后在静态检查中发现smoke目标返回volumes是tuple，修正为直接迭代，旧登记完整保留在mrgr_oari_targeted_v1_registration_attempt01；正式训练前重新登记，未改变任何训练预算、损失或正式权重。另有上述GPU折间启动竞争与第三训练的受控恢复，原失败日志、退出码、last.pt身份和恢复证据完整保留；不能把该运行称为从未中断。其余异常若有保留在controller/status和scheduling日志中，不静默替换模型、精度或数据。
产物包含：独立设计、训练protocol/config/结构/源代码哈希、CPU/GPU工程检查、完整300轮日志与checkpoint SHA256、GPU资源和锁记录、1400份预测身份与文件哈希、统一封存、35份GT64访问审计、逐案例逐draw指标、按案例配对统计、同卡成本比较、独立核验及acceptance.json。大数组/checkpoint保留scratch，仓库导出其身份和全部分析证据，不复制或覆盖历史大文件。
本轮不自动改变论文正式模型或重选检查点。最终结论限于两个固定训练种子、两类布局与已使用35案例；不能单独证明模型掌握可泛化几何规律或某专门模块不可替代。
