# Ray-Senseiver 输出初始化：独立两臂小预算协议 v1

用户授权停止原塌缩 run；原 run 已在第113轮完整原子保存边界停止，完成9379 updates，标记 FAILED_LEARNING / NEAR_ZERO_COLLAPSE / EARLY_TERMINATED。last/best与全部历史保留，后续900条/GT评价取消。原诊断报告中“继续300轮”仅代表当时授权，已被本轮指令覆盖；不改写历史报告。

本轮仅检验：训练测量支持的输出尺度初始化，能否避免近零塌缩并使任务适配产生有效学习。非唯一原因假设，不外推原Senseiver能力。原模型、官方共享参数语义、完整262144 XYZ query、全部valid rays（包括有效零）、P_diff/K96、FP32和完整原loss均不变。冻结Ours/DeepSets。

数据固定 source-index train 顺序前8例，Pool A，seed20261011、sampler固定epoch0，自然A3–5/B3–5/H2，两两不交。固定顺序、每连续2例一组、不shuffle；与完整训练在线重采样不同。H不输入网络、不参与尺度估计，仍参与原loss，因此是固定训练子集可学习性探针，不是独立泛化Val。TrainEightGuard仅允许8个train projection NPY，禁GT64/128、Val/test、Pool B和foreign checkpoint。

仅用全部有效A/B射线（含valid零），float64归约：χ=Σ(length*y)/(1e6Σlength²)。y单位ppm·m、length单位m；原projector常量摩尔分数满足y=1e6 length χ。调用已有mole_to_mass，使用原16.043/28.965常数，得到C_init。非法分母/尺度立即停止，无经验fallback。仅logit计算clamp到[1e−6,1−1e−6]并记录触界。对全部8例A/B完整射线通过原projector核验常量场（FP32 atol .002 ppm·m，rtol2e−5）。完整case/station/valid ray flat index、测量路径tensor hashes、数量、分子分母、尺度与初态hash写入注册清单。

Control从原seed fresh随机初始化；Treatment从同一fresh初态复制，唯一差异decoder.postproc：weight乘0.01、bias填logit(C_init)，weight不清零。全部其他参数逐元素相等。无warm start。工程smoke各1步使用独立临时模型/optimizer，仅在其自身目录保存/读回；正式训练随后重新按seed构建并核对注册初态hash，不读取smoke权重。

每臂最多40 epochs，8例/轮，effective batch2、4 steps/轮、最多160 updates；AdamW lr2e−4/wd1e−4，clip1，cosine T_max300，原L_hold+.5L_cross+.1L_obs+.05L_cons及内部归约全部直接复用。batch2/accum1，若OOM仅改chunk/checkpoint/device或batch1/accum2并单独记录工程配置，不减少ray/query或改loss。不杀其他进程。两臂只可读各自last.pt完整恢复模型/optimizer/scheduler/RNG/update/退出状态；epoch0.pt仅归档，已科学停止的臂不能续跑。

正式训练前将协议、配置、固定输入、尺度、初态和源码hash提交Git；入口验证HEAD内注册文件字节及冻结源码。工程smoke在正式训练前独立执行；测试覆盖尺度/单位、输出层差异、退出规则、完整恢复、query梯度/轴序和访问边界。不得以失败结果为由修改seed/case/尺度或增加搜索。

epoch0保存完整状态；每epoch记录训练中聚合指标和轮末固定8例测量评价，H/cross/observed L1/L2、所有loss项、输出min/max/mean/std及case变化。训练中聚合跨4次更新，轮末评价来自同一权重。每5轮（以及epoch0）健康检查：全8例真实postproc logits分位数、<-12比例及sigmoid导数；固定case0/1/2完整空间/差异、未clip encoder/latents/decoder梯度；每个实际step三组postclip梯度和更新L2；normal vs zero只将raw/normalized测量置零，保留geometry/path/masks，MA相对分母为normal meanabs+1e−12，并报告绝对差。检查259个预固定query的full vs chunk37全部参数/每query梯度；原纯相对判据保留，同时预登记绝对+相对L2数值容差3e−5+2e−4*reference，预测max abs2e−6，避免近零分母误报。正式输出始终完整64³。记录GPU CUDA event elapsed（含排队和host间隙，不是独占kernel时间）、墙钟、峰值与实际updates；共享GPU成本不当作受控benchmark。

退出规则严格解释“第5/20轮后”为epoch>5/epoch>20，健康检查在5倍数轮末。B最早在10、15连续两次触发；C最早在25、30连续两次触发。所有退出事件含条件和计数，恢复保持计数。

- A：NaN/Inf、query漏梯度/轴错、丢valid、访问违规或optimizer组无真实更新立即停该臂，保留上一次原子last有效状态，写error及影响。修复必须新版本重新登记，不能静默续改。
- B：连续两次eligible检查同时H L1≥.98、>95%logits<−12、输出mean<.01C_init，停止为NEAR_ZERO_COLLAPSE；同时报告空间/case及测量依赖。
- C：第20轮后，每5轮以**轮末固定8例H**计算最近5轮均值和前5轮均值，连续两次最近均值≥.98且相对改善<1%，实现/更新正常则FAILED_TO_LEARN。
- D：H明显低于零场、非塌缩且改善则继续至40轮；最大预算停止为BUDGET_COMPLETE。

两臂仅在共同epoch/update位置matched比较，终点另列；所有曲线保留。Treatment最低GO：最后3次健康检查H均<.90；无工程异常/塌缩；固定诊断case至少一对relative MA≥1e−3作为“可辨case变化”的操作阈值；至少2个诊断case normal-vs-zero relative MA≥1e−3；encoder/latents/decoder梯度和真实更新全部有效。后两类相对MA分母均normal/pair首个预测meanabs+1e−12，并报告绝对值。以上仅工程可学习性条件，不证明3D正确或达到旧模型精度。

NO-GO保留，不追加倍率/bias/lr/loss搜索；仅建议一个下一阶段干预。consistency小loss不等于小梯度，不预设为唯一原因。即使GO，本轮只登记正式训练建议，不自动启动300轮。大checkpoint在外部独立目录；Git只提交代码、配置、清单、标量、曲线、访问审计、中文报告及README更新，最终经7897代理推送。

入口：`python scripts/run_senseiver_init_probe.py --phase prepare` → 提交注册 → `--phase smoke` → `--phase train --arm control` / `--arm treatment`。报告入口只读取本实验已存标量，不读GT/权重。
