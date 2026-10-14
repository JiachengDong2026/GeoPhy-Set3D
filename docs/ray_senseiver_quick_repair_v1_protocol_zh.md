# Ray-Senseiver 快速定位与单次修复闭环 v1

本轮完整执行一次：A冻结latent读出 → B按证据只选一个最小结构修复并另行提交设计 → C实际fresh两臂重建对照。不能只交后续建议，不搜索loss/增益/latent数量，不读取GT64/128、Val、Pool B或test，不训练/重评Ours/DeepSets；历史源码、数据、checkpoint、报告和结论不改。

阶段A固定source-index前16个train，0–7拟合、8–15独立检查。复用已登记V4-G1-d0布局真实station IDs和顺序，每例全部相同几何/有效射线身份，目标为所有有效输入normalized测量均值，保留有效零。读取原Pool A，不按测量或误差选站。直接均值oracle及仅拟合集目标均值常量为参照，oracle独立核对raw/Y_SCALE和normalized字段及mask。

仅只读加载最新consistency Treatment last35（140历史更新、FAILED_TO_LEARN、relative=consistency=0），SHA 448ef11e82b0b29e26be2ecc0d10909f575b649cb30c1b00a4142e18219bec71，独立eval冻结encoder全部参数/buffers。一次提取16份完整64×64 latents，后续头训练不再运行encoder。所有训练统计仅0–7：latent逐元素均值、centered latent全局RMS、目标均值/population std；近零≤1e−12即退化终止读出，不用检查集补估。

A头：flatten标准化完整latent，FP64 dual岭回归，λ=.001×trace(ZZᵀ)/8，保存条件数和解残差，不搜索。B头：fresh原Decoder结构、固定域中心Fourier query（物理中心.3m，归一化坐标0），标准化scalar目标、线性输出、MSE；FP32 AdamW lr2e−4/wd.01（明确原默认值）、full-fit8 batch，seed20261011，最多200更新，无scheduler。只以每次更新后拟合MSE选best（含step0），不让检查集反向或选择best。最终才统一评价两头的fit/check MSE/MAE及散点；check MSE<常量基准50%只是本轮可读参考，不能称三维重建或普遍泛化。

B选择规则：线性有效/query失败优先decoder；两者均失败且oracle正确优先输入；两者均有效则结合已有阶段响应只选一个有依据的小改动并明确不确定性。若线性失败/query有效，已有decoder可读证据优先输入弱路径方向。不能按C结果再选第二个版本。阶段A后形成独立设计、唯一变量、参数/计算成本及未证明事项；代码/配置/初态/固定输入hash提交后才运行C。新增结构必须在新文件，不修改历史模型或core。

C原前8train、sampler始终epoch0、原A3–5/B3–5/H2两两不交、原顺序每连续2例一batch。两臂fresh seed20261011，共享参数逐元素一致且Control复现已登记尺度初态a51ae38bb2d67202a4766de25c0e01bc9832fb8887756d2f7ebbee41a38fc5be；原C_init=.0015582415568457682、postproc weight×.01、bias=logit(C_init)。新增参数单独记录，不加载阶段A、smoke或失败训练权重。Control原结构，Treatment唯一选定修复。

两臂相同原SmoothL1 hold1/cross.5/observed.1，relative=consistency=0；诊断仍计算原relative及consistency，不加训练梯度。不改变beta.01/epsilon1e−6/Y_SCALE/mask/归约，全部valid ray、完整64³ XYZ/yheight、P_diff/K96；FP32 effective batch2、AdamW lr2e−4/wd1e−4、clip1、cosine T_max300。各最多80epochs/320steps，每轮4步，不因total下降判成功。H在训练loss中，仅固定子集可学习性。epoch0完整保存；每轮train/同权重eval L1/L2/SmoothL1/输出/真实梯度与更新/时间显存；每5轮health含logits、空间/case变化、normal-zero测量依赖、三组梯度及query分块梯度核验。normal-zero只置零测量，分母normal meanabs+1e−12。

工程异常立即停止并保存最后有效原子checkpoint/错误/访问审计；新修复须另登记，不静默改已开始run。科学停止：严格epoch>5、连续两次health同时H L1≥.98、>95%logit<−12、mean<.01C_init→NEAR_ZERO_COLLAPSE。严格epoch>20，最近5轮均值H L1≥.98，L1/L2各相对前5轮改善均<1%，且固定case0/1/2不足两例normal-zero relative MA≥.001，连续两次→FAILED_TO_LEARN。改善或依赖形成则最多80；epoch40不自动退出。只从本臂last恢复完整optimizer/scheduler/RNG/计数，科学已终止不得续跑。OOM仅工程chunk/checkpoint/device或batch1/accum2，不丢ray/query，不杀他人进程。

GO候选必须Treatment最后3次health H L1均<.90，无异常/塌缩、至少一固定case pair relative MA≥.001、至少两例测量依赖≥.001，encoder/latents/decoder梯度及更新有效。只在共同epoch/update配对，不横比读出和重建loss。否则明确失败/有限改善归档，不继续结构或loss搜索；成功也只冻结候选建议，不启动300轮。必要测试限定新读出/训练统计分割、唯一修复及共享初态、冻结、退出/恢复/访问边界与适用回归。大tensor/权重外部，Git保存标量、关键图、报告、hash；完成更新README并经7897推送后停止。
