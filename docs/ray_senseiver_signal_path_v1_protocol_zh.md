# Ray-Senseiver 测量信号通路独立诊断：训练前登记 v1

本轮检查射线测量→packed ray token→encoder线性预处理/LayerNorm→cross-attention与3个encoder block→最终latents→decoder→logit→sigmoid的实际响应。没有训练，不创建optimizer、不写.grad、不执行step、不改参数/buffers/激活/尺度/attention设置，不扫描增益或超参数，不读三维GT。完成后只提出一个有证据支持的待验证改动建议，本轮不实现。

历史初始化、relative和consistency有限实验均没有形成有效测量相关学习；最新两臂35轮/140更新FAILED_TO_LEARN、正式NO-GO。继续保持这些历史记录及Ours/DeepSets冻结。203维中测量占1维只是结构线索，不预设其为根因；小梯度不等于不可学习。

只读加载最新consistency探针Treatment：epoch0.pt SHA `950cb7be68e20bdccda7c9f8e002c71410a44b8ff3416c6384310730130b982a`，第35轮last.pt SHA `448ef11e82b0b29e26be2ecc0d10909f575b649cb30c1b00a4142e18219bec71`。两者relative=0/consistency=0，原尺度初态hash a51ae38bb2d67202a4766de25c0e01bc9832fb8887756d2f7ebbee41a38fc5be，C_init=.0015582415568457682。独立模型副本eval，checkpoint的config/epoch/updates/source与注册hash均核验。原文件永不写，禁止恢复训练。

固定原train0/1/2、Pool A、sampler epoch0、原A输入，复用已登记A/B/H身份；H只作固定测量评价，不输入网络。wrong_case循环0←1、1←2、2←3，读取source在target的相同station IDs/顺序和ray位置的测量，通过原_views提取，绝不滚动source自身不同A站位数组。共享Pool A几何逐字段核对；不能保持身份则停止该模式报告，不补插值/抽站。SignalGuard继承并收紧原TrainEightGuard，实际只准train0..3的4个projection文件、Pool A geometry和精确两个Treatment checkpoint；自己的诊断NPZ可读。禁GT64/128、Val、Pool B、test、其他权重；不通过关闭旧guard扩大权限。

每模型×3 target×5模式：normal、zero（raw/normalized置0）、wrong_case、gain_plus×1.01、gain_minus×.99。只改输入副本的两个测量字段，所有station/path/direction/near/origin/valid/view mask/query及其他输入保留对象身份；目标仍target的原A/H测量。有效零测量保留。不视为采集工况或augmentation。

预登记主要完整场前向严格30次（2×3×5），每次完整64³=262144、XYZ轴序、y为高度，全部valid ray，原chunk16384及P_diff/K96。每次started/returned/saved分别追加并flush到账本。保存成功的结果按身份/hash跳过，不重复forward；有started而无完整saved则停止要求核查，不静默重跑。固定原259 query子集为torch.linspace(0,262143,259).long，局部autograd另6个图（2×3），与30个完整场分账。每次完整场两次projector（A/H）共60次。没有模型smoke完整场额外前向；单元测试只用合成小输入和少量query，另记录。

只读hook采集实际调用：valid raw/normalized、packed tokens、encoder preproc、各cross KV LayerNorm、每次cross residual后latents、每个encoder block、encoder final；decoder原96维坐标和64维learned query分别标注固定项、query preproc、纯cross-attention以及含residual/MLP的cross layer输出、postproc logits及最终浓度。共享layer_n按真实顺序记录block2/block3，不能覆盖。decoder所有chunk统计按元素求和/平方和及真实计数归约，保存chunk shape清单，最后总query数262144；不抓attention矩阵、不改need_weights/fastpath或forward。

normal配对每个干预、每阶段记录shape/调用数/有效元素数、min/max/mean/std/RMS/非有限数，差MAE/RMS/max、相对RMS、cosine/相关及近零分母次数。相对RMS=RMS(diff)/RMS(normal)，分母≤1e−12为null，不制造大比值；浓度另报旧relative MA=MAE/(normal meanabs+1e−12)，两者不同。排除padding/invalid；固定项应不变。不同阶段维数/尺度/LayerNorm不同，不能仅凭这些比值宣称信息增减。

encoder线性preproc严格按真实通道拆分：measurement[0:1]、raw geometry[1:11]（entry3/exit3/direction3/normalized length1）、Fourier[11:203]（entry96/exit96）以及bias。以真实W分块F.linear相加，核验重构真实输出（FP32 atol2e−6/rtol2e−5），保存输入分布/RMS、映射贡献RMS、weight块Frobenius和分布、测量列64个系数初末变化。线性分解不延伸到LayerNorm之后，贡献小或bias大不单独证明丢弃测量。normal/zero/wrong真实preproc变化独立记录。

有限局部autograd：u=原raw/Y_SCALE作为leaf，raw=u×Y_SCALE、normalized=u同步赋给输入副本，ray_tokens仍读真实raw字段。每个模型/target只encode一次、decode原259坐标一次。采集preproc、final latent、sampled logit/C；每阶段4个固定Rademacher探针phi=sum(sign*x)/sqrt(numel)，seed20261015+100*stage+probe，logit/C使用同符号以比较激活链。对u求梯度的L2/RMS/范围、invalid梯度数与非有限情况，无完整Jacobian、不写参数.grad。不创建optimizer。保存每个方向的seed派生规则/sign hash/shape/归一化。记录sigmoid导数、dlogC/dlogit=1−sigmoid(z)，区分绝对幅度缩放与相对浓度响应。

利用30次已存gain±1%分别计算四阶段同探针有限差分(plus−minus)/.02，对照grad_phi_u·u。事先固定通过标准absolute error≤1e−7+.05×max(abs(FD),abs(AD))；失败原样保留，不放宽容差。完整场no_grad与局部autograd可能触发既有不同attention内核及query chunk形状，FP32抵消/量化也可影响微小FD，需同时保留local-vs-full normal scalar差，并解释不能自动当作断梯度。局部梯度仅当前输入附近响应，不证明信息可逆或提高某响应会提高3D质量。

全部30预测用原projector与target原A/H测量评价relative L1/L2，不用source目标，不计算GT误差。输出/大baseline中间tensor/checkpoint留外部；Git存协议/配置/身份/清单/源码hash/逐阶段标量/贡献/局部梯度/FD/账本/图/报告。必要图包括：各阶段正常/干预响应、初末测量贡献、latent/logit/C敏感性、固定case normal/zero/wrong的XYZ切片（诊断预测，不称正确羽流）。

执行前先提交本协议、配置、checkpoint身份、固定输入/模式清单与源码hash。执行后确认所有参数和buffers前后hash一致，所有hooks移除、.grad全None、optimizer steps=0，source/history/data/checkpoint hashes不变。必要测试覆盖字段/身份、贡献重构、共享调用次序、非等长chunk归约、梯度变量、访问边界及适用模型回归。明确实现错误先报告影响与修复方案，不静默改被冻结模型或重训。OOM仅可减少诊断批量/chunk，不改模型或丢射线，不干扰他人进程。

最终用证据回答八个定位问题；允许无法明确定位。不能把跨case差异当测量依赖，不能把sigmoid小导数本身当失败根因。只选一个下一阶段假设，本轮不实现、不调loss/增益/latent、不训练300轮。更新README，经7897代理push后停止。
