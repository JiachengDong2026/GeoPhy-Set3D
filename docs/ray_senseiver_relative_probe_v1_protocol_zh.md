# Ray-Senseiver relative-L1 开/关：训练前协议 v1

本轮检验measurement loss内部relative-L1是否在极小测量较多时强化整体输出压低方向。不预设唯一因果或关闭必成功。原run与初始化探针均终止并保持历史结果，Ours/DeepSets冻结。新独立配置、objective、训练与诊断入口，不修改原selfsup数值函数、FROZEN_LOSS、原validator或monkey patch。

唯一干预：Control measurement_discrepancy(relative_l1=.1)，Treatment relative_l1=0，分别作用hold/cross/observed。外层hold1/cross.5/observed.1/consistency.05保持，内部beta.01、relative epsilon1e−6、原Y_SCALE、valid mask/归约与consistency公式不变。relative为**逐case sum_valid abs residual / clamp_min(sum_valid abs target,1e−6)**，不是逐ray除target。新增包装直接显式调用已有函数。Treatment只以诊断/共同评价计算relative，训练total图不使用该诊断项。无删零/极小测量、ray weighting、log变换、TV或GT。

完整复用初始化探针注册文件：source-index train前8例/Pool A、seed20261011、固定sampler epoch0的A3–5/B3–5/H2与case顺序/连续两例一batch。全部A/B/H tensor identity和ray清单逐项核验。直接复制原scale.json，不重新估计，C_init=.0015582415568457682；两臂fresh输出weight×.01、bias=−6.462637843768738，全部参数逐元素一致、state hash必须等于原注册Treatment初态。禁止任何训练/smoke/失败checkpoint warm start。H不输入network，仍在训练loss，不是独立泛化Val。仅8例train projection可读，禁GT64/128、Val/test、PoolB和foreign checkpoint。

阶段一先登记和提交协议/配置/清单/初态/源码hash，再进行forward/autograd零step诊断，不创建optimizer。固定全部8例、每例fresh相同尺度模型与非训练常量场C=0/1e−8/1e−7/1e−6/1e−5/C_init，不按结果选择。对同一预测分别求measurement SmoothL1、原始relative及其.1加权、hold/cross/observed、consistency和两种total的loss、encoder/latents/decoder梯度L2、完整输出dL/dC符号/幅度/总和（均匀输出平移方向）和常量scalar dL/dc。正导数表示梯度下降降低浓度；零点abs子梯度可能与正近零不同。输出/参数梯度cosine与dot用于说明抵消，不能把范数直接相加。常量无网络参数，参数梯度记null，不能伪称encoder梯度为0。

每例A/B/H按target/Y_SCALE精确零、(0,1e−6]、>1e−6固定三组，仅诊断观测幅度，无真实背景/羽流标签含义，不参与训练筛选。各组数量/分位数保留；用完整原分母/valid数分解每组smooth/relative loss，并**经真实K96 projector autograd至常量scalar**记录dL/dc。不能换成组内分母或残差符号统计。分组贡献之和与完整分支导数核验（atol2e−4+rtol5e−5）。再用float64均匀场解析y=1e6 length mass_to_mole(c)标量检查，不改变FP32训练物理函数。记录exact-zero与近零差异。

Control与原objective在相同预测的loss各分量及输出/模型梯度核验；两臂loss/梯度差必须等于.1×加权relative项（数值容差梯度L2 2e−6+2e−5×reference）。实际训练前工程smoke每臂独立1步，检查首步全部梯度/真实更新、H不入网、合法自身状态保存/恢复及完整query分块；随后重新fresh初始化科学两臂。smoke不作为初态。

科学每臂最多40epochs，8例/轮、batch2/accum1、4updates/轮、最多160updates；FP32 AdamW2e−4/wd1e−4/clip1/cosine T_max300，原完整64³/query262144、全部valid rays、P_diff/K96。可因OOM只调整工程chunk/checkpoint/device或batch1/accum2并显式记录；不杀其他实验、不减少ray/query。仅从本臂last恢复模型/optimizer/scheduler/RNG/update/history/退出计数；已科学停止不得继续。初态epoch0完整保存。正式前提交全部冻结文件，启动验证HEAD字节和source/data hashes。

每轮保存train过程聚合及轮末固定8例统一H/cross/observed L1/L2（与训练coefficient无关）、smooth/relative诊断项、外层loss项、输出min/max/mean/std/case差，每step三组梯度与真实更新。每5轮和epoch0健康检查：全8例真实postproc logit分位数、<-12比例与sigmoid导数；固定case0/1/2空间XYZ邻点差、case差、两臂目标的梯度分支/dot/cosine及normal-vs-zero（仅raw/normalized测量置零，几何/path/mask不变）。relative MA=绝对MA/(normal meanabs+1e−12)，同时报绝对差；至少2个case≥1e−3为有效依赖。每健康检查259固定query full/chunk37梯度及预测一致性（沿用原预登记绝对+相对数值容差），正式完整输出不变。记录墙钟、GPU event elapsed、峰值显存和真实updates；共享GPU非受控性能测试。

严格“第5轮后/第20轮后”解释为epoch>5/epoch>20，eligible检查分别10/25起。

- 工程异常（NaN/Inf、缺梯度、query漏/轴错、丢valid、访问违规、组无真实更新）立即停止，保存上一次完整有效last和错误记录。明确错误须报告影响、另版本登记重新开始，禁止静默修复原科学run。
- 连续两次检查H L1≥.98、>95%logit<−12、mean<.01C_init，NEAR_ZERO_COLLAPSE（最早15）。
- 第20轮后连续两次：最近5轮**轮末**H L1均值≥.98，相比前5轮L1与L2相对改善均<1%，且本次固定case有效依赖不足2例，实现/更新正常则FAILED_TO_LEARN（最早30）。所有窗口/条件/计数保留。若L2持续改善或依赖已形成，不能仅L1缓慢而误停，继续40上限。

仅在共同epoch/update匹配比较，各臂终点另列。GO：Treatment最后3健康H L1均<.90，无工程/塌缩，固定case至少一对relative MA≥1e−3（沿用上一轮操作标准，分母首个预测meanabs+1e−12），至少2例有效测量依赖，encoder/latents/decoder梯度/更新有效。不是3D正确或Ours精度证据。

若未GO但Treatment共同预算终点相对自身epoch0的H L2或加权measurement SmoothL1改善≥1%（预登记辅助改善操作阈值），标记PARTIAL_LEARNING_EVIDENCE，同时明确正式训练决策仍NO-GO；即使两臂FAILED_TO_LEARN也可有该辅助证据，不掩盖终止分类。若无此辅助改善则NO-GO。需同时列T−C，不能把初态降低或指标偏好改变当非平凡测量学习。不得外推relative-L1对所有模型有害；若后续使用不同loss作正式基线必须披露适配差异。

不根据诊断搜索任何系数/seed/beta/lr/loss组合。仅一次两臂，最多40/160；无论GO/PARTIAL/NO-GO，不启动300轮或追加搜索。只提出一个下一阶段任务。Git提交协议/代码/清单/标量/图/中文报告/审计/README，checkpoint外部独立目录，结束经7897代理push。

入口：`python scripts/run_senseiver_relative_probe.py --phase prepare` → 提交 → `--phase diagnose` → `--phase smoke` → 提交诊断/smoke → `--phase train --arm control` / `treatment`。报告仅用本实验已保存标量。
