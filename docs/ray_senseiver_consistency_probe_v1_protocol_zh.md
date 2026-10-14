# Ray-Senseiver consistency 开/关：最后一次小预算loss对照 v1

研究问题：当前固定结构、输入编码和尺度初始化下，consistency是否影响测量依赖与非平凡输出形成？上轮relative关闭仅缓解输出压低，两臂均FAILED_TO_LEARN；consistency梯度较大不等于方向冲突，不预设去除后成功。

唯一干预：两臂内部relative_l1=0，Control = hold + .5cross + .1observed + .05原consistency，Treatment仅前三项。直接复用上一轮独立包装合法组件，新增显式consistency系数。原selfsup/FROZEN_LOSS/旧validator/模型/投影器全部不改，无monkey patch。beta .01、epsilon1e-6、Y_SCALE32284.894580078173、原mask/归约/相对误差分母不改。原relative及consistency继续计算诊断，但Treatment total不引用两者。无ray weighting、删零测量、监督、TV、激活或其他科学配置改变。

复用原注册source-index train前8例，Pool A，seed20261011，sampler始终epoch0，A3–5/B3–5/H2两两不交，case顺序与连续2例batch固定，全部tensor identity核验。只允许这8例projection；禁GT64/128、Val、Pool B、test、外来checkpoint。H只进入loss，绝不输入network，因此这是训练子集可学习性探针，不是独立泛化验证。

不重新估计尺度，直接复制原scale与fixed_inputs清单。C_init=.0015582415568457682，输出层weight×.01，bias=−6.462637843768738。两臂fresh全参数逐元素相同，必须匹配注册state SHA a51ae38bb2d67202a4766de25c0e01bc9832fb8887756d2f7ebbee41a38fc5be。禁任何smoke/已训练/失败checkpoint warm start。协议、配置、源码、数据/历史/初态hash先提交，运行时逐项核验HEAD已提交注册字节。

阶段一：不创建optimizer，全部8例同一fresh模型完整预测，只forward/autograd，state hash不变、.grad均None。分别求外层加权measurement SmoothL1、.05consistency、Control/Treatment total的loss、encoder（不含latent）/learned latents/decoder梯度L2、完整输出dL/dC统计和均匀平移导数。记录参数空间/输出空间分别的dot和cosine，以及各组||g_m+g_c||、||g_m||²+||g_c||²+2dot，不能直接相加范数。Control与上轮relative-off Treatment同预测loss分量bitwise相等，输出和全参数梯度数值一致；Control−Treatment及梯度差=.05consistency，梯度余量容差2e−6+2e−5×reference，loss容差2e−8+2e−4相对值。

共同幅度探针仅α=1/.1/.01，同时缩放同一C_A/C_B；原projector、consistency公式/epsilon保持。记录四分支loss与dL/dα。consistency=αd/(αm+epsilon)，其导数d×epsilon/(αm+epsilon)²≥0，意味着epsilon使共同缩小也能降该项；这与压小A/B差异的方向须分清，不能由此直接断定完整训练因果。不会搜索α或按诊断改设定。

工程smoke每臂独立临时fresh模型1更新，验证H不入网、完整64³/XYZ query、全valid rays/几何、所有组非零有限梯度及真实更新、自身状态保存恢复、固定259 query与chunk37输出及梯度一致性（沿用既有容差）。之后科学两臂重新fresh初始化。诊断及smoke提交后才启动科学训练。

每臂最多40epochs、4updates/epoch、160steps，effective batch2（允许OOM改batch1/accum2）、FP32 AdamW lr2e−4/wd1e−4/clip1/cosine T_max300。完整64³=262144 query、全部valid ray（含零测量）、原P_diff/K96。只允许工程query chunk/checkpoint/device调整并记录，不减ray/query，不动他人进程。只从本臂last.pt恢复model/optimizer/scheduler/RNG/history/health/事件/退出计数，科学已终止禁止恢复。

保存epoch0完整状态。每轮保留train聚合和轮末固定8例统一H/cross/observed relative L1/L2、SmoothL1、原relative/consistency诊断、两种目标分量、输出min/max/mean/std/case差、各step梯度与真实更新、时间/显存。每5轮和epoch0健康检查：全8例logit分位数、<-12比例与sigmoid导数；固定case0/1/2空间邻点差、case差、同图梯度分支/dot/cosine、正常与置零测量差；只置零raw/normalized测量，geometry/path/mask不变。依赖relative MA=absolute MA/(normal meanabs+1e−12)，至少2例≥1e−3；同时保留绝对MA。每健康检查query/chunk梯度一致性。

退出规则完全复用上轮版本，无阈值调整：“第5/20轮后”严格epoch>5/>20，首次可检查10/25。

- NaN/Inf、梯度缺失、query/轴序错、丢valid、访问违规或无真实更新，立即工程停止；保留最后有效checkpoint及错误。明确错误先报告影响，再版本化重新登记，不静默改科学run。
- 连续两次满足H L1≥.98、>95% logit<−12、mean<.01C_init，NEAR_ZERO_COLLAPSE（最早15）。
- 第20轮后连续两次满足最近5轮轮末平均H L1≥.98、相比前5轮L1及L2改善均<1%、有效依赖不足2例，确认更新正常后FAILED_TO_LEARN（最早30）。窗口、计数和条件完整记录；L2持续改善或有效依赖形成不因L1单独慢而误停。
- epoch40无前述停止为BUDGET_COMPLETE。

只在共同epoch/update配对比较统一指标，不横排不同total。各自终点另列。GO仅Treatment最后3健康H L1均<.90，无工程/塌缩，固定case至少一对relative MA≥1e−3（原操作标准），至少2例测量依赖，三组梯度和更新有效。不证明3D正确，也不要求达到Ours精度。若只有SmoothL1/L2改善而无case/测量依赖，明确有限指标改善（沿用上轮辅助改善≥1%作描述标签），正式训练仍NO-GO。

本轮是最后一次逐项loss排查。若失败，结束loss搜索，只建议一个后续任务：独立检查测量信息能否通过当前ray-token编码与latent bottleneck被有效利用；本轮不实现。即使GO，也只冻结候选和正式训练建议，不自动训练300轮。不扫描beta/lr/relative/consistency，不改GO阈值/预算，不把失败Senseiver列成功强基线，不归为原Senseiver能力上限。Ours/DeepSets、历史源码/数据/报告/结论保持。Git存代码/配置/注册/标量/曲线/中文报告/审计；大checkpoint在外部独立目录。完成经本机7897代理push后停止。

入口：`python scripts/run_senseiver_consistency_probe.py --phase prepare` → 提交 → `--phase diagnose` → `--phase smoke` → 提交 → `--phase train --arm control` / `treatment`。只读最终审计和标量报告使用本轮独立脚本。
