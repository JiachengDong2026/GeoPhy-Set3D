# 监督Ray-Senseiver稳定性→正式训练→冻结评价（训练前协议）

本轮保留有监督候选，历史纯自监督REPAIR_FAILED_NO_GO与历史FP32 consistency回归失败不变。原始254913参数架构、完整64³/全部valid射线含精确零、无新残差/lifting/3D卷积、原测量编码/归一化/GT64预处理。GT只交给监督NMSE及独立评价，不进入model。seed20261011、FP32、AdamW lr2e-4/wd1e-4/clip1，有效batch2。独立目录、全部fresh，不恢复任何旧科学终止run。

阶段A使用历史过滤train序号0–7、[26,27,45,42]，直接复用历史8例GT均值初始化与weight×.01。唯一新配置：按更新cosine T_max4000/eta_min2e-5，最多4000更新/1200秒。调度和预算都不同，历史比较仅共同前2000步，不单独归因调度。每100步评估8例，每200做0/1/2测量置零和donor1/2/0；固定几何/path/mask不变。平均NMSE≤.2、relativeL1≤.4、较平均模板NMSE下降≥30%、至少2例relativeMA≥.001，且3例wrong测量NMSE增量均>0，连续3个200步联合检查通过即科学停止并进入B。至少1000步后每200联合检查用最近/前200步训练NMSE，连续2次改善<2%且未通过则平台退出；明确拟合差(NMSE≥.98)且无依赖才FAILED_TO_LEARN，已有空间/测量学习则UNRESOLVED_STABILITY。预算未通过同样按真实状态归档，不续跑。

阶段B在A通过后自动登记并fresh启动，预算从开始固定为300epoch/7200秒。使用官方166train、36Val、Pool A，复用监督MRGR原ViewSampler，训练3–6站/epoch零基变化、shuffle，Val4站/epoch0。仅完整train GT64估计bias，Val统计不用于初始化。逐case归一化voxel MSE是唯一loss，与冻结监督MRGR的SmoothL1+.1relativeL1不同；不能声称完全同条件或纯架构因果。cosine按epoch T_max300/eta_min2e-5，每轮完整Val，唯一best selector=Val平均逐case NMSE。实际最低值每次都保存best；独立显著改善anchor累计下降≥.5%才重置计数，至少50轮且连续30轮无此改善早停。至少30轮后，train与Val末10轮均≥.98，最近两个相邻10轮改善均<2%且确认当前无测量依赖，停止FAILED_TO_LEARN。每20轮固定train0/1/2同几何normal/zero/wrong检查；至少40轮、连续2个health间Val改善<2%且无两例依赖，停止STALLED_WITHOUT_DEPENDENCE。出现空间学习时不套用历史measurement阈值。

所有阶段：NaN/Inf、配对/轴序/梯度/更新错误立即工程暂停，保留最后可靠checkpoint；只修明确工程错误并版本化，不改科学配置。OOM最多一次batch1/accum2，完整ray/query不变；再次OOM或资源不足保存工程暂停，不杀他人进程。完整保存optimizer/scheduler/RNG/更新及epoch游标/早停计数/累计墙钟，显式只从本阶段last恢复，科学停止禁止恢复。时间上限在下一更新前检查，收尾Val/保存可多出必要开销但不能追加更新。

B只有有效学习时冻结best：至少最后Val NMSE<.98、train NMSE<.98、峰值/空间变化非退化、至少2/3固定train依赖且wrong测量误差逐例报告、三组梯度/更新有效。该门槛只用于是否作为有效学习模型评价，不是重新选epoch，最佳epoch始终原Val NMSE唯一selector。冻结best后先完成measurement-only的全部900预测保存与hash封存，再独立GT64评价，禁止重选epoch/调参。严格复用V4五档×五layout×36Val清单和旧指标/36case配对bootstrap(10000,seed2026101801)，比较监督MRGR及自监督Ours/DeepSets，标签、训练loss、selector/采样/预算分别标注。计时固定case0/18/35、G1/G3/G5 draw0、10warmup/30重复；共享GPU说明局限。参考已有冻结标量可复用，不重新训练参考。GT128、test、Pool B始终禁止。

本阶段A登记先提交源码/配置/身份/hash后实际运行；B实现、完整train/Val白名单、train均值、源码/初态和900清单在B启动前另行提交冻结登记。A源码不因B实现而修改。B结束后自动冻结/评价，若工程环境不能继续，报告真实“工程暂停/仍运行”状态，不写已完成。
