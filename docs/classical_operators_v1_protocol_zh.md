# DeepONet / FNO v1：训练前协议

基于本地 main 和缓存 origin/main `3f977a3e31c4e128435bb8caa8556eb6457bd34b`。GitHub 7897 代理拒绝、17890 CONNECT 和直连超时；远端最新性未确认，属于明确环境限制。独立 scratch worktree / branch experiment/classical_operators_v1。第一阶段核验见 history_audit.json：GeoLift-DeepSets-Sum 300轮/24900更新、best 第292轮及900预测逐文件SHA均核验；不重训旧模型。当前主模型为已冻结 Plain seed20261011 SHA 0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c，旧DeepSets报告的MRGR v1 SHA1573…是历史参考，不重命名旧比较表。

## 固定架构与来源

DeepONet：Lu et al., Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nature Machine Intelligence 2021, arXiv:1910.03193。标准 branch/trunk 内积加标量bias后sigmoid。branch对每个完整64×64测量/几何图使用三层stride2 CNN（16/32/32）和64维MLP，再按实际station ID升序放入5个槽，附加view有效标志，flatten后MLP 325→128→64。通道为归一化原始测量、valid、路径长度、入射/出射XYZ和方向XYZ共12。所有有效射线保留，包括零测量；CNN是普通降采样编码，不筛射线。5槽覆盖训练A/B3–5和评价V4；padding不占真实槽，超过5个真实view显式报错。trunk为XYZ→128→128→64的tanh MLP，坐标[-1,1]对应物理[0,.6]，ij顺序XYZ，z最快。8192查询分块并activation checkpoint，始终解码64³。没有注意力、GeoLift或本文模块。

FNO：Li et al., Fourier Neural Operator for Parametric Partial Differential Equations, ICLR2021, arXiv:2010.08895。4层标准三维谱卷积（四个正/负频率象限，rFFT最后轴）、pointwise 1×1通道混合和GELU；宽16、各轴8模态、单侧8点零padding，6→16输入映射、16→64→1输出映射和sigmoid。无MRGR/OARI/GCA/可学习GeometryLifting。

固定输入映射复用已审计ExplicitRayOperator的K96/三线性stencil。原物理P是质量分数转摩尔分数的非线性算子，**不称其存在全局线性伴随**。令T为归一化摩尔分数线积分，则J_P(0)^T/chi'(0)=T^T，使用完全已知常数chi'(0)实现T^T y和T^T1，无浓度先验/GT。输入三通道：T^T(y/Y_SCALE)/(T^T1)（零支持处精确0），log1p(T^T1)，支持二值；另加XYZ共6通道。不是三维真场或其插值。invalid/padding先置零；CPU固定顺序scatter保证重复一致，CPU输入缓存由完整测量/几何/mask字节标识，上限512条约1.5GiB；stencil LRU128MiB。该普通伴随适配不是创新网络。独立NumPy合成射线积分/转置及内积检验必须先通过。

文献用于架构定义；当前网络不可用，不声称本轮已下载和逐段复核论文全文。配置一次固定，无pilot后架构/学习率/loss搜索。

## 数据、目标和公平性

正式配置除独立模型/experiment/output外，与train_deepsets_selfsup_v1.yaml严格逐项一致。Pool A 166train/36Val；官方stateless A3–5/B3–5/H2、Val A4/B4/H4；同归一化、valid/view mask及全部有效射线。H仅进原目标，不传模型；不读取GT64/GT128/PoolB/test/外来权重。严格直接调用training.selfsup.objective/run_epoch/selection_metric，原loss权重1/.5/.1/.05、beta/epsilon不动。训练/选模GTReadGuard继承worker访问审计。

300epoch/24900更新、batch2/accum1、AdamW2e-4/wd1e-4、clip1、FP32、seed20261011、原cosine T_max300、loader4workers/persistent/pin。pilot也沿用T_max300，不改成30。fresh模型与smoke同seed初态hash核对，不加载smoke/pilot/他人权重。

## 工程与有界pilot

先CPU语义/数值测试，再GPU0真实A5/B5/H2 batch2，完整64³前向、原objective、反传、clip和一次实际AdamW更新；检查有限、范围、所有梯度与参数更新、H排除、合法字段、invalid NaN、padding/置换、全invalid、normal/zero/recipient几何上的donor测量、重复推理、序列化。失败日志保留；纯工程错误可有说明的修复，不能改科学设定。

两模型各独立30轮上限pilot。固定官方train索引0…31（32例）、完整36Val，子集选择不看测量/GT；保留原case_id/sample_index sampler，shuffle使用原seed generator。每轮16更新，共480，不是正式24900更新。epoch0完整train/Val原run_epoch基线与零场基线必须记录；每轮完整Val。科学检查在epoch0/10/20/29/30，最后29和30两次共同通过才GO，不能取中途最好权重过关。

门槛预先固定为既有原始适配判据（senseiver_selfsup_decision_v1_protocol）与同目标修复协议更严必要项的交集，避免放宽：

1. 所有loss/梯度/参数/更新有限；真实更新非零；最终train总loss及heldout较epoch0改善。
2. 最后两次完整Val H relative L1均<=.99，且相对真正epoch0至少改善5%、绝对改善>=.01。
3. 固定诊断全部分支mean>=1e-8、近零(<1e-10)体素<95%、空间std/mean>=.05；Val聚合不近零、跨case MA/mean>=1e-3，无持续饱和（>95% logit<-10或>10不通过）。
4. train索引0/1/2、donor1/2/3，固定sampler epoch0；A/B等权：至少2/3例 normal-zero sum(abs diff)/max(sum(abs normal),1e-30)>=1e-3，且绝对MA>=1e-8；至少2/3例错配使H relative L1增大>=1e-6。
5. Val索引0/1/2/3、donor1/2/3/4，同recipient站位替换donor raw/normalized，几何/path/mask完全不动；至少2/4例zero响应>=1e-3且绝对MA>=1e-8，正常H比zero H至少低.001。
6. 所有工程/信息边界审计及第二套CPU数值复核通过，两个模型分别决策。

合成固定几何可辨识性协议使用不同loss与解析GT，其方向/GT门槛不移植到本次GT-free pilot。此范围区分在任何新训练之前确定。任一必要门槛失败即NO-GO；未完成/数值身份无法复核为INCONCLUSIVE。30轮不达标不延长、不改阈值、不自动修复候选；保留best/last/日志。仅GO可fresh正式训练；远端main最新性未确认时正式阶段暂缓，并标明环境限制。

## 正式与评价（仅满足准入时）

正式fresh初始化，保存optimizer/scheduler/RNG/loader generator/每次实际更新和日志，每轮83更新，固定300轮24900更新。不得warm-start，原selector唯一选best；结束后冻结SHA，重放measurement-only selector。全部900条严格复用历史station_subsets.json和完整tensor身份，封存预测和hash后才启动独立GT64进程。复用VolumeMetrics/relative_projection_metrics及case_means/statistics配对bootstrap10000 seed2026101801、36case先五layout平均、95%逐项CI；不把900行当独立样本。不读取最终test。当前主模型必须在相同旧900协议上取得冻结预测后才配对，历史DeepSets对旧MRGR v1表仅历史参照，不冒称当前主模型。

## 资源与保护

只使用GPU0既有全局UUID flock，15秒双空闲快照，60秒有界获取；controller持续冲突监测，只停止自身新session进程组。GPU1–3不使用，不恢复旧queue，不终止他人。home新增worktree因配额失败，改scratch成功；新大数组/checkpoint仅scratch。所有起始tracked文件逐字节保护，包括model registry，不修改任何旧源码或结果。新模型仅在独立training.classical_operators.REGISTRY注册，历史build_model白名单不动。Git只新增源码、配置、协议、标量、身份、审计、报告。所有阶段独立提交且运行前source manifest封存。

工程测试附记（首次GPU/optimizer之前）：首个独立NumPy fixture在y=.2的格点上，FP32与FP64计算使极小支持分别落在格点两侧；coverage归一化后的值/二值支持对这种舍入不连续。保留cpu_tests.log，合成fixture改为y=.213、z=.307，避开该精度边界，未改模型/算子/容差/科学门槛。第二次5项CPU测试全部通过。真实回投影按既有FP32 stencil定义，薄弱覆盖处归一化敏感性作为限制报告。
