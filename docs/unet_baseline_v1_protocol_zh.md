# 标准 3D U-Net v1：训练前协议

## 科学假设与修改边界

H1：普通3D U-Net直接处理测量和几何回投影网格，三维卷积、多尺度和跳跃连接可能改善条件重建。
H2：若近零方向的共享自监督优化、sigmoid饱和等为主要瓶颈，仅替换为U-Net仍可能失败。只读固定常量场诊断提供局部证据，不证明训练失败的唯一原因。

从远端main 279814b5152c3916913766e7a0fb28134e7d320d创建独立 experiment/3d_unet_baseline_v1，不合并main。保护全部旧文件、主模型 uniform_plain_set_interaction_seed20261011、正式GeoLift-DeepSets-Sum与Senseiver/GNOT/DeepONet/FNO（包括独立v2分支）全部结果，前后哈希核验。大文件仅写/scratch/djc/GeoPhy-Set3D/unet_baseline_v1。

## 固定架构（查看任何pilot结果前冻结）

标准五层3D U-Net，编码器通道16/32/64/128/256，每层两次Conv3d(k3,pad1,bias=False)+GroupNorm(8组,affine=True,eps1e-5)+ReLU。MaxPool3d(2)四次。解码器每级trilinear插值至对应skip尺寸(align_corners=False)，Conv3d(k1,bias=True)降通道，skip concatenation，双卷积块；通道依次128/64/32/16。末层Conv3d(16,1,k1,bias=True)输出logits，原sigmoid，[0,1] CH4质量分数。无注意力、MRGR/OARI/GCA或新模块。不作架构搜索。激活checkpoint仅用于逐块内存管理（use_reentrant=False），不改变数学前向或目标，训练与CPU测试固定启用。

输入直接实例化冻结baseline_fno_v1.FixedRayBackprojection：CPU确定性K96线性摩尔分数T，前三通道为(Tᵀy)/(Tᵀ1)（y=raw/Y_SCALE）、log1p(Tᵀ1)、支持>0。实现使用J(0)ᵀ/chi'(0)，所有有效射线保留，invalid/padding NaN在哈希/计算前归零；cache512和stencil128MiB沿用。再拼接XYZ各[-1,1]、ij/zfast坐标共6通道64³。回投影不是浓度真值。H不进网络，无GT/PoolB/test真值/监督权重用于输入。

## 唯一初始化

只读复用已冻结v2 c0=0.00019270035664658265、z0=-8.554181412833294。原始方法：train166、epoch0、seed20261011的A/B有效射线，x=1e6 L/Y_SCALE、y=Y/Y_SCALE；FP64固定80次二分求Σx clip((xq-y)/0.01,-1,1)=0；c0=bq/[a-(a-b)q]，a=1/16.043,b=1/28.965。包含零测量，不剔除异常有效观测；H/Val/test/GT均不参与。必须核对全部166投影及index/geometry/statistics SHA与冻结v2完全相同，否则停止，不重新选择尺度。

输出残差缩放直接沿用v2规则：seed20261011 fresh U-Net，固定train0..3的epoch0 A/B，M=maxabs(初始logits-原末层bias)，alpha=min(1,0.1/M)，M=0则1。只将现有末层weight乘alpha，bias设z0，不增加参数。记录alpha、参数hash及分布；不是多个候选试跑。要求|z0|+.1<=10、预测有限在(0,1)，初始均值>=1e-8、近零比例<.95、饱和比例<.95；初始空间CV和响应只记录，终态必须过科学门槛。

## 训练前只读目标诊断

在网络实现前执行原training.selfsup.objective：固定train0/1与2/3，epoch0官方A/B/H。用可微共享常量浓度0、1e-10、1e-8、1e-6、冻结c0，分别记录total与hold/cross/observed/consistency对独立A/B常量叶子的真实导数，以及投影误差/零场基线。另固定A=c0,B=.5c0检查一致性方向。无optimizer、不选初始化、不改目标。正dL/dc表示梯度下降在该常量方向减小输出；零场处abs/clamp不可微，AD给框架子梯度，不能当作局部全空间结论或网络训练必然结局。可继续一次有界pilot，不据结果修loss。

## 数据、优化和预算

原training.selfsup.objective/run_epoch/selection_metric直接复用；官方PoolA train166、Val36，全部202测量文件及元数据在训练前登记SHA。原sampler、掩码、K96、归约、hold1/cross.5/observed.1/consistency.05、beta.01、relativeL1.1、epsilon1e-6、Y_SCALE32284.894580078173全部不变。

FP32、batch2/accum1、seed20261011、AdamW2e-4/wd1e-4、clip1、cosineT_max300。pilot train0..31/Val全部36，每轮16实际更新，最多30轮/480次。fresh，不warm-start旧模型。CPU真实优化测试权重丢弃；GPU先batch2完整64³显存检查（零更新），不改分辨率/射线。初始完整train/Val与两者零场基线。epoch0/10/20/29/30诊断，初始化和逐轮checkpoint+SHA、optimizer/scheduler/RNG。每次成功update计数/真实梯度和更新范数fsync。异常原子保存当前状态、真实计数、原因、RNG、实际输入ledger与访问审计，不重放更新；不能处理不可捕获SIGKILL/断电，此限制如实保留。

GPU复用现有全局UUID flock，15秒双空闲快照，最多60秒预约，可用任何真正空闲GPU；监测冲突，仅停止自身进程。不抢占、不停止他人、不恢复旧队列。资源不安全则停止，不缩减科学配置。

## CPU与科学门槛（首次GPU更新前冻结）

CPU必须通过完整batch2/64³形状、有限/range、同seed逐位初态与重复输出、checkpoint序列化、invalid/padding/全无效NaN、所有有效射线计数、H只进objective、GT/外来checkpoint拒绝、recipient几何不变的normal/zero/donor原始及归一化值干预、真实非零梯度及AdamW参数变化、旧模型注册/loss/冻结文件不变和主模型C0/C1逐位回归。纯工程失败保留日志修复，不改科学配置。

GO要求epoch29和30逐项同时通过：

- 全loss/梯度/参数/更新有限且真实非零；训练总loss与H较e0改善。
- 完整Val H<=.99，较真正e0相对改善>=5%且绝对>=.01。
- 固定train0/1/2 donor1/2/3，Val0/1/2/3 donor1/2/3/4。所有案例、A/B、normal/zero/wrong_case每个完整场分别mean>=1e-8，nearzero(<1e-10)<.95，空间std/mean>=.05，abs(logit)>10比例<.95。
- A/B等权，每个案例normal-zero relative MA=sumabs(diff)/sumabs(normal)，absolute MA=meanabs(diff)。train至少2/3同时relative>=1e-3且absolute>=1e-8；至少2/3错配H增加>=1e-6。
- Val至少2/4同时满足relative>=1e-3与absolute>=1e-8；另外至少2/4正常H优于置零>=.001。严格按本任务分别计数，保存两集合交集作描述，不混用更严格旧v2联合规则。错配只用donor在recipient站位的raw与相应normalized，保留recipient几何/路径/mask。
- 独立NumPy FP64完整64³/K96三线性物理重投影、H指标、全部场统计与干预/gates一致；工程与访问边界通过。额外记录Val跨案例响应，不新增未要求的科学门槛。

任一已证实必要失败NO_GO；未完成且缺乏可审计失败证据INCONCLUSIVE；不得仅loss下降或H略低于1宣布成功。禁止正式300轮/新900预测/GT64/128/PoolB/test真值、新学习率/架构/尺度候选、修改loss/epsilon或延长预算。GO也只报告，等待后续用户授权。
