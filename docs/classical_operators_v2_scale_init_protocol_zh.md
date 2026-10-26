# DeepONet / FNO v2 输出尺度初始化：训练前协议

基线为 origin/main 279814b5152c3916913766e7a0fb28134e7d320d。独立分支 experiment/classical_operators_v2_scale_init；所有大文件位于 /scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init。保护全部既有源码、配置、记录和 checkpoint；不合并 main。

## 唯一尺度规则（计算前冻结）

仅使用官方 train split 全部 166 个案例、seed 20261011、epoch 0 的原 DisjointABHSampler 选出的 A/B。每个分支每条有效射线计一次，不访问 H 数值。包括合法零观测，不剔除大观测；任何非有限值、负测量或非正有效长度立即失败。文件身份可整文件哈希，但估计只索引 A/B。Val/test/GT/Pool B/外来权重均不得用于校准。

原始 Y 为 ppm m；c 为 [0,1] CH4 质量分数；a=1/16.043，b=1/28.965，chi(c)=a*c/(b+(a-b)*c)，chi'(c)=a*b/(b+(a-b)*c)^2。常量场的原始投影为 1e6*L*chi(c)。Y_SCALE=32284.894580078173 仅归一化投影，非浓度缩放。

令 x=1e6*L/Y_SCALE，y=Y/Y_SCALE，求唯一稳健常量摩尔分数 q，使 sum x*clip((x*q-y)/0.01,-1,1)=0。固定 FP64 二分 [0,1] 80 次，无候选搜索；这是全射线等权 Huber(beta=0.01) 常量场匹配。c0=b*q/(a-(a-b)*q)，z0=log(c0/(1-c0))。记录射线数、观测分位数、来源 SHA、梯度方程、计算过程。常量场不是三维重建。

残差校准固定 train 索引 0..3 的 epoch0 A/B，全 64³，fresh seed20261011。DeepONet 残差为原 logits 减现有输出 bias，FNO 为原 logits 减最终卷积 bias。M 为所有这些体素残差的最大绝对值。alpha=min(1,0.1/M)（M=0 则 alpha=1）。DeepONet 只将 branch 最后 Linear 的 weight 和 bias 乘 alpha、原输出 scalar bias 设 z0；FNO 只将 readout 最后 Conv3d weight 乘 alpha、其原 bias 设 z0。其余参数逐位不变，参数数目和 forward 均不变。

初始化必要条件：c0 严格处于 (0,1)，所有初始输出有限且在 (0,1)，诊断均值>=1e-8，近零(<1e-10)比例<.95，abs(logit)>10 比例<.95；为避免用随机残差掩盖尺度本身的冲突，要求 abs(z0)+0.1<=10。任何失败停止，不修改估计器/阈值/激活。初始化空间 CV 和条件响应记录但不作初始资格门槛；它们仍是终态原有科学门槛。CPU 必须完成确定性、invalid/padding、H/GT 边界、干预、真实有限非零梯度和 AdamW 更新、序列化、旧 C0/C1 逐位回归，才允许 GPU。

## 已冻结的训练与科学门槛

只有通过前置条件才实施一次 fresh pilot：train0..31、Val36；batch2 FP32；seed20261011；AdamW lr2e-4 wd1e-4；clip1；cosine T_max300；30epoch、每轮16实际更新、最多480。原 objective/run_epoch/selection_metric、输入/物理 K96/采样/损失完全不变。独立 v2 注册与配置，不恢复 v1。初始化和每轮 checkpoint+SHA、optimizer/scheduler/RNG；每次成功 update fsync；失败原子保存实际计数/RNG/访问，无训练重放。epoch0 全 train/Val/零场基线；诊断0/10/20/29/30。

epoch29和30均须通过：训练总loss与H误差改善、有限真实非零更新；Val H<=.99且相对e0改善>=5%、绝对>=.01；全部固定诊断分支均值>=1e-8、近零比例<.95、空间std/mean>=.05、饱和比例<.95；Val跨案例MA/mean>=1e-3。train0/1/2 donor1/2/3，A/B等权：>=2/3零响应relative>=1e-3且absolute>=1e-8，>=2/3错配H增加>=1e-6。Val0..3 donor1..4：>=2/4同时满足零响应relative>=1e-3、absolute>=1e-8、正常H比置零低>=.001（保留v1更严格联合条件）。错配只替换 recipient station 的 raw/normalized 测量，几何/mask/路径不变。独立 NumPy CPU 完整场/投影/干预/门槛复核。

GPU 用已有全局 UUID flock、间隔15秒的双空闲快照和有界等待；不抢占/终止别人任务，资源不足串行或停止，不降分辨率/射线。任一必要条件失败 NO_GO；缺失必要证据 INCONCLUSIVE。前置阶段科学尺度冲突可直接 NO_GO，并明确 pilot 未运行，后续项目不适用。

禁止正式300epoch、900预测、GT64/128/PoolB/test真值、候选搜索、第三模型、修改主模型/DeepSets、自动合并或正式训练。GO只报告，待用户后续确认。
