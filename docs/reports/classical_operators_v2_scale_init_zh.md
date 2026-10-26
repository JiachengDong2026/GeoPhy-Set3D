# DeepONet / FNO 输出尺度感知初始化 v2 验证

本轮只检验一次确定性的现有参数初始化干预。两个模型独立判定；本报告不构成正式精度排名。
最终结论：DeepONet **NO_GO**；FNO **NO_GO**。每模型完成30轮/480次更新；后续正式训练资格取决于所有原门槛共同通过。

## 冻结与实验身份

基于远端 main `279814b5152c3916913766e7a0fb28134e7d320d`；训练前直接 GitHub ls-remote 与本地 origin/main 一致。独立分支 `experiment/classical_operators_v2_scale_init`；工作区 `/scratch/djc/GeoPhy-Set3D-classical-operators-v2-scale-init`。原 main 因目录迁移显示的删除未暂存；原目录符号链接保持，迁移后文件与 Git blob 已核对。
既有 v1 发布清单 88 项、6 个 v1/主模型/DeepSets checkpoint 身份核验通过；原有 4924 项跟踪文件逐项哈希未变。DeepSets 300epoch/24900updates 与主模型继续冻结。旧主模型 C0/C1 固定输入输出逐位回归通过。

## v1 只读尺度审计

| 模型 | 初始 Val H | 末态 Val H | 原始更新证据 | 主要失败 |
|---|---:|---:|---|---|
| DeepONet | 135.985614 | 约 1 | checkpoint 96；进程仅知 96–111 | 近零塌缩，第7轮梯度范数检查中断 |
| FNO | 137.864663 | 0.989392991 | 完整持久化 480 | 饱和与 Val 置零收益不足 |

固定正常诊断场的 v1 初始均值约为 DeepONet 0.48427、FNO 0.49325，logits 均接近 0。DeepONet 末态正常场平均均值约 6.58e-20，logit 均值范围 -76.12..-45.08，饱和比例 1；FNO 末态正常场平均均值约 8.01e-5，logit 均值约 -13.62..-13.22，最高饱和比例 0.970848。完整 168 个初/末场的均值、std、min/max、logit 分位数与两端饱和、空间 CV、近零比例保存在 v1_scale_audit.json。train/Val 全基线、曲线、正常/置零/错配和历史独立复核也一并封存。
v1 初始 smoke 的实际梯度范数约 41.706（DeepONet）、25.786（FNO），均有非零 AdamW 更新；这是历史 A5/B5 smoke，不能与本轮不同采样 CPU batch 的范数做严格数值因果比较。DeepONet 保存末态真实梯度约 3.68e-12；不能据此断言中断瞬间的唯一机制。epoch0 评估记录 gradient_norm_max=0 不代表零梯度。
这些证据同时包含初始输出偏大和后续塌缩/饱和；不能将输出尺度认定为唯一或已经证明的主要原因。

## 只由训练测量确定的 c0

c 是 CH4 质量分数，[0,1]；原始 Y 单位 ppm·m。a=1/16.043，b=1/28.965，χ(c)=ac/[b+(a-b)c]，χ′(c)=ab/[b+(a-b)c]²。常量场的物理投影为 10^6 L χ(c)。Y_SCALE=32284.894580078173 是投影归一化，不是浓度尺度。实际原始 K96/grid_sample 算子和 ExplicitRayOperator 均已通过常量场数值核验，误差见 initialization_calibration.json。
仅使用官方 train166、seed20261011、epoch0 的确定性 A/B。H 集合只用于验证不相交，不索引 H 测量。投影整文件字节用于 SHA 身份核验；计算只提取 A/B 有效射线。无 Val/test/GT/Pool B/外来权重进入尺度估计。
固定 x=10^6 L/Y_SCALE，y=Y/Y_SCALE；通过 FP64 80次二分求 Σx·clip((xq-y)/0.01,-1,1)=0，这是等射线权 Huber 常量匹配。保留所有合法零测量和极大值，不做观测截断/剔除。非法有效射线立即失败。
共有 2,977,913 条有效射线，零测量 401,366 条；q=0.0003478588565711439，c0=0.0001927003566465826，z0=-8.55418141283；根方程残差 -1.8626e-09。有效长度范围 3.3244532e-07..0.83407873 m。中心满足 |z0|+0.1<=10。该常量只确定初始尺度，不是重建场或真实三维浓度统计。

## 唯一初始化干预与 CPU 测试

在查看 c0 与残差结果之前，commit 30a5e0f 已冻结估计器、固定 train0..3 A/B 校准样本和 alpha=min(1,0.1/maxabs(residual))。DeepONet 仅缩放 branch 最后 Linear 的 weight/bias，并设置原 scalar bias=z0；FNO 仅缩放最终输出卷积 weight，并设置该层原 bias=z0。相同 v1 类实例，无新增参数、模块或 forward 改写。

| 模型 | 参数 | α | 初始场均值（固定 train0 A） | CPU 真实梯度范数 | CPU AdamW 更新 L2 |
|---|---:|---:|---:|---:|---:|
| deeponet | 222033 | 0.5494241551 | 0.00018612255 | 0.027477826 | 0.092044792 |
| fno | 4196657 | 1 | 0.00018457115 | 0.02122445 | 0.028217738 |

全 64³、batch2 CPU 测试通过：参数/输出同种子逐位重现、允许参数白名单、有限物理范围、必要近零/饱和条件、padding/invalid NaN、H 不进入模型、正常/置零/recipient-station 错配、真实有限非零梯度与 AdamW 更新、序列化逐位一致。每模型 CPU 测试实际更新1次，独立丢弃；GPU pilot 从全新初始化开始。GTReadGuard 用合成路径负例验证；真实访问审计未出现任何 GT/外来 checkpoint 读取。全部有效射线计数另由 CPU 输入/显式 stencil 复核。
初始输出尺度明显下降且未饱和，但这不等于条件重建成功或梯度传播必然改善。两个模型都必须通过终态空间变化/条件响应/测量收益门槛；未把较小初始梯度解释成更好的梯度。

## 固定 pilot 及科学验收

严格复用原 objective/run_epoch/selection_metric、采样与 K96 物理：train0..31，Val36；FP32 batch2；seed20261011；AdamW lr2e-4/wd1e-4，clip1，cosine T_max300；每轮16步、30轮、480步。每步更新 journal fsync，每轮含 optimizer/scheduler/RNG 的 checkpoint 与 SHA；初始化 checkpoint；诊断0/10/20/29/30。独立 NumPy FP64 从保存完整场进行三线性 K96 积分，复核投影、H 误差、置零/错配响应与全部门槛。每模型210个完整场，主实现与独立积分的干预门槛均一致。

| 模型 | 初始 Val H | e29 Val H | e30 Val H | e30 Val 场均值 | 实际更新 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| deeponet | 1.03309618 | 1.00000092 | 1.00000085 | 5.323023e-09 | 480 | NO_GO |
| fno | 1.03256019 | 0.981881968 | 0.986214495 | 8.07547736e-05 | 480 | NO_GO |

FNO 的 e29/e30 验证误差虽均低于0.99，但相较真实epoch0仅改善约4.908%/4.488%，不足5%；最大饱和比例约0.96160/0.96510，仍超过95%。e30四个Val案例A/B等权的正常优于置零H收益约6.83e-5、2.79e-4、5.12e-5、1.23e-4，均未达到0.001。它有达到阈值的训练置零/错配响应和Val输出响应，但未形成足够的Val测量收益。DeepONet末态绝对置零响应仅约1e-11量级，不能用相对响应掩盖近零输出。

两个末轮的门槛逐项如下（不使用中途最好 checkpoint 替代）：

| 模型/epoch | 全分支健康 | train置零 | train错配 | Val响应+置零收益 | Val≤0.99 | 初始改善≥5%且≥0.01 | Val健康 |
|---|---|---|---|---|---|---|
| deeponet/29 | 失败 | 失败 | 失败 | 失败 | 失败 | 失败 | 失败 |
| deeponet/30 | 失败 | 失败 | 失败 | 失败 | 失败 | 失败 | 失败 |
| fno/29 | 失败 | 通过 | 通过 | 失败 | 通过 | 失败 | 通过 |
| fno/30 | 失败 | 通过 | 通过 | 失败 | 通过 | 失败 | 通过 |

输出/响应不能由 loss 下降替代。A/B等权固定 donor；Val 保留 v1 更严格的同一案例同时满足响应和置零收益条件。各案例数值见 *_interventions.json 及 *_numpy_projected_interventions.json。输出健康覆盖正常、置零和错配全部分支，比 v1 代码仅检查正常分支更严格，已在训练前协议明确。

## 登记与工程限制

保留全部原始失败日志：准备脚本第一次未给 collate 提供空 hold 字段，在任何模型/参数更新之前修复；CPU 测试第一次未正确保留不同视角数的 batch padding，在任何 optimizer 更新之前修复；launcher helper common 名冲突，在 GPU 分配之前通过独立模块名修复。注册保留 initial 与逐次修正前副本。另一次独立尺度审计最初错误地直接使用原始 FP64 几何，而非模型可见的 FP32 几何，导致选值哈希断言失败；保留日志后按原 Dataset._views 精度复核通过，未重算或修改 c0。均无 c0/alpha 规则、网络、优化或阈值变更；没有试跑候选尺度。
初始登记遗漏 Val36 投影逐文件 SHA；事后全部与冻结数据构建 manifest 核对一致，见 val_identity_supplement.json。该补充不能冒充训练前登记。训练前标识仍包括 source_index/geometry/statistics 与所有166训练文件。此项是登记完整性缺陷，报告不宣称工程登记毫无遗漏；必要科学门槛失败本身已足以 NO_GO。
当前异常处理支持捕获异常/SIGTERM后原子保存模型、实际更新计数、optimizer/scheduler/RNG 和访问审计；无法保证断电或 SIGKILL 的处理器执行，本次两个 pilot 均正常完成，无失败恢复或训练重放。

## 结论与边界

尺度初始化改善了初始物理量级，使 DeepONet 能在本次预算内持续执行真实更新；这不证明最初的输出尺度是唯一原因。只要任一必要条件失败，就不具备正式训练资格。若两模型均 NO_GO，则本轮不支持“仅初始化足以解决失败”的强假设；仍可能涉及测量稀疏/可辨识性、目标对近零解的优化吸引或架构适配，但本轮没有把这些替代机制逐一识别为原因。
本轮没有正式300epoch、新900预测、GT64/GT128、Pool B 或 test 真值读取，没有新算法、超参数/尺度搜索，没有恢复v1或修改主模型/DeepSets。没有合并main。迁移后的 /data/djc 数据与现有链接保持原状；大型checkpoint/数组留在scratch，不加入Git。

## 交付索引

- 协议：docs/classical_operators_v2_scale_init_protocol_zh.md。
- 机器登记、源文件/数据 SHA、c0、初始化分布、v1审计、CPU日志、曲线、干预与独立复核、checkpoint清单、资源、最终状态、交付哈希：docs/reports/artifacts/classical_operators_v2_scale_init/。
- 独立注册：src/geophy_set3d/training/classical_operators_v2_scale_init.py；配置：configs/classical_operators_v2_scale_init/；执行/复核：scripts/classical_operators_v2_scale_init/。
- 全62份初始化/逐轮checkpoint与全场NPZ：/scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init/{deeponet,fno}/；每文件SHA见checkpoint_manifest.json与diagnostics.json。
- 原始完整资源快照和每步日志：上述scratch根目录 scheduling/、gpu_snapshots.jsonl、各模型pilot/。
