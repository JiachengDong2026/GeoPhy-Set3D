# 可学习几何尺度融合 v2：20轮匹配筛查（seed 20261018）

状态：三臂均完成20轮、每臂1,660次更新。联合主gate：**未通过，不晋级正式长训**。所有误差为留出投影相对L1，不是三维场误差；未读取三维GT或最终35例。

## 设计与公平对照

修复上一版固定几何嵌入没有独立可退化幅度的问题：新增零初始化5→3可学习投影，直接加到尺度softmax logits。D输入原OARI五维descriptor；C输入content激活的前五通道，两者均有757个新增参数、总345,435参数。设置新增投影权重和bias为0时，D严格退化为不额外使用O数值的内容融合。相比固定嵌入，这允许训练自行决定几何强度和符号。C/D不具有完全相同的函数类；C的额外线性head可代数并入原content logits，改变参数化/优化；D引入新O特征。因此这里只做到参数个数匹配，不能声称有效函数容量严格匹配。

B为已有多尺度344,678参数。B/C/D共享基础初态；C/D完整state初态相同。三个完整初始输出逐位相等。新增残差输出初始为0，首步上游零梯度是预期；一次真实smoke更新后全部8个参数张量均有有限非零梯度。四项CPU测试、16个B1/B2×mask组合通过。

三臂物理GPU1顺序执行；166训练、36验证，seed 20261018，固定第20轮末权重，原300轮cosine前缀、目标、batch2、FP32、Q96、归一化、站位与shuffle一致。完整注册hash `c47ac77d9a55b8878fef5c4a6afd4e00d278cb290c38ccd41dccff04a4c68f57`。未按指标挑选epoch或案例。

| 模型 | 普通均值 | 普通P90 | 聚簇5布局均值 | 热态每轮 |
|---|---:|---:|---:|---:|
| 多尺度MRGR | 9.2592% | 16.1374% | 45.6300% | 33.31s |
| 匹配内容logits融合 | 9.1110% | 16.4715% | 49.5676% | 34.47s |
| 可学习几何logits融合 | 8.9756% | 16.6126% | 51.4628% | 34.22s |

## 配对变化与门槛

36案例成对bootstrap20,000次，负差为改善；困难布局先在案例内平均五布局。区间不包含训练种子不确定性。两个主对照D/B与D/C都要求普通均值至少改善1%、普通P90和聚簇均值恶化不超过2%、耗时比≤1.20、300轮外推≤5小时、正确性和输出非坍缩通过。

| 对照 | 普通相对变化 | 普通差95%CI（百分点） | 普通胜/负 | 聚簇相对变化 | 聚簇差95%CI（百分点） |
|---|---:|---|---:|---:|---|
| C_content_vs_B_multiscale | -1.600% | [-0.4082, +0.1238] | 22/14 | +8.629% | [+2.0964, +5.7412] |
| D_geometry_vs_B_multiscale | -3.063% | [-0.5929, +0.0615] | 28/8 | +12.783% | [+1.9519, +10.2900] |
| D_geometry_vs_C_content | -1.486% | [-0.3932, +0.1304] | 20/16 | +3.823% | [-1.4200, +5.6132] |

- D vs B_multiscale：`{"gate_passed": false, "checks": {"ordinary_mean": true, "ordinary_p90": false, "cluster5_mean": false, "runtime": true, "five_hour": true, "correctness": true, "output_sanity": true}}`。

- D vs C_content：`{"gate_passed": false, "checks": {"ordinary_mean": true, "ordinary_p90": true, "cluster5_mean": false, "runtime": true, "five_hour": true, "correctness": true, "output_sanity": true}}`。


## 结果解释

普通均值的改善没有覆盖尾部和困难几何，因此按预注册规则停止该候选。匹配内容控制C本身也在聚簇布局明显退化，说明整个新增融合/细尺度特征及其优化路径可能已有跨布局适配问题；不能把D相对B的全部困难退化归因于几何条件。D/C困难均值差区间跨0，也不足以证明几何条件独立导致退化。

可能原因包括：随机3–5视角训练与聚簇布局的可观测性分布不同，尺度gate可能依赖训练中的几何—内容相关性；细尺度校正更容易补偿投影而未形成稳健先验；原OARI已经利用相同方向覆盖信息，新增小头没有增加独立观测。以上是待验证解释，不是三维GT或PDE证据。v1与v2首种子和数值轨迹不同，不能把两版绝对分数直接视作架构优劣。

## 同权重实际条件干预

预注册首2验证案例×A/B共4次输入；固定D第20轮权重，开/关新增几何logits，保留原OARI与可见性mask，不更新参数。输出场相对L1差均值 0.050502%，尺度alpha绝对差均值 0.01201380。正常H为 11.034561%，关闭额外几何后 11.030693%。权重hash保持不变。这只能证明条件进入实际输出，不是重训消融或总体收益证据。

普通72次A/B重建的尺度权重均值（32/16/8）：C `[0.31407742036713493, 0.368419016400973, 0.3175035599205229]`，D `[0.32583659928705955, 0.3709719632234838, 0.303191435833772]`；分支残差范数均值分别 16.2424/17.0603。这些只描述gate行为，alpha不是物理贡献率或误差置信度；不同尺度特征和输出映射的幅度也参与最终结果。不能单凭alpha推断峰位或形态改善。

## 解释边界与证据

当前为已知开发验证集上的单种子短程结果；投影改善不保证三维、峰值、质心或物理真实性改善。GPU数值轨迹可产生差异，不将小收益视作已建立的几何因果证据。D的300轮按本轮实测外推 2.851小时，不是完成的训练。联合gate通过后才做第二种子配对复核，由主agent决定正式训练。

![三臂训练曲线](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/curves.png)

- [预注册协议](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/protocol.json)
- [逐案例汇总、配对区间与gate](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/comparison.json)
- [CPU/GPU smoke及梯度](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/smoke.json)
- [同权重条件干预](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/condition_intervention/summary.json)
- [独立算术/文件hash复核](../../outputs/research/mrgr_adaptive_geometry_v2_seed20261018/scalar_verification.json)

每臂目录保留20轮metrics、站位摘要、loader RNG、每例普通误差、180条cluster5记录、输出非坍缩检查、GT guard审计、config、checkpoint hash。60轮/4,980次更新均可复核。算术核验通过：`True`。
