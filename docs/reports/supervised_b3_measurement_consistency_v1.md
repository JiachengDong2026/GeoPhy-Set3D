# B3: measurement-domain consistency regularization

## 1. B3 motivation

检验低权重测量一致性是否帮助 supervised 三维重建。B3 完全复用 B2 GeoLift-GCSI-Res3D：两者均319042参数，encoder/lifting/set interaction/GCA/decoder/output未变。

## 2. 64³ capacity audit implication

冻结 audit：operator L1≈0.117%，naive64≈1.657%，optimized-fit≈0.624%，held-out≈1.577%。这不构成理论容量下界，但排除了强等式约束的实验依据。

## 3. Model-error-aware design

The differentiable projection loss is used as a soft measurement-domain regularizer rather than an exact equality constraint, because the frozen 64³ representation exhibits non-negligible discrepancy against 128³-generated reference measurements.

保留 SmoothL1(beta1)+0.1×sample-relative-L1；没有PDE、TV、sparsity或label-free训练。

## 4. Observed-view loss

L_obs=per-case valid-ray SmoothL1(P_diff(C_pred64,G_input)/32284.894580078173, saved Y_input_raw/32284.894580078173, beta0.01)，batch取均值。仅使用有效射线；合法零测量保留、invalid及padding排除、all-invalid贡献0。冻结投影器K96、align_corners=True、midpoint、mass-to-mole/path length不变。

## 5. Held-out-view loss

Pool A V3–6 input+1 disjoint hold。保持旧hold0 sampler全部input station IDs，然后用独立stable blake2 seed从补集取hold。seed/epoch/index/case决定采样，不依赖worker/global RNG。Held-out raw/geometry只参与loss，不进入网络。Validation fixedV4输入与B2一致，独立fixedhold1仅作诊断。

## 6. Lambda gate

每组从同一个冻结B2 best开始，20 epochs、AdamW lr5e−5/weight_decay1e−4、FP32/batch2/clip1/seed20261011，固定epoch20评价，screening checkpoint不是正式B3。

| Candidate | λobs | λhold | Voxel L1 | Voxel L2 | Observed L1 | Held-out L1 | Physics/sup grad | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| weak | 0.05 | 0.025 | 0.101825760 | 0.110018695 | 0.053159654 | 0.059266202 | 0.015613 | False |
| medium | 0.1 | 0.05 | 0.102286047 | 0.110649280 | 0.053478644 | 0.059475134 | 0.031225 | False |
| strong | 0.2 | 0.1 | 0.102433977 | 0.110489399 | 0.053372451 | 0.059315424 | 0.062451 | False |

B2 baseline voxel L1/L2=0.098906572/0.106766953，observed/held L1=0.052551609/0.057912163。Selected=None。

预先冻结规则：voxel恶化>.005淘汰；held-out/observed必须改善；combined physics gradient>2×sup淘汰；eligible中held-out最大改善，差≤.001视为接近并取较小λ。完整gradient norms见lambda_gate/summary.json，均finite/nonzero，reference/geometry detached。

| Candidate | Supervised | Weighted observed | Weighted held-out | Combined physics |
|---|---:|---:|---:|---:|
| weak | 0.232406318 | 0.0022174099 | 0.00143474294 | 0.00362848584 |
| medium | 0.232406318 | 0.00443481933 | 0.00286948611 | 0.00725697167 |
| strong | 0.232406318 | 0.00886963867 | 0.00573897222 | 0.0145139433 |

## 7. Formal training protocol

通过gate后仅固定一组λ，从同seed随机初始化独立训练300 epochs，166train/36val、Pool A only，AdamW lr2e−4/wd1e−4，cosineTmax300，batch2/FP32/clip1，无early stop。Epoch1–20 λ0；21=0、40=target线性ramp；41+target。最小fixedValA supervised loss选best，projection不用作selection。PoolB仅完整训练/冻结best后评价，test不推理。

Gate未通过，按要求停止；未运行正式300 epochs。下列正式B3指标不适用，不能把gate结果冒充正式结果。

## 8. Val-A reconstruction

冻结B2历史reference（本轮未重新运行Pool B），正式B3未训练：

| Model | MAE | RMSE | L1 | L2 |
|---|---:|---:|---:|---:|
| B2 | 0.0003289512528 | 0.001631771536 | 0.09890657171 | 0.1067669526 |
| B3 formal | N/A | N/A | N/A | N/A |

## 9. Val-B reconstruction

冻结B2历史reference（本轮未重新运行Pool B），正式B3未训练：

| Model | MAE | RMSE | L1 | L2 |
|---|---:|---:|---:|---:|
| B2 | 0.0003239379437 | 0.001727856549 | 0.0967994521 | 0.1141240144 |
| B3 formal | N/A | N/A | N/A | N/A |

## 10. V3/V4/V6

冻结B2的5 deterministic draws历史reference，未评估正式B3：

| Condition | B2 L1 | B3 formal L1 |
|---|---:|---|
| V3 | 0.147768762 | N/A |
| V4 | 0.101159708 | N/A |
| V6 | 0.083671518 | N/A |

## 11. Projection-domain metrics

本轮同协议重评B2 Val-A，正式B3指标不适用。只用raw ppm·m valid rays；mean per-case relative errors。

| Domain | B2 L1 | B2 L2 | B3 formal |
|---|---:|---:|---|
| observed | 0.052551609 | 0.061474385 | N/A |
| heldout | 0.057912163 | 0.066549740 | N/A |

## 12. Clustered V4

冻结B2的5 deterministic draws历史reference，未评估正式B3：

| Condition | B2 L1 | B3 formal L1 |
|---|---:|---|
| clustered-V4 | 0.652041832 | N/A |

## 13. Shortcut tests

冻结B2 normal/wrong geometry/zero measurement voxel L1：0.098906572 / 0.861167511 / 1.000556064。

本轮B2 fixed volume/reference的hold geometry sanity：correct/wrong L1=0.057912163/0.642925143。正式B3 shortcuts未运行。

## 14. GO/NO-GO

**B3 NO-GO**。保留B2为当前主模型，不追加λ搜索，由负责人决定下一步。

No candidate passed the predeclared reconstruction, measurement and gradient criteria; see per-candidate deltas for the actual rejection reasons. 不进行额外λ搜索；projection consistency在当前协议下未获得足以进入正式训练的证据。

weak: voxel恶化=0.002919189；held-out改善=-0.001354039；淘汰原因：held-out未改善、observed未改善.

medium: voxel恶化=0.003379475；held-out改善=-0.001562971；淘汰原因：held-out未改善、observed未改善.

strong: voxel恶化=0.003527405；held-out改善=-0.001403261；淘汰原因：held-out未改善、observed未改善.

本轮没有supervised-only同预算fine-tuning对照，不能将变化完全归因于regularizer；结论仅限这个预先定义的screening协议，不推断所有measurement consistency方法均无效。

Ran 79 tests in 20.543s，全部通过，含worker0/2/4与legacy站点、mask/all-invalid/raw units/两条梯度路径/architecture/parameter/warmup/ramp测试。input-hold overlap严格0。冻结projector、B2source/config/checkpoint/report/旧artifact、完整dataset文件hash与inventory均未变。GPU3 all-rayV6+hold1/batch2/K96 smoke无OOM，峰值allocated约2.8GB，未chunk/subsample。

实现及数值产物source commit见checkpoint和environment；本报告生成时HEAD=406910c797aeccdb95c59ddda385621fd1bdc6c8。源manifest冻结训练文件hash；report-only revision与训练数值源分开。GitHub最终同步commit由终端报告给出。

产物：`/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_b3_phys_v1`；comparison/projection/sparse CSV已标注baseline、screening或historical reference阶段；四张正式B2/B3 figures仅在正式run完成时生成，gate NO-GO时不生成正式对比图。测试日志、protocol、inventory、gate metrics/gradients、checkpoint路径均可查。


## 15. Matched supervised-only fine-tuning control

The original gate compared fine-tuned physics candidates against the unfine-tuned frozen B2 checkpoint. A matched λ=0 continuation control was therefore added to separate continued-training drift from the effect of measurement-consistency regularization.

前14章节保留原gate的历史结果与NO-GO记录。本节给出补充control后的最新判断。没有重训weak/medium/strong，没有新λ/beta/loss/architecture实验。

Control从同一个冻结B2 best启动，严格匹配166train、Pool A、V3–6 input+hold1、seed20261011、20 epochs、AdamW lr5e−5/wd1e−4、constant scheduler、FP32/batch2/clip1。仅λobs=λhold=0；训练不调用observed/heldout projection loss或其backprop。保持相同模型319042参数。评价唯一的epoch20 final checkpoint，未从20 epochs选best。

自动重放全部166×20个case/index/epoch的冻结sampler，与三组candidate核对input与hold station IDs；三组共9960次比较，全部一致、重叠0。另核对36个Val-A fixedV4+hold1逐case station IDs和valid ray counts，全部一致。此处重放证明采样协议一致；旧candidate没有保存全部实际batch轨迹。

仅评价Val-A/Pool A；raw ppm·m valid rays，固定K96与同一projection metrics实现。Pool B/test均未使用。

| Model | Voxel L1 | Voxel L2 | MAE | RMSE | Obs L1 | Obs L2 | Hold L1 | Hold L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen B2 | 0.09890657171 | 0.1067669526 | 0.0003289512528 | 0.001631771536 | 0.05255160908 | 0.06147438495 | 0.05791216309 | 0.0665497395 |
| FT-control lambda0 | 0.102151361 | 0.1107233413 | 0.0003403310242 | 0.00169518161 | 0.05376393279 | 0.06259799644 | 0.05967526157 | 0.06795313417 |
| weak | 0.1018257602 | 0.1100186949 | 0.0003393936448 | 0.0016856173 | 0.05315965398 | 0.06194560081 | 0.05926620196 | 0.06748395204 |
| medium | 0.1022860471 | 0.1106492801 | 0.0003403671258 | 0.001693007315 | 0.05347864449 | 0.06218692662 | 0.05947513441 | 0.06748380203 |
| strong | 0.1024339766 | 0.1104893986 | 0.000341437527 | 0.00169296984 | 0.05337245101 | 0.06190283829 | 0.05931542404 | 0.06735987595 |

Candidate vs frozen B2表示20 epoch continuation后的整体变化，包含continued-training与regularization共同作用；candidate vs λ=0 control表示这个匹配实验中regularization的增量影响。两者均保留，旧candidate-vs-B2结果见第6/14节。单seed、CUDA grid_sample backward非逐bit确定性；不做复杂显著性检验或普适因果结论。

Control − frozen B2：voxel_l1=+0.003244789, voxel_l2=+0.003956389, observed_l1=+0.001212324, heldout_l1=+0.001763098。

Δ=candidate−control，负值为candidate更好；relative improvement=(control−candidate)/control，正值为candidate更好。完整8指标delta与relative improvement见control/candidate_vs_control.csv。

| Candidate | Δvoxel L1 | Δvoxel L2 | Δobs L1 | Δhold L1 | Voxel L1 improvement | Obs L1 improvement | Hold L1 improvement | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| weak | -0.000325601 | -0.000704646 | -0.000604279 | -0.000409060 | +0.3187% | +1.1239% | +0.6855% | False |
| medium | +0.000134686 | -0.000074061 | -0.000285288 | -0.000200127 | -0.1318% | +0.5306% | +0.3354% | False |
| strong | +0.000282616 | -0.000233943 | -0.000391482 | -0.000359838 | -0.2767% | +0.7281% | +0.6030% | False |

36个case逐对比较，change=candidate−control；abs(change)≤1e−12归为相等，无显著性检验。

| Candidate | Domain | Improved cases | Worsened cases | Equal cases | Mean paired change | Median paired change |
|---|---|---:|---:|---:|---:|---:|
| weak | voxel_l1 | 22 | 14 | 0 | -0.000325601 | -0.000242952 |
| weak | heldout_l1 | 24 | 12 | 0 | -0.000409060 | -0.000422390 |
| medium | voxel_l1 | 16 | 20 | 0 | +0.000134686 | +0.000217078 |
| medium | heldout_l1 | 21 | 15 | 0 | -0.000200127 | -0.000350612 |
| strong | voxel_l1 | 16 | 20 | 0 | +0.000282616 | +0.000505210 |
| strong | heldout_l1 | 23 | 13 | 0 | -0.000359838 | -0.000399904 |

评估前冻结规则：voxel L1 worsening≤.001，observed L1 worsening≤.001，held-out明确改善定义为相对下降≥1%；多个eligible优先held-out改善最大，差≤.001则weak>medium>strong较小λ优先。

本轮最低held-out L1 candidate为weak，相对control的held-out改善=0.6855%。Control本身相对冻结B2有continued-training drift；三组candidate的held-out/observed mean L1均略优于control，weak的voxel mean L1也略优于control。按预先定义的1%工程阈值，改善幅度不足以通过gate；这个NO-GO不构成regularizer导致原有恶化或完全无增量收益的证据。

weak: held-out improvement below predeclared 1%

medium: held-out improvement below predeclared 1%

strong: held-out improvement below predeclared 1%

**Revised B3 gate: definitive NO-GO**。no candidate meets the predeclared measurement gain and voxel preservation criteria。B2 remains the current main model。不追加λ搜索，不正式训练B3，不修改projector或增加projection loss。下一阶段建议network architecture innovation；本轮未实现。

Ran 89 tests in 21.160s，全部通过。数据/B2/projector/旧weak-medium-strong source-config-checkpoint-metrics-artifacts冻结hash均未变；原summary全部字段值与原报告正文保留。控制实验产物：`/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_b3_phys_v1/lambda_gate/control`，包括checkpoint、training log、effective config、metrics、per-case paired CSV、sampling与integrity验证。Control源码commit见environment/checkpoint；最终GitHub同步commit由终端给出。
