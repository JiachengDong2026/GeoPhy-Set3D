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
