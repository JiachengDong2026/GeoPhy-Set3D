# 64³ Representation Capacity Audit v1

## 1–2. Why this audit / current C64 mismatch

既有全train/Pool A audit显示：K96 P_diff(C_GT128)对saved K256 reference的pooled relative L1≈0.0958%、L2≈0.3157%；naive endpoint-aligned GT64的L1≈1.3993%、L2≈3.0778%。
operator mismatch、naive downsample mismatch、64³ representation capacity floor是三个不同概念。
本轮直接优化64³体积，只检验measurement representability，既不证明唯一重建，也不将有限优化结果称为exact theoretical lower bound。

## 3–5. Protocol, cases/stations, optimization

只用12 train cases，按frozen source_index的train顺序evenly-spaced indices=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]；未按测量误差挑选案例、未做复杂难度分层、无val/test或Pool B实验。
复用EvaluationViewSampler的deterministic farthest-point spread逻辑：先从64个Pool A stations选16 fit，再从剩余48个选16 held-out；所有case共用同一geometry IDs。
fit IDs=[49, 10, 58, 19, 18, 2, 29, 25, 63, 33, 38, 3, 37, 43, 41, 32]，azimuth span=173.574°；held-out IDs=[45, 22, 59, 60, 62, 26, 56, 36, 53, 11, 15, 13, 46, 61, 23, 35]，span=169.689°。
目标严格是saved projections_train.npy = P_ref(C_GT128,K256)，ppm·m；没有生成新reference或使用P_ref(C_GT64)。
GT64直接读取现有gt_volume_64.npy，GT128仅用于operator reference阶段；所有数组/source hashes核对。

C_opt=sigmoid(Z)，Z0=logit(clamp(GT64,1e−6,1−1e−6))；FP32/cuda:3，Adam lr0.01、eps1e−12，最多2000 steps。
Adam的小epsilon用于已归一化、按valid rays平均、低浓度sigmoid变量的微小梯度，不改变measurement objective；没有optimizer/lr搜索。
冻结P_diff：grid_sample、align_corners=True、midpoint K96、mass-to-mole转换、path length与valid mask，源码未修改。
loss=mean_valid SmoothL1(Y_pred/32284.894580078173,Y_ref/32284.894580078173)，beta1，仅fit rays；没有voxel supervision、TV、sparsity、mass penalty或任何网络。
200 steps内best fit relative L1改善不足1e−5 absolute即early stop；保留最低fit loss iterate，held-out从未用于优化、early stop或iterate selection。
Naive阶段使用未clamp的原GT64；sigmoid初始化的epsilon背景影响另存initial_fit_relative_l1，未混入naive结果。

## 6–8. Operator / naive64 / optimized64; fit and held-out

| Stage | Subset | L1 mean / median / max (%) | L2 mean / median / max (%) |
|---|---|---:|---:|
| operator | fit | 0.117307 / 0.091369 / 0.239603 | 0.358041 / 0.299856 / 0.634566 |
| operator | heldout | 0.116345 / 0.090787 / 0.249504 | 0.357030 / 0.310540 / 0.658998 |
| naive64 | fit | 1.656998 / 1.438944 / 3.194456 | 3.530893 / 3.507193 / 5.335952 |
| naive64 | heldout | 1.664341 / 1.454509 / 3.158366 | 3.601724 / 3.468664 / 5.564314 |
| optimized64 | fit | 0.624358 / 0.495559 / 1.535897 | 1.366283 / 1.291783 / 2.488037 |
| optimized64 | heldout | 1.577390 / 1.040984 / 4.552113 | 2.677928 / 2.328676 / 6.395537 |

以上为mean/median/max per-case ratios；summary.json另外保存按原始valid-ray sums累计的pooled指标。
逐case与summary.csv完整保留raw ppm·m MAE、RMSE、bias(prediction−saved reference)、max absolute error和relative L1/L2，所有分母只含valid rays。

| Train index | Fit naive → optimized L1 (%) | Held-out naive → optimized L1 (%) | Steps / best step | Recovery (fit) |
|---:|---:|---:|---:|---:|
| 0 | 1.458413 → 0.174366 | 1.466292 → 0.509190 | 803 / 803 | 0.931715 |
| 15 | 1.216545 → 0.121698 | 1.243185 → 0.605289 | 944 / 944 | 0.957921 |
| 30 | 1.220694 → 0.393630 | 1.222990 → 0.746152 | 565 / 565 | 0.720468 |
| 45 | 1.063605 → 0.324123 | 1.074132 → 0.713891 | 564 / 564 | 0.740149 |
| 60 | 2.192513 → 0.973787 | 2.170274 → 2.581703 | 558 / 558 | 0.601246 |
| 75 | 1.105389 → 0.510512 | 1.136529 → 0.982908 | 525 / 525 | 0.577332 |
| 90 | 2.573282 → 1.535897 | 2.575185 → 3.362847 | 504 / 504 | 0.444528 |
| 105 | 3.194456 → 1.353910 | 3.158366 → 4.552113 | 574 / 574 | 0.619611 |
| 120 | 1.119854 → 0.437871 | 1.111954 → 0.813635 | 561 / 561 | 0.648563 |
| 135 | 1.419475 → 0.480606 | 1.442726 → 1.099060 | 511 / 511 | 0.712887 |
| 150 | 1.843807 → 0.659061 | 1.874811 → 1.740372 | 581 / 581 | 0.693577 |
| 165 | 1.475946 → 0.526831 | 1.495654 → 1.221520 | 570 / 570 | 0.692836 |

recovered_fraction=(E_naive64−E_opt64)/(E_naive64−E_operator)，以relative L1计算；fit mean=0.695069，median=0.693206。
它可能超过1：有限ray拟合可以低于GT128的operator mismatch，operator结果是比较参考而非数学下界；未将recovery截断到[0,1]。
best fit case={'case_id': 'case1_g1_transient_scheme_a_v1_2s_p0600_w1_equal', 'relative_l1': 0.0012169840230177183}；worst fit case={'case_id': 'case2_streamwise_g2_transient_scheme_a_v1_2s_p0300_w3_equal', 'relative_l1': 0.015358967508661718}；fit L1>1%的cases=['case2_streamwise_g2_transient_scheme_a_v1_2s_p0300_w3_equal', 'case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet2_half']。

## 9. Voxel plausibility diagnostics

active fraction固定定义为C>1e−4 mass fraction；total concentration proxy=sum(C)，只是网格proxy，不称真实气体质量。
voxel similarity只作diagnostic，不用于PASS条件或优化。以下均按12个最终体积汇总：

```json
{
  "volume": {
    "min": {
      "mean": 1.5532142919975436e-07,
      "median": 1.4619753585520812e-07,
      "max": 2.2667238397389156e-07,
      "min": 1.2202488619550422e-07
    },
    "max": {
      "mean": 0.7139783973495165,
      "median": 0.7448585629463196,
      "max": 0.9333125948905945,
      "min": 0.4362289607524872
    },
    "mean": {
      "mean": 0.0036074304967805737,
      "median": 0.003974199198206127,
      "max": 0.006527829444876479,
      "min": 0.0014062346639600281
    },
    "active_fraction": {
      "mean": 0.11223506927490234,
      "median": 0.10608482360839844,
      "max": 0.22506332397460938,
      "min": 0.036197662353515625
    },
    "total_concentration_proxy": {
      "mean": 945.6662601480467,
      "median": 1041.812474614547,
      "max": 1711.2313219976998,
      "min": 368.6359797491376
    }
  },
  "difference_vs_gt64": {
    "relative_l1_vs_gt64": {
      "mean": 0.08547052904827696,
      "median": 0.056968554981114715,
      "max": 0.2655935493355841,
      "min": 0.04012485024635556
    },
    "relative_l2_vs_gt64": {
      "mean": 0.1927789986673449,
      "median": 0.17125619130488917,
      "max": 0.4031111094508438,
      "min": 0.12415587496969675
    }
  },
  "implausible_case_ids": []
}
```

补充screens：max>0.9且>5×GTmax、GT≤1e−4背景中超过25%升至>1e−4、平均邻居差分>3×GT且>1%×GTmax；这些是粗略诊断，不是可证明的物理合理性标准。
没有加入TV variant；主结果保持measurement-only，B3仍应保留supervised voxel loss。仅首个选定train case执行同一配置zero-init sanity：初始fit L1=100.018049%，最终=100.148480%，held-out=100.153087%，steps=200。L1下降=False；它不参与主Gate分类，也不证明更长优化或其它初始化必然失败。

## 10–11. Capacity conclusion and B3 recommendation

**FAIL**。NO-GO for strong 64³ projection-domain constraint. This finite optimizer run does not establish a theoretical representation floor; the owner should decide tolerance or another physics representation. 本次12/12个case的fit L1下降，held-out均值仍为1.577390%，indices=[60, 90, 105]还比其naive结果更差，因此不满足本轮整体Gate。它反映有限优化与held-out兼容性限制，不能单独归因为64³表达能力不足。

工程Gate要求fit mean和median均<0.5%，held-out均<1.0%，且两个subset的mean/median均优于其naive结果；held-out在1%–1.5%以内则为CONDITIONAL PASS。不是理论阈值或统计显著性检验。
本轮只测试12个GT64-initialized train volumes、16 fit+16 held-out stations，FP32误差远小于本Gate尺度；结果不证明所有64³体积都能精确表示GT128，也不证明网络能自动找到这些解。
GT64初始化提供很强的空间先验，少量zero-init诊断不能取代这一限定；voxel非唯一性仍存在。
本次正式case在504–944步按fit-L1 plateau停止。优化目标为SmoothL1，最低loss iterate不必同时具有最低relative L1；这些有限优化结果不能据此证明不可消除的capacity floor。没有为获得PASS追加lr、loss、初始化或regularizer搜索。

下一阶段：NO-GO for strong 64³ projection-domain constraint
本轮没有实现B3，没有改成128³网络，也未重新生成dataset、训练B0/B1/B2或研究其它inverse methods。

## Evidence and integrity

Audit implementation commit见implementation_manifest.json；报告生成时commit=`823a685f49dedd9bc4959d39f16d34c45c33e6fb`。
Ran 63 tests in 17.148s，全部OK。完整性：

```json
{
  "dataset_unchanged": true,
  "dataset_files": 1262,
  "b0_b1_b2_checkpoints_and_old_artifacts_unchanged": true,
  "protected_model_config_projector_report_sources_unchanged": true,
  "selected_source_gt128_unchanged": true,
  "projector_source_unchanged": true,
  "test_model_evaluation": false,
  "pool_b_used": false,
  "passed": true,
  "audit_implementation_unchanged_during_run": true,
  "audit_numerical_implementation_unchanged": true
}
```

Test未用于模型/volume优化或评估，dataset inventory只读取hash用于immutability；Pool B没有使用。
输出：`/data/djc/GeoPhy-Set3D/outputs/audits/representation_capacity_64_v1`，含protocol、selected cases/stations、per_case_metrics.jsonl、summary.json/CSV、optimized_volumes、logs与两张图。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/audit_64_representation_capacity.py --device cuda:3
# 已完成结果只重新生成报告：
python scripts/audit_64_representation_capacity.py --report-only
```
