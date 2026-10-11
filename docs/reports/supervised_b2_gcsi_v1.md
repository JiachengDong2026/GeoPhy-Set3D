# GeoLift-GCSI-Res3D / B2 supervised set interaction v1

## 1–2. 科学目标与B1的区别

B1 score_v=f(F_v,g_v)独立判断view重要性；B2先让每个voxel内的views交换信息，再使用相同GCA scorer。
目标是验证context-aware view importance是否改善稀疏输入与冗余/互补geometry，优先级为V3、clustered V4、Val-B、普通Val-A。
没有重新训练B0/B1；复用已经冻结的300-epoch best checkpoints与原始结果。

## 3. Architecture

Y/valid/path → unchanged SharedEncoder2D → unchanged GeometryLifting@32³ →
geometry-conditioned token projection → one per-voxel view self-attention block →
unchanged GeometryConditionedAttentionPooling → unchanged ResidualDecoder3D → sigmoid [B,1,64,64,64]。
只改fusion/interaction；encoder/lifting/decoder/output/normalization保持B1不变。
同seed20261011下，encoder/decoder及GCA pooling初始参数也与B1逐元素一致，单元测试验证。

## 4. Geometry-conditioned token

复用B1 g=[α endpoint-normalized, β endpoint-normalized, distance/6.5m, station azimuth/180°]。
角度来自同一lifting.angular_grid，固定距离/azimuth尺度从frozen acquisition metadata核对。
独立geometry embedding为4→16→16（Conv3D1×1×1+SiLU）；token=Linear48→32([lifted32,geometry16])。
Q/K/V均由这些tokens产生，geometry在interaction前已参与。
geometry enters the set interaction through each view token; no explicit pairwise geometry bias is used in B2 v1.

## 5. Set interaction

每个voxel的tokens为[V,32]；实现为[B×32³,V,32]，没有在voxel间做attention。
一层pre-norm block，d_model32、heads4、head_dim8、无dropout：
X1=X+W_out Attention(W_Q LN(X),W_K LN(X),W_V LN(X))，
X2=X1+FFN(LN(X1))；FFN32→64→32，SiLU。
A_ij^h=masked-softmax_j(Q_i^h·K_j^h/√8)。
F'_v=X2_v，再调用原B1 pooling score(F'_v,g_v)。不存在view-index positional embedding。
本版新增token projection、MHSA和FFN；参数规模小，但本比较并非参数完全匹配的独立容量ablation，不单独宣称因果归因。

## 6. Masks 与 permutation

m=visibility AND view_mask，按voxel/view定义。
inactive features/descriptors在token构造前置零；invalid keys的logit为−∞，invalid values置零；
invalid query rows和所有无效query/key pair的权重严格零；attention与FFN residual后再次mask。
全invalid voxel在softmax前替换为空集合的安全零logits，随后将全部权重清零，interaction output与pooled feature均严格零。
每view共享所有变换；同步permutation使context tokens equivariant、prediction invariant。
GPU FP32 prediction/padding误差要求≤1e−5，CPU synthetic测试更严格。

## 7. 参数与GPU memory

B0=306,625；B1=308,578；B2=319,042。
相对B1增加interaction参数10,464（3.391%）；总量<400k。
真实train/Pool A的B2/V6 FP32前向反向通过，prediction/attention与全部梯度finite，interaction gradient非零。
peak allocated=2.267GiB，reserved=2.502GiB。
naive voxel batching可在RTX4090运行，本轮未加入chunking或改变模型定义。

## 8. Quick gate 与正式协议

一次4-train-case fixed-V4/epoch0 gate，120 epochs、constant lr2e−4。
loss 11.3422167 → 0.053354444，
relative L1 112.390776 → 0.533265676。
全部gradient finite=True；interaction nonzero=True；
attention finite=True；pairwise prediction max MAE=0.00753228739；
permutation max difference=1.78813934e-07；padding max difference=1.49011612e-07。
gate passed=True。只确认快速功能性学习，没有根据gate调参；正式训练从随机初始化重新开始。

正式B2只用seed20261011，166 train cases/Pool A/V3–6/hold0；36 val/Pool A/V4，epoch0/同station IDs。
FP32、batch2/accumulation1、AdamW lr2e−4/weight_decay1e−4、clip1，300 epochs、cosine T_max300，无early stopping。
loss仍为SmoothL1(beta1)+0.1×mean sample-relative-L1，仅gt_volume_64监督。
minimum fixed Val-A total supervised loss选best；Pool B仅在300-epoch训练完成且best冻结后评估。
未加入projection/hold-view/physics/TV/sparsity loss或128³监督，未搜索horizon，也未追加seed。

## 9–10. Val-A / Val-B

| Model | Condition | MAE | RMSE | Relative L1 | Relative L2 | Best epoch | Best val loss |
|---|---|---:|---:|---:|---:|---:|---:|
| B0-300 | Val-A-V4 | 0.000366204519 | 0.00189701317 | 0.108330088 | 0.122842339 | 298 | 0.0108352199 |
| B0-300 | Val-B-V4 | 0.000350901212 | 0.00194462389 | 0.104499652 | 0.125768912 | 298 | 0.0108352199 |
| B1-GCA-300 | Val-A-V4 | 0.000352648292 | 0.00173714184 | 0.103430683 | 0.112474137 | 290 | 0.010344921 |
| B1-GCA-300 | Val-B-V4 | 0.000338253416 | 0.00171533199 | 0.100377647 | 0.111749698 | 290 | 0.010344921 |
| B2-GCSI-300 | Val-A-V4 | 0.000328951253 | 0.00163177154 | 0.0989065717 | 0.106766953 | 288 | 0.00989229028 |
| B2-GCSI-300 | Val-B-V4 | 0.000323937944 | 0.00172785655 | 0.0967994521 | 0.114124014 | 288 | 0.00989229028 |

指标为mean per-case，物理mass-fraction尺度；pooled与逐case完整保留。
B2单case推理重现best training validation指标；best epoch在报告中为1-based，checkpoint/log为0-based。
B0/B1主表与原结果精确复用，checkpoint SHA256与历史artifact hashes均核对。

## 11–12. Sparse views 与 clustered

| Model | V3 L1 | V4 random L1 | V6 L1 | Clustered V4 L1 |
|---|---:|---:|---:|---:|
| B0-300 | 0.164152179 | 0.111192044 | 0.090625252 | 0.698387958 |
| B1-GCA-300 | 0.152993707 | 0.105805935 | 0.085665191 | 0.653769222 |
| B2-GCSI-300 | 0.147768762 | 0.101159708 | 0.083671518 | 0.652041832 |

random V3/V4/V6与clustered V4各5 deterministic draws，EvaluationViewSampler、seed20261011。
三个模型逐case/draw使用完全相同station IDs，程序直接核对保存的JSONL pairings。
random-V4平均不同于主表固定reference-V4；未运行V2/V5/V8或完整condition matrix。
可选spread-V4仅一次既有deterministic selection，三个模型L1分别为：
0.099979957 / 0.094321094 / 0.090885146。
B0/B1只新增了这项低成本spread评估，所有输出写到本轮目录，旧结果未改写。

关键improvement（正值=新模型更好，负值=恶化）：

| Comparison | Condition | L1 absolute improvement | Percentage points | Relative improvement |
|---|---|---:|---:|---:|
| B1-GCA-300 vs B0-300 | Val-A-V4 | +0.004899405 | +0.489941 | +4.5227% |
| B1-GCA-300 vs B0-300 | Val-B-V4 | +0.004122005 | +0.412201 | +3.9445% |
| B1-GCA-300 vs B0-300 | V3 | +0.011158472 | +1.115847 | +6.7976% |
| B1-GCA-300 vs B0-300 | clustered-V4 | +0.044618735 | +4.461874 | +6.3888% |
| B2-GCSI-300 vs B1-GCA-300 | Val-A-V4 | +0.004524111 | +0.452411 | +4.3741% |
| B2-GCSI-300 vs B1-GCA-300 | Val-B-V4 | +0.003578194 | +0.357819 | +3.5647% |
| B2-GCSI-300 vs B1-GCA-300 | V3 | +0.005224945 | +0.522495 | +3.4151% |
| B2-GCSI-300 vs B1-GCA-300 | clustered-V4 | +0.001727391 | +0.172739 | +0.2642% |
| B2-GCSI-300 vs B0-300 | Val-A-V4 | +0.009423517 | +0.942352 | +8.6989% |
| B2-GCSI-300 vs B0-300 | Val-B-V4 | +0.007700200 | +0.770020 | +7.3686% |
| B2-GCSI-300 vs B0-300 | V3 | +0.016383417 | +1.638342 | +9.9806% |
| B2-GCSI-300 vs B0-300 | clustered-V4 | +0.046346126 | +4.634613 | +6.6362% |

improvements.csv同时保存L1/L2 absolute、percentage-point与relative improvements。
V3与clustered的意义优先于普通Val-A的小幅波动，不因单个主表分数改善就宣称学到物理冗余/互补机制。

## 13. Shortcut tests

| Model | Normal L1 | Wrong geometry L1 | Zero measurement L1 |
|---|---:|---:|---:|
| B0-300 | 0.108330088 | 0.863009479 | 1.000360778 |
| B1-GCA-300 | 0.103430683 | 0.870937598 | 1.000403184 |
| B2-GCSI-300 | 0.098906572 | 0.861167511 | 1.000556064 |

zero measurement仅将Y/rawY置零，保留valid/path/geometry；wrong geometry循环错位station geometry与valid/path，Y顺序不动。
它们检查完整模型对measurement和geometry的依赖，不能单独把退化归因于interaction。

## 14. Interaction diagnostics

固定val indices0/18/35，spread/clustered V4 draw0。仅保存各head和全head的voxel平均V×V矩阵、IDs、azimuth及pooling mean weights。
矩阵平均按有效query voxel数量归一化。没有保存训练attention volumes或开展额外interpretability项目。

| Val index | Selection | Max deviation from uniform |
|---:|---|---:|
| 0 | spread | 0.7498709 |
| 18 | spread | 0.74932182 |
| 35 | spread | 0.74908531 |
| 0 | clustered | 0.65034908 |
| 18 | clustered | 0.66425323 |
| 35 | clustered | 0.67194754 |

attention finite=True；query/key mask正确=True；
pooling invalid/padded严格0=True；all-invalid安全=True；
nonuniform=True，max deviation from uniform=0.749870896。
interaction row-sum最大误差=2.38418579e-07，pooling sum误差=2.38418579e-07。
trained prediction permutation max difference=2.68220901e-07；
padding max difference=4.17232513e-07；context equivariance max difference=9.53674316e-07。
controlled synthetic输入包含10°/11°/100°三个geometry tokens：finite=True，
max deviation from uniform=0.62705338。
该sanity不要求100°view权重最高，也不是科学验收阈值。nonuniform矩阵只证明非均匀交互，不能证明已学会去冗余。
单元测试额外证明一个view的feature/geometry改变可以影响另一个view的context，并检查invalid内容不影响valid结果。

## 15. GO / NO-GO 与B3边界

B2满足本轮工程GO检查，建议保留并进入B3。该结论限定于一个训练seed的验证比较。 clustered L1仅改善0.172739个百分点（相对0.2642%），不足以据此声称解决view冗余问题。 Val-B relative L2恶化0.237432个百分点（相对2.1247%）；GO依据冻结的L1工程检查，不代表所有指标改善。

冻结的简单工程参考：主表L1相对恶化≤2%，V3或clustered L1改善≥0.005（0.5个百分点），
wrong geometry/zero measurement L1>normal的1.2倍，mask与permutation正确。
这些是操作性判断，不是论文统计显著性阈值；raw L1/L2变化全部公开。

```json
{
  "val_a_not_materially_worse": true,
  "val_b_not_materially_worse": true,
  "v3_or_clustered_clear_gain": true,
  "shortcuts_normal": true,
  "mask_permutation_correct": true
}
```

下一阶段：加入measurement-domain physics consistency。本轮没有实现B3。
单seed/36-val-case结果不能证明跨seed稳定、统计显著或独立盲测test性能。

## Reproducibility 与 integrity

报告生成时commit=`49002a5734fb42226ee589089c8d6570df5ef38c`；B2训练源码与配置SHA256见training_source_manifest.json，正式训练从已提交的实现启动。
B0/B1保留原训练source/provenance，不修改其配置/权重/报告/旧实验输出。
Ran 56 tests in 17.052s，全部OK。
dataset 1262 files unchanged=True；
B0/B1 artifacts unchanged=True；
B0/B1 model/config/report unchanged=True；
numeric training sources unchanged=True。
test没有用于训练/推理/selection，integrity只读hash test artifacts；Pool B始终evaluation-only。

所有artifacts在`/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_b1_b2_300`：summary.json、comparison.csv、sparse_views.csv、clustered_v4.csv、
spread_v4.csv、improvements.csv、effective_protocol.yaml、integrity_summary.json、tests.log、逐case/draw结果和interaction_diagnostics.json。
四张图：b0_b1_b2_main_metrics.png、b0_b1_b2_sparse_views.png、b0_b1_b2_clustered.png、set_interaction_example.png。
模型权重与完整训练日志保持在外部outputs，不纳入Git。

```bash
python scripts/compare_b0_b1_b2_300.py --prepare
python scripts/overfit_supervised_b2.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b2_300.yaml
python scripts/compare_b0_b1_b2_300.py --device cuda:3
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
```
已有B2实验只允许用其自身last.pt与同一config恢复；没有恢复B0/B1 checkpoint或重训它们。
