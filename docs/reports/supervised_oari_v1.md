# GeoLift-OARI-Res3D supervised v1

## 1. Motivation

本轮问题：显式建模ray-ray geometric relation和local directional diversity，是否优于B2的generic view self-attention，尤其在clustered V4和V3？
优先级固定为clustered V4、V3、Val-B、ordinary Val-A。只做一个随机seed、一套轻量结构和一个训练预算。

## 2. Why B2 is insufficient

B2先让view tokens交换feature，再由GCA pooling融合；它保留per-view geometry，但没有explicit pairwise ray bias或ray-set diversity descriptor。
历史clustered V4中B1 L1≈0.65377、B2≈0.65204，说明generic attention对这一残余问题收益很小。
本轮完整复用冻结B2-300 checkpoint与metrics，不重训B0/B1/B2，不恢复B3 physics loss。

## 3. OARI overview

unchanged SharedEncoder2D → unchanged GeometryLifting@32³ → per-view features/geometry →
ray relations + local directional descriptor → one OARI block → unchanged GCA pooling →
unchanged ResidualDecoder3D → sigmoid concentration [B,1,64,64,64]。
相同seed下encoder/lifting/decoder/GCA初始化逐元素与B2一致；interaction内保留的geometry embedding、QKV、FFN和LayerNorm初始化也一致，单元测试验证。
OARI explicitly encodes the geometric relationship between measurement rays and the local directional diversity of the available view set.

## 4. Ray direction definition

d_v(x)=normalize(voxel_xyz−station_origin_v)，使用lifting实际XYZ网格和真实3D origin，绝不以station azimuth代替direction。
有效集合m=visibility AND view_mask；invalid directions、features、geometry在计算前安全置零。
保留B2 g_v=[α normalized, β normalized, distance/6.5m, azimuth/180°]。

## 5. Pairwise ray geometry

r_ij=[abs(dot(d_i,d_j)), sqrt(max(1−cos²,0)), abs(azi_i−azi_j)/180]，维度3。
当前acquisition范围0–180°，角差没有wrap seam；sign-insensitive crossing angle为acos(abs_cos)/(π/2)，范围0–1。
self-pair保留且精确为[1,0,0]；r_ij=r_ji，geometry bias symmetric，总QK attention无需symmetric。
invalid_i OR invalid_j的descriptor与bias为0。

## 6. Local observability descriptor

S=(Σ_valid d_v d_vᵀ)/max(N_valid,1)，3×3 symmetric structure tensor。
torch.linalg.eigvalsh得到降序λ1≥λ2≥λ3≥0，数值负roundoff截0，再按trace归一化。
o=[λ1_norm,λ2_norm,λ3_norm,λ3_norm/max(λ1_norm,eps),N_valid/V_input]，维度5。
V_input=view_mask.sum，排除padding但保留locally invisible input views，从而同时描述coverage并保持padding invariance。
N_valid=0时S/o严格0，无NaN；single ray自然接近[1,0,0]，不伪造isotropy。
当前CUDA/cuSOLVER不支持过大3×3 batched eigvalsh，因此仅将solver batch分为4096个voxel；数学定义不变。
几何是固定acquisition data，descriptor在no_grad内计算，避免退化/重复eigenvalue对station坐标的无意义求导；learned embeddings/bias正常求导。
称为local directional observability descriptor或ray-set geometric diversity descriptor；不声称Fisher information、严格tomography observability或inverse-problem condition number。
synthetic clustered/spread/single-view均finite且clustered/spread descriptors不同；不把λ3高低当成所有3D ray sets的科学验收条件。

## 7. Ray-relational attention

g:4→16→16沿用B2；o:5→8+SiLU；concat[feature32,geometry16,observability8]后Linear56→32。
每voxel独立处理unordered V≤6 set；一层pre-norm d32/h4/head_dim8，FFN32→64→32+SiLU。
logits_ij^h=Q_i^h K_j^h/√8 + φ_h([r_ij,o])。
φ=Linear8→16+SiLU+Linear16→4，最后weight/bias均zero-init；没有新depth、spatial/global/cross-voxel attention或view-index positional encoding。
zero-init第一步pairwise final layer gradient非零；第一次update后hidden layer gradient也非零，测试和gate确认。

## 8. Masking and permutation

invalid key logits=−∞、invalid values=0；invalid query/invalid pair weights和residual后context严格0。
全invalid voxel先以安全零logits替换空集合，再把weights清零；interaction output与pooled feature为0，decoder prediction仍按原decoder定义。
参数在views间共享，physical geometry同步permutation产生equivariant context与invariant prediction。
trained GPU FP32 prediction permutation max diff=3.27825546e-07；padding max diff=2.08616257e-07，均要求≤1e−5。
context equivariance max diff=9.53674316e-07。
attention row-sum error=2.38418579e-07；GCA error=2.38418579e-07。
all-invalid safe=True；mask correct=True；bias mask=True。

## 9. Parameter count

B2=319,042；OARI=319,558；absolute increase=516，relative increase=0.161734%。
这是lightweight增量，未做parameter-matched capacity control；本轮对比不能严格分离capacity与各geometry分支的因果贡献。
OARI正式训练peak allocated=2.658GiB，reserved=3.152GiB；
observed mean epoch time：B2=8.128s，OARI=8.906s。这不是独立受控throughput benchmark。

## 10. Quick gate

只运行一次4-train-case fixed V4/epoch0，120 epochs，FP32/AdamW constant lr2e−4，无结构调参。
loss 12.7082968 → 0.0585262366；
relative L1 125.860659 → 0.584973863。
prediction范围[3.16901424e-05,0.278705209]；case prediction最大pairwise MAE=0.00710038655。
all gradients finite=True；ray-relational gradient非零=True；hidden gradient非零=True；observability gradient非零=True。
pairwise bias finite=True；gate passed=True。正式训练重新随机初始化，未fine-tune B2或gate checkpoint。

正式协议完全与B2相同：166 train/Pool A/V3–6，36 val/Pool A/fixed V4/同station IDs；seed20261011。
FP32、batch2、AdamW lr2e−4/weight_decay1e−4、clip1、cosine T_max300、完整300 epochs、无early stopping。
L=SmoothL1(beta1)+0.1×sample-relative-L1；没有L_obs/L_hold、projection/TV/sparsity/PDE。
minimum fixed Val-A-V4 supervised loss选择best；Pool B在完整训练结束和best冻结后仅评估。

## 11. Val-A / 12. Val-B

| Model | Condition | MAE | RMSE | Relative L1 | Relative L2 | Best epoch | Best val loss |
|---|---|---:|---:|---:|---:|---:|---:|
| B2-GCSI-300 | Val-A-V4 | 0.000328951253 | 0.00163177154 | 0.0989065717 | 0.106766953 | 288 | 0.00989229028 |
| B2-GCSI-300 | Val-B-V4 | 0.000323937944 | 0.00172785655 | 0.0967994521 | 0.114124014 | 288 | 0.00989229028 |
| OARI-300 | Val-A-V4 | 0.000318948119 | 0.00157575463 | 0.0936943054 | 0.102252169 | 278 | 0.00937094984 |
| OARI-300 | Val-B-V4 | 0.000304358372 | 0.00154879068 | 0.090659654 | 0.101354021 | 278 | 0.00937094984 |

指标为mean per-case，MAE/RMSE是原始mass-fraction尺度；逐case和pooled结果完整保留。
best epoch在表中为1-based，checkpoint/log为0-based。OARI单case评估核对重现training best validation metrics。

## 13. V3/V4/V6 / 14. Clustered V4

| Model | V3 L1/L2 | Random V4 L1/L2 | V6 L1/L2 | Clustered V4 L1/L2 | Spread V4 L1/L2 |
|---|---:|---:|---:|---:|---:|
| B2-GCSI-300 | 0.147768762 / 0.186404201 | 0.101159708 / 0.115919500 | 0.083671518 / 0.091319521 | 0.652041832 / 0.795646310 | 0.090885146 / 0.101514764 |
| OARI-300 | 0.136302304 / 0.171246484 | 0.097138745 / 0.113509298 | 0.080064644 / 0.087622656 | 0.565044783 / 0.722468837 | 0.089940354 / 0.104809349 |

每个随机V3/V4/V6与clustered V4为5个deterministic draws，原EvaluationViewSampler/seed20261011，与B2逐case/draw station IDs一一对应。
spread V4复用既有protocol，仅draw0；B2 metrics直接复用，无新增baseline训练或推理。
clustered_v4_per_draw.csv与clustered_v4_per_case.csv保存两模型各draw/case metrics；对应完整JSONL保存IDs、azimuth和pool。
普通fixed V4与5-draw random V4不是同一subset。

| Condition | Metric | Absolute improvement | Percentage points | Relative improvement % |
|---|---|---:|---:|---:|
| Val-A-V4 | relative_l1 | +0.005212266 | +0.521227 | +5.2699 |
| Val-A-V4 | relative_l2 | +0.004514784 | +0.451478 | +4.2286 |
| Val-B-V4 | relative_l1 | +0.006139798 | +0.613980 | +6.3428 |
| Val-B-V4 | relative_l2 | +0.012769994 | +1.276999 | +11.1896 |
| V3 | relative_l1 | +0.011466457 | +1.146646 | +7.7597 |
| V3 | relative_l2 | +0.015157718 | +1.515772 | +8.1316 |
| V4 | relative_l1 | +0.004020963 | +0.402096 | +3.9749 |
| V4 | relative_l2 | +0.002410202 | +0.241020 | +2.0792 |
| V6 | relative_l1 | +0.003606873 | +0.360687 | +4.3108 |
| V6 | relative_l2 | +0.003696865 | +0.369687 | +4.0483 |
| clustered-V4 | relative_l1 | +0.086997048 | +8.699705 | +13.3422 |
| clustered-V4 | relative_l2 | +0.073177473 | +7.317747 | +9.1972 |
| spread-V4 | relative_l1 | +0.000944792 | +0.094479 | +1.0395 |
| spread-V4 | relative_l2 | -0.003294585 | -0.329459 | -3.2454 |

正值代表OARI改善；判断重点始终是clustered V4和V3，不能只用普通Val-A微小变化包装核心创新。
clustered V4有5/5个draw的mean L1改善，范围1.962334至11.498207个百分点。
尽管均值明显改善，OARI clustered mean L1仍为0.565045，不声称已解决退化geometry重建。
spread V4 relative L2恶化0.329459个百分点；本次GO不代表所有condition/metric均改善。

## 15. Geometry-stratified analysis

score为域中心(0.3,0.3,0.3)m真实3D station-to-voxel directions的mean(1−abs_cos)，仅distinct unordered pairs。
这是简单sample geometry diversity score，区别于网络在32³各voxel计算的5维descriptor。
existing V3/V4/V6 random和clustered samples的case×draw结果保存geometry_stratified.csv，比较B2 error、OARI error和B2−OARI improvement。

| Condition | n case×draw | Diversity vs B2 L1 ρ | Diversity vs OARI L1 ρ | Diversity vs improvement ρ |
|---|---:|---:|---:|---:|
| all | 720 | -0.654890 | -0.651528 | -0.367054 |
| V3 | 180 | -0.304439 | -0.315888 | -0.092519 |
| V4 | 180 | -0.310538 | -0.294626 | -0.193757 |
| V6 | 180 | -0.062496 | -0.065496 | -0.015488 |
| clustered-V4 | 180 | -0.499053 | -0.503892 | +0.192787 |

Spearman仅描述性；同case多draw有重复，clustered draws也共享station windows，不作为独立样本，不报告统计显著性或因果解释。
所有条件合并的diversity vs improvement ρ=-0.367054，clustered内部ρ=+0.192787；跨条件的负相关不能解释成每个regime内部都随diversity降低而单调增益。

## 16. Shortcut tests

| Model | Normal L1 | Wrong geometry L1 | Zero measurement L1 | Wrong relation L1 |
|---|---:|---:|---:|---:|
| B2-GCSI-300 | 0.098906572 | 0.861167511 | 1.000556064 | — |
| OARI-300 | 0.093694305 | 0.865844590 | 1.000322006 | 0.117096906 |

wrong geometry使用既有geometry-only循环错位，Y不变；zero measurement只把normalized/raw Y清零，valid/path/geometry保留。
这检查完整模型对measurement和geometry的依赖，不能单独归因于OARI。
wrong relation保持measurements、per-view station geometry和observability不变，仅在interaction内对pairwise关系双轴循环错配。
关系错配使L1恶化2.340260个百分点。该probe不作为硬GO条件；pairwise branch接到logits的独立功能测试另外报告。

## 17. Diagnostics

只保存val indices0/18/35 × spread/clustered draw0，共6个小JSON：IDs、azimuth、pairwise crossing角矩阵、observability、各head attention和final GCA weights。
包含valid-pair voxel平均与一个中心voxel；不保存所有voxel attention，不开展大型3D interpretability。
trained pairwise bias确实改变attention=True，max change=0.633948922；
observability变化改变interaction=True，context max change=0.108226568。
trained全部gradients finite=True；pairwise gradient非零=True；observability gradient非零=True。
attention finite=True；bias abs max=4.04222107；bias symmetry diff=0。
这些诊断证明连接与数值正常，不证明最优view weighting、uniqueness或解决ill-posedness。

## 18. GO / NO-GO

**OARI GO**。本seed下OARI满足冻结工程GO条件，可冻结为Final architecture candidate v1。V3改善+1.146646个百分点，clustered V4改善+8.699705个百分点。

沿用B2工程参考：主表Val-A/Val-B L1相对恶化≤2%；V3或clustered绝对L1改善≥0.005；wrong geometry/zero measurement L1>normal×1.2；mask/permutation/connectivity全部正常。
这些是冻结的operational references，不是统计阈值；L2变化独立完整报告。

```json
{
  "val_a_not_materially_worse": true,
  "val_b_not_materially_worse": true,
  "v3_or_clustered_clear_gain": true,
  "shortcuts_normal": true,
  "mask_permutation_and_connectivity_correct": true
}
```

下一步：Evaluate measurement-residual refinement on top of the frozen OARI architecture.
单seed/36 val cases不能证明跨seed稳定或独立test泛化。本轮没有追加sanity revision、结构搜参或下一阶段模块。

## Reproducibility and integrity

报告生成commit=`e6df4fe018bc1f6189e7e5832f48ec0b4e8d1a0b`；正式训练source commit=`a4de075f51bbce8c4042bed41c314e453b83a259`，training_source_manifest.json保存代码与配置SHA256。
Ran 115 tests in 24.167s，全部OK；保留B0/B1/B2/B3所有原tests。
dataset 1262 files unchanged=True；
B0/B1/B2 checkpoint unchanged=True；model sources unchanged=True；
B3 artifacts unchanged=True；projector unchanged=True；all historical artifacts unchanged=True。
test没有用于训练/推理/selection；integrity仅只读hash其文件。Pool B始终evaluation-only。没有重新生成dataset或改变Pool A/B。

artifacts=`/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_oari_300`，含summary.json、comparison.csv、sparse_views.csv、clustered_v4.csv、per-draw/per-case CSV、geometry_stratified.csv、shortcuts.csv、effective_protocol.yaml、integrity_summary.json、tests.log、diagnostics/和figures/。
五张图：b2_oari_main_metrics.png、b2_oari_sparse_views.png、b2_oari_clustered.png、geometry_diversity_vs_improvement.png、oari_attention_example.png。
checkpoints/logs仍位于外部outputs，不纳入Git。

```bash
python scripts/overfit_supervised_oari.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_oari_300.yaml
python scripts/compare_b2_oari_300.py --prepare
python scripts/compare_b2_oari_300.py --device cuda:3
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
```
已有gate禁止重复覆盖；已有300-epoch实验仅使用自身last.pt和同config恢复，不能恢复B2权重。
