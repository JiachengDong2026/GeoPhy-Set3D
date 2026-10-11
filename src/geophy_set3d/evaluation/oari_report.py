"""Report the frozen B2/OARI comparison without overstating geometric evidence."""
import json
import subprocess
from pathlib import Path


def render_oari_report(root, repo):
    root, repo = Path(root), Path(repo)
    s = json.loads((root/'summary.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=repo).strip()
    labels = ('B2-GCSI-300', 'OARI-300'); b2, oari = [s['models'][k] for k in labels]
    gate = json.loads((Path(oari['logs']).parent/'supervised_oari_300_overfit'/'overfit_summary.json').read_text())
    tests = (root/'tests.log').read_text()
    if not tests.rstrip().endswith('OK'):
        raise AssertionError('All tests must pass before completing OARI report')
    test_line = next(line for line in tests.splitlines() if line.startswith('Ran '))
    count = int(test_line.split()[1]); diag = s['interaction']
    parameter_increase = oari['parameter_count']-b2['parameter_count']
    parameter_percent = 100*parameter_increase/b2['parameter_count']
    main = ['| Model | Condition | MAE | RMSE | Relative L1 | Relative L2 | Best epoch | Best val loss |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    sparse = ['| Model | V3 L1/L2 | Random V4 L1/L2 | V6 L1/L2 | Clustered V4 L1/L2 | Spread V4 L1/L2 |',
              '|---|---:|---:|---:|---:|---:|']
    shortcuts = ['| Model | Normal L1 | Wrong geometry L1 | Zero measurement L1 | Wrong relation L1 |',
                 '|---|---:|---:|---:|---:|']
    for label, model in s['models'].items():
        for condition in ('Val-A-V4', 'Val-B-V4'):
            m = model['evaluations'][condition]['mean_per_case']
            values = ' | '.join(f'{m[k]:.9g}' for k in ('mae', 'rmse', 'relative_l1', 'relative_l2'))
            main.append(f"| {label} | {condition} | {values} | {model['best_epoch']} | {model['best_val_loss']:.9g} |")
        values = [model['sparse'][k]['mean_per_case'] for k in ('V3', 'V4', 'V6', 'clustered-V4')]
        values.append(model['spread']['mean_per_case'])
        sparse.append('| '+label+' | '+' | '.join(f"{m['relative_l1']:.9f} / {m['relative_l2']:.9f}" for m in values)+' |')
        values = [f"{model['evaluations'][k]['mean_per_case']['relative_l1']:.9f}" for k in ('Val-A-V4', 'wrong_geometry', 'zero_measurement')]
        values.append(f"{model['evaluations']['wrong_relation']['mean_per_case']['relative_l1']:.9f}" if 'wrong_relation' in model['evaluations'] else '—')
        shortcuts.append('| '+label+' | '+' | '.join(values)+' |')
    improvements = ['| Condition | Metric | Absolute improvement | Percentage points | Relative improvement % |',
                    '|---|---|---:|---:|---:|']
    for r in s['improvements']:
        improvements.append(f"| {r['condition']} | {r['metric']} | {r['absolute_improvement']:+.9f} | {r['improvement_percentage_points']:+.6f} | {r['relative_improvement_percent']:+.4f} |")
    correlations = ['| Condition | n case×draw | Diversity vs B2 L1 ρ | Diversity vs OARI L1 ρ | Diversity vs improvement ρ |',
                    '|---|---:|---:|---:|---:|']
    for condition, result in s['geometry_analysis']['correlations'].items():
        values = [result['rho'][k] for k in ('b2_relative_l1', 'oari_relative_l1', 'l1_absolute_improvement')]
        correlations.append('| '+condition+f" | {result['n']} | "+' | '.join('undefined' if x is None else f'{x:+.6f}' for x in values)+' |')
    decision = 'OARI GO' if s['retain_oari'] else 'OARI NO-GO'
    key_gain = {r['condition']:r['improvement_percentage_points'] for r in s['improvements'] if r['metric'] == 'relative_l1'}
    if s['retain_oari']:
        interpretation = ('本seed下OARI满足冻结工程GO条件，可冻结为Final architecture candidate v1。'
            f"V3改善{key_gain['V3']:+.6f}个百分点，clustered V4改善{key_gain['clustered-V4']:+.6f}个百分点。")
        if key_gain['clustered-V4'] < .5:
            interpretation += ' clustered收益未达到0.5个百分点参考线，因此不声称已解决clustered redundancy。'
        next_step = 'Evaluate measurement-residual refinement on top of the frozen OARI architecture.'
    else:
        failed = ', '.join(k for k, v in s['go_criteria'].items() if not v)
        interpretation = f'本seed下未满足冻结工程GO条件：{failed}。保留B2作为冻结主模型；不将OARI包装为最终核心创新。'
        next_step = '保留冻结B2；本轮不追加搜参、multi-seed或measurement-residual refinement。'
    relation_l1 = oari['evaluations']['wrong_relation']['mean_per_case']['relative_l1']
    normal_l1 = oari['evaluations']['Val-A-V4']['mean_per_case']['relative_l1']
    relation_delta = relation_l1-normal_l1
    if relation_delta > 0:
        relation_text = f'关系错配使L1恶化{100*relation_delta:.6f}个百分点。'
    else:
        relation_text = f'关系错配没有恶化L1（变化{100*relation_delta:+.6f}个百分点），不能用该shortcut声称relations带来实测收益。'
    source = json.loads((root/'training_source_manifest.json').read_text())
    text = f'''# GeoLift-OARI-Res3D supervised v1

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
trained GPU FP32 prediction permutation max diff={diag['prediction_permutation_max_difference']:.9g}；padding max diff={diag['padding_max_difference']:.9g}，均要求≤1e−5。
context equivariance max diff={diag['context_permutation_equivariance_max_difference']:.9g}。
attention row-sum error={diag['interaction_weight_sum_max_error']:.9g}；GCA error={diag['pooling_weight_sum_max_error']:.9g}。
all-invalid safe={diag['all_invalid_safe']}；mask correct={diag['interaction_mask_correct']}；bias mask={diag['pairwise_bias_mask_correct']}。

## 9. Parameter count

B2={b2['parameter_count']:,}；OARI={oari['parameter_count']:,}；absolute increase={parameter_increase:,}，relative increase={parameter_percent:.6f}%。
这是lightweight增量，未做parameter-matched capacity control；本轮对比不能严格分离capacity与各geometry分支的因果贡献。
OARI正式训练peak allocated={s['runtime']['OARI-300']['peak_allocated_bytes']/2**30:.3f}GiB，reserved={s['runtime']['OARI-300']['peak_reserved_bytes']/2**30:.3f}GiB；
observed mean epoch time：B2={s['runtime']['B2-GCSI-300']['mean_epoch_seconds']:.3f}s，OARI={s['runtime']['OARI-300']['mean_epoch_seconds']:.3f}s。这不是独立受控throughput benchmark。

## 10. Quick gate

只运行一次4-train-case fixed V4/epoch0，120 epochs，FP32/AdamW constant lr2e−4，无结构调参。
loss {gate['initial']['loss']['total']:.9g} → {gate['final']['loss']['total']:.9g}；
relative L1 {gate['initial']['metrics']['mean_per_case']['relative_l1']:.9g} → {gate['final']['metrics']['mean_per_case']['relative_l1']:.9g}。
prediction范围[{gate['prediction_min']:.9g},{gate['prediction_max']:.9g}]；case prediction最大pairwise MAE={gate['max_pairwise_prediction_mae']:.9g}。
all gradients finite={gate['all_gradients_finite']}；ray-relational gradient非零={gate['pairwise_gradient_nonzero']}；hidden gradient非零={gate['pairwise_hidden_gradient_nonzero']}；observability gradient非零={gate['observability_gradient_nonzero']}。
pairwise bias finite={gate['pairwise_bias_finite']}；gate passed={gate['passed']}。正式训练重新随机初始化，未fine-tune B2或gate checkpoint。

正式协议完全与B2相同：166 train/Pool A/V3–6，36 val/Pool A/fixed V4/同station IDs；seed20261011。
FP32、batch2、AdamW lr2e−4/weight_decay1e−4、clip1、cosine T_max300、完整300 epochs、无early stopping。
L=SmoothL1(beta1)+0.1×sample-relative-L1；没有L_obs/L_hold、projection/TV/sparsity/PDE。
minimum fixed Val-A-V4 supervised loss选择best；Pool B在完整训练结束和best冻结后仅评估。

## 11. Val-A / 12. Val-B

{'\n'.join(main)}

指标为mean per-case，MAE/RMSE是原始mass-fraction尺度；逐case和pooled结果完整保留。
best epoch在表中为1-based，checkpoint/log为0-based。OARI单case评估核对重现training best validation metrics。

## 13. V3/V4/V6 / 14. Clustered V4

{'\n'.join(sparse)}

每个随机V3/V4/V6与clustered V4为5个deterministic draws，原EvaluationViewSampler/seed20261011，与B2逐case/draw station IDs一一对应。
spread V4复用既有protocol，仅draw0；B2 metrics直接复用，无新增baseline训练或推理。
clustered_v4_per_draw.csv与clustered_v4_per_case.csv保存两模型各draw/case metrics；对应完整JSONL保存IDs、azimuth和pool。
普通fixed V4与5-draw random V4不是同一subset。

{'\n'.join(improvements)}

正值代表OARI改善；判断重点始终是clustered V4和V3，不能只用普通Val-A微小变化包装核心创新。

## 15. Geometry-stratified analysis

score为域中心(0.3,0.3,0.3)m真实3D station-to-voxel directions的mean(1−abs_cos)，仅distinct unordered pairs。
这是简单sample geometry diversity score，区别于网络在32³各voxel计算的5维descriptor。
existing V3/V4/V6 random和clustered samples的case×draw结果保存geometry_stratified.csv，比较B2 error、OARI error和B2−OARI improvement。

{'\n'.join(correlations)}

Spearman仅描述性；同case多draw有重复，clustered draws也共享station windows，不作为独立样本，不报告统计显著性或因果解释。

## 16. Shortcut tests

{'\n'.join(shortcuts)}

wrong geometry使用既有geometry-only循环错位，Y不变；zero measurement只把normalized/raw Y清零，valid/path/geometry保留。
这检查完整模型对measurement和geometry的依赖，不能单独归因于OARI。
wrong relation保持measurements、per-view station geometry和observability不变，仅在interaction内对pairwise关系双轴循环错配。
{relation_text}该probe不作为硬GO条件；pairwise branch接到logits的独立功能测试另外报告。

## 17. Diagnostics

只保存val indices0/18/35 × spread/clustered draw0，共6个小JSON：IDs、azimuth、pairwise crossing角矩阵、observability、各head attention和final GCA weights。
包含valid-pair voxel平均与一个中心voxel；不保存所有voxel attention，不开展大型3D interpretability。
trained pairwise bias确实改变attention={diag['pairwise_bias_changes_attention']}，max change={diag['pairwise_attention_max_change']:.9g}；
observability变化改变interaction={diag['observability_changes_interaction']}，context max change={diag['observability_context_max_change']:.9g}。
trained全部gradients finite={diag['all_gradients_finite']}；pairwise gradient非零={diag['pairwise_gradient_nonzero']}；observability gradient非零={diag['observability_gradient_nonzero']}。
attention finite={diag['attention_finite']}；bias abs max={diag['pairwise_bias_abs_max']:.9g}；bias symmetry diff={diag['pairwise_bias_symmetry_max_difference']:.9g}。
这些诊断证明连接与数值正常，不证明最优view weighting、uniqueness或解决ill-posedness。

## 18. GO / NO-GO

**{decision}**。{interpretation}

沿用B2工程参考：主表Val-A/Val-B L1相对恶化≤2%；V3或clustered绝对L1改善≥0.005；wrong geometry/zero measurement L1>normal×1.2；mask/permutation/connectivity全部正常。
这些是冻结的operational references，不是统计阈值；L2变化独立完整报告。

```json
{json.dumps(s['go_criteria'], indent=2)}
```

下一步：{next_step}
单seed/36 val cases不能证明跨seed稳定或独立test泛化。本轮没有追加sanity revision、结构搜参或下一阶段模块。

## Reproducibility and integrity

报告生成commit=`{commit}`；正式训练source commit=`{source['commit']}`，training_source_manifest.json保存代码与配置SHA256。
{test_line}，全部OK；保留B0/B1/B2/B3所有原tests。
dataset {s['integrity']['dataset_files']} files unchanged={s['integrity']['dataset_unchanged']}；
B0/B1/B2 checkpoint unchanged={s['integrity']['b0_b1_b2_checkpoints_unchanged']}；model sources unchanged={s['integrity']['b0_b1_b2_model_sources_unchanged']}；
B3 artifacts unchanged={s['integrity']['b3_artifacts_unchanged']}；projector unchanged={s['integrity']['projector_unchanged']}；all historical artifacts unchanged={s['integrity']['all_historical_artifacts_unchanged']}。
test没有用于训练/推理/selection；integrity仅只读hash其文件。Pool B始终evaluation-only。没有重新生成dataset或改变Pool A/B。

artifacts=`{root}`，含summary.json、comparison.csv、sparse_views.csv、clustered_v4.csv、per-draw/per-case CSV、geometry_stratified.csv、shortcuts.csv、effective_protocol.yaml、integrity_summary.json、tests.log、diagnostics/和figures/。
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
'''
    (repo/'docs/reports/supervised_oari_v1.md').write_text(text)
    terminal = dict(current_git_commit=commit, training_source_commit=source['commit'],
        oari_model_name='GeoLift-OARI-Res3D', b2_parameter_count=b2['parameter_count'],
        oari_parameter_count=oari['parameter_count'], parameter_increase_absolute=parameter_increase,
        parameter_increase_relative_percent=parameter_percent, runtime=s['runtime'], observability_dimension=5,
        pairwise_dimension=3, pairwise_bias_network='8 -> 16 -> 4; SiLU; zero-init final layer',
        quick_gate_passed=gate['passed'], oari_best_epoch=oari['best_epoch'],
        oari_best_val_loss=oari['best_val_loss'],
        models={label:dict(val_a=m['evaluations']['Val-A-V4']['mean_per_case'],
            val_b=m['evaluations']['Val-B-V4']['mean_per_case'],
            sparse={k:v['mean_per_case'] for k,v in m['sparse'].items()},
            spread=m['spread']['mean_per_case']) for label,m in s['models'].items()},
        oari_vs_b2_improvements=s['improvements'],
        geometry_diversity_vs_improvement_spearman=s['geometry_analysis']['correlations'],
        shortcuts={key:oari['evaluations'][key]['mean_per_case'] for key in ('Val-A-V4', 'wrong_geometry', 'zero_measurement', 'wrong_relation')},
        permutation_max_difference=diag['prediction_permutation_max_difference'],
        padding_max_difference=diag['padding_max_difference'], all_invalid_safe=diag['all_invalid_safe'],
        pairwise_geometry_branch_gradient_nonzero=diag['pairwise_gradient_nonzero'],
        observability_branch_gradient_nonzero=diag['observability_gradient_nonzero'],
        pairwise_bias_changes_attention=diag['pairwise_bias_changes_attention'],
        observability_changes_interaction=diag['observability_changes_interaction'],
        attention_finite=diag['attention_finite'], integrity=s['integrity'],
        pool_b_evaluation_only=True, test_used=False, all_tests_passed=True,
        tests_passed_count=count, test_result=test_line, final_decision=decision,
        final_architecture_candidate_v1=s['final_architecture_candidate_v1'], next_step=next_step)
    (root/'terminal_summary.json').write_text(json.dumps(terminal, indent=2)+'\n')
    print(json.dumps(terminal, indent=2)); return terminal
