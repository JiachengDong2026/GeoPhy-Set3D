"""Concise report and terminal evidence for the completed B2 comparison."""
import json
import subprocess
from pathlib import Path


def render_b2_report(root,repo):
    root,repo=Path(root),Path(repo)
    s=json.loads((root/'summary.json').read_text());models=s['models'];interaction=s['interaction']
    labels=['B0-300','B1-GCA-300','B2-GCSI-300'];b1,b2=models[labels[1]],models[labels[2]]
    gate=json.loads((Path(b2['logs']).parent/'supervised_b2_gcsi_300_overfit/overfit_summary.json').read_text())
    smoke=json.loads((root/'memory_smoke.json').read_text())
    tests=(root/'tests.log').read_text();test_line=next(line for line in tests.splitlines() if line.startswith('Ran '))
    if not tests.rstrip().endswith('OK'):raise AssertionError('All tests must pass')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip()
    main=['| Model | Condition | MAE | RMSE | Relative L1 | Relative L2 | Best epoch | Best val loss |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    sparse=['| Model | V3 L1 | V4 random L1 | V6 L1 | Clustered V4 L1 |','|---|---:|---:|---:|---:|']
    shortcuts=['| Model | Normal L1 | Wrong geometry L1 | Zero measurement L1 |','|---|---:|---:|---:|']
    for label in labels:
        m=models[label]
        for condition in ('Val-A-V4','Val-B-V4'):
            values=m['evaluations'][condition]['mean_per_case']
            main.append('| '+label+' | '+condition+' | '+' | '.join(f'{values[k]:.9g}' for k in ('mae','rmse','relative_l1','relative_l2'))+f" | {m['best_epoch']} | {m['best_val_loss']:.9g} |")
        sparse.append('| '+label+' | '+' | '.join(f"{m['sparse'][key]['mean_per_case']['relative_l1']:.9f}" for key in ('V3','V4','V6','clustered-V4'))+' |')
        shortcuts.append('| '+label+' | '+' | '.join(f"{m['evaluations'][key]['mean_per_case']['relative_l1']:.9f}" for key in ('Val-A-V4','wrong_geometry','zero_measurement'))+' |')
    gains=['| Comparison | Condition | L1 absolute improvement | Percentage points | Relative improvement |',
           '|---|---|---:|---:|---:|']
    for row in s['improvements']:
        if row['metric']=='relative_l1':
            gains.append(f"| {row['comparison']} | {row['condition']} | {row['absolute_improvement']:+.9f} | {row['improvement_percentage_points']:+.6f} | {100*row['relative_improvement']:+.4f}% |")
    diagnostic_rows=['| Val index | Selection | Max deviation from uniform |','|---:|---|---:|']
    for e in interaction['examples']:
        diagnostic_rows.append(f"| {e['case_index']} | {e['selection']} | {e['max_deviation_from_uniform']:.8g} |")
    if s['retain_b2']:
        conclusion='B2满足本轮工程GO检查，建议保留并进入B3。最优先的V3/clustered收益见下表；该结论限定于一个训练seed的验证比较。'
    else:
        conclusion='B2未满足全部工程GO检查，暂不建议直接进入B3。保留本轮结果供负责人判断，没有做head/block/hidden-dimension搜索或追加seed。'
    if (b2['sparse']['V3']['mean_per_case']['relative_l1']>=b1['sparse']['V3']['mean_per_case']['relative_l1'] and
        b2['sparse']['clustered-V4']['mean_per_case']['relative_l1']>=b1['sparse']['clustered-V4']['mean_per_case']['relative_l1']):
        conclusion+=' Explicit set interaction did not outperform independent geometry-conditioned pooling on either prioritized sparse/clustered L1 condition.'
    next_stage='加入measurement-domain physics consistency。' if s['recommend_b3'] else '本轮不进入B3。'
    text=f'''# GeoLift-GCSI-Res3D / B2 supervised set interaction v1

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

B0={models[labels[0]]['parameter_count']:,}；B1={b1['parameter_count']:,}；B2={b2['parameter_count']:,}。
相对B1增加interaction参数{b2['interaction_parameter_count']:,}（{100*b2['interaction_parameter_count']/b1['parameter_count']:.3f}%）；总量<400k。
真实train/Pool A的B2/V6 FP32前向反向通过，prediction/attention与全部梯度finite，interaction gradient非零。
peak allocated={smoke['gpu_peak_allocated_bytes']/2**30:.3f}GiB，reserved={smoke['gpu_peak_reserved_bytes']/2**30:.3f}GiB。
naive voxel batching可在RTX4090运行，本轮未加入chunking或改变模型定义。

## 8. Quick gate 与正式协议

一次4-train-case fixed-V4/epoch0 gate，120 epochs、constant lr2e−4。
loss {gate['initial']['loss']['total']:.9g} → {gate['final']['loss']['total']:.9g}，
relative L1 {gate['initial']['metrics']['mean_per_case']['relative_l1']:.9g} → {gate['final']['metrics']['mean_per_case']['relative_l1']:.9g}。
全部gradient finite={gate['all_gradients_finite']}；interaction nonzero={gate['interaction_gradient_nonzero']}；
attention finite={gate['interaction_finite']}；pairwise prediction max MAE={gate['max_pairwise_prediction_mae']:.9g}；
permutation max difference={gate['prediction_permutation_max_difference']:.9g}；padding max difference={gate['padding_max_difference']:.9g}。
gate passed={gate['passed']}。只确认快速功能性学习，没有根据gate调参；正式训练从随机初始化重新开始。

正式B2只用seed20261011，166 train cases/Pool A/V3–6/hold0；36 val/Pool A/V4，epoch0/同station IDs。
FP32、batch2/accumulation1、AdamW lr2e−4/weight_decay1e−4、clip1，300 epochs、cosine T_max300，无early stopping。
loss仍为SmoothL1(beta1)+0.1×mean sample-relative-L1，仅gt_volume_64监督。
minimum fixed Val-A total supervised loss选best；Pool B仅在300-epoch训练完成且best冻结后评估。
未加入projection/hold-view/physics/TV/sparsity loss或128³监督，未搜索horizon，也未追加seed。

## 9–10. Val-A / Val-B

{'\n'.join(main)}

指标为mean per-case，物理mass-fraction尺度；pooled与逐case完整保留。
B2单case推理重现best training validation指标；best epoch在报告中为1-based，checkpoint/log为0-based。
B0/B1主表与原结果精确复用，checkpoint SHA256与历史artifact hashes均核对。

## 11–12. Sparse views 与 clustered

{'\n'.join(sparse)}

random V3/V4/V6与clustered V4各5 deterministic draws，EvaluationViewSampler、seed20261011。
三个模型逐case/draw使用完全相同station IDs，程序直接核对保存的JSONL pairings。
random-V4平均不同于主表固定reference-V4；未运行V2/V5/V8或完整condition matrix。
可选spread-V4仅一次既有deterministic selection，三个模型L1分别为：
{models[labels[0]]['spread']['mean_per_case']['relative_l1']:.9f} / {b1['spread']['mean_per_case']['relative_l1']:.9f} / {b2['spread']['mean_per_case']['relative_l1']:.9f}。
B0/B1只新增了这项低成本spread评估，所有输出写到本轮目录，旧结果未改写。

关键improvement（正值=新模型更好，负值=恶化）：

{'\n'.join(gains)}

improvements.csv同时保存L1/L2 absolute、percentage-point与relative improvements。
V3与clustered的意义优先于普通Val-A的小幅波动，不因单个主表分数改善就宣称学到物理冗余/互补机制。

## 13. Shortcut tests

{'\n'.join(shortcuts)}

zero measurement仅将Y/rawY置零，保留valid/path/geometry；wrong geometry循环错位station geometry与valid/path，Y顺序不动。
它们检查完整模型对measurement和geometry的依赖，不能单独把退化归因于interaction。

## 14. Interaction diagnostics

固定val indices0/18/35，spread/clustered V4 draw0。仅保存各head和全head的voxel平均V×V矩阵、IDs、azimuth及pooling mean weights。
矩阵平均按有效query voxel数量归一化。没有保存训练attention volumes或开展额外interpretability项目。

{'\n'.join(diagnostic_rows)}

attention finite={interaction['attention_finite']}；query/key mask正确={interaction['interaction_mask_correct']}；
pooling invalid/padded严格0={interaction['pooling_mask_correct']}；all-invalid安全={interaction['all_invalid_safe']}；
nonuniform={interaction['nonuniform']}，max deviation from uniform={interaction['max_deviation_from_uniform']:.9g}。
interaction row-sum最大误差={interaction['interaction_weight_sum_max_error']:.9g}，pooling sum误差={interaction['pooling_weight_sum_max_error']:.9g}。
trained prediction permutation max difference={interaction['prediction_permutation_max_difference']:.9g}；
padding max difference={interaction['padding_max_difference']:.9g}；context equivariance max difference={interaction['context_permutation_equivariance_max_difference']:.9g}。
controlled synthetic输入包含10°/11°/100°三个geometry tokens：finite={interaction['redundancy_sanity']['finite']}，
max deviation from uniform={interaction['redundancy_sanity']['max_deviation_from_uniform']:.9g}。
该sanity不要求100°view权重最高，也不是科学验收阈值。nonuniform矩阵只证明非均匀交互，不能证明已学会去冗余。
单元测试额外证明一个view的feature/geometry改变可以影响另一个view的context，并检查invalid内容不影响valid结果。

## 15. GO / NO-GO 与B3边界

{conclusion}

冻结的简单工程参考：主表L1相对恶化≤2%，V3或clustered L1改善≥0.005（0.5个百分点），
wrong geometry/zero measurement L1>normal的1.2倍，mask与permutation正确。
这些是操作性判断，不是论文统计显著性阈值；raw L1/L2变化全部公开。

```json
{json.dumps(s['go_criteria'],indent=2)}
```

下一阶段：{next_stage}本轮没有实现B3。
单seed/36-val-case结果不能证明跨seed稳定、统计显著或独立盲测test性能。

## Reproducibility 与 integrity

报告生成时commit=`{commit}`；B2训练源码与配置SHA256见training_source_manifest.json，正式训练从已提交的实现启动。
B0/B1保留原训练source/provenance，不修改其配置/权重/报告/旧实验输出。
{test_line}，全部OK。
dataset {s['integrity']['dataset_files']} files unchanged={s['integrity']['dataset_unchanged']}；
B0/B1 artifacts unchanged={s['integrity']['b0_b1_artifacts_unchanged']}；
B0/B1 model/config/report unchanged={s['integrity']['b0_b1_model_config_report_unchanged']}；
numeric training sources unchanged={s['integrity']['numerical_training_sources_unchanged']}。
test没有用于训练/推理/selection，integrity只读hash test artifacts；Pool B始终evaluation-only。

所有artifacts在`{root}`：summary.json、comparison.csv、sparse_views.csv、clustered_v4.csv、
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
'''
    (repo/'docs/reports/supervised_b2_gcsi_v1.md').write_text(text)
    terminal=dict(current_git_commit=commit,b2_parameter_count=b2['parameter_count'],
        additional_vs_b1=b2['interaction_parameter_count'],b2_best_epoch=b2['best_epoch'],b2_best_val_loss=b2['best_val_loss'],
        models={label:dict(val_a=models[label]['evaluations']['Val-A-V4']['mean_per_case'],
            val_b=models[label]['evaluations']['Val-B-V4']['mean_per_case'],
            sparse_l1={key:value['mean_per_case']['relative_l1'] for key,value in models[label]['sparse'].items()}) for label in labels},
        b2_vs_b1_improvements=[r for r in s['improvements'] if r['comparison']=='B2-GCSI-300 vs B1-GCA-300'],
        b2_shortcuts_l1={key:b2['evaluations'][key]['mean_per_case']['relative_l1'] for key in ('Val-A-V4','wrong_geometry','zero_measurement')},
        interaction={key:value for key,value in interaction.items() if key not in ('examples','redundancy_sanity')},
        integrity=s['integrity'],test_set_used=False,pool_b_evaluation_only=True,all_tests_passed=True,test_result=test_line,
        retain_b2=s['retain_b2'],recommend_b3=s['recommend_b3'],next_stage=next_stage,conclusion=conclusion)
    (root/'terminal_summary.json').write_text(json.dumps(terminal,indent=2)+'\n')
    print(json.dumps(terminal,indent=2));return terminal
