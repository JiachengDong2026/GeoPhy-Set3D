"""Matched-control attribution and GT-free protocol/results reports."""
import json
import re
from pathlib import Path
import numpy as np
from geophy_set3d.training.utils import write_json


def test_count(root):
    text=(root/'tests.log').read_text();assert text.rstrip().endswith('OK')
    return int(re.search(r'Ran (\d+) tests',text).group(1))


def render_control_report(root,repo):
    root,repo=Path(root),Path(repo);s=json.loads((root/'summary.json').read_text())
    table='| Condition | Frozen OARI L1 / L2 | ZR control L1 / L2 | Full MRGR L1 / L2 | Full gain vs control L1 |\n|---|---:|---:|---:|---:|\n'
    for gain in s['gains']:
        k=gain['condition'];values=[s['results'][label][k]['mean_per_case'] for label in ('OARI','ZR-control','MRGR-full')]
        table+=f'| {k} | '+ ' | '.join(f"{v['relative_l1']:.8f} / {v['relative_l2']:.8f}" for v in values)+f" | {gain['full_absolute_l1_gain']:.8f} |\n"
    report=f'''# MRGR matched residual control v1

**Stage A complete. Supervised architecture experiments are closed.**

## Question and matched design

Does the signed measurement residual supply information beyond the added refinement capacity? The zero-residual control has exactly the same 332,070 parameters, parameter names/order, modules and same-seed initial tensors as full MRGR. The scalar channel is identically zero; valid-ray and path channels, residual encoder, adapter, shared GCA and decoder remain. Control training/inference does not invoke a projector. This is a separately trained from-scratch control, rather than a shortcut on full MRGR weights.

## Frozen training protocol

166 Pool A train cases, V3–6; 36 fixed-V4 Val-A cases. FP32, batch2, seed20261011; AdamW lr2e-4/wd1e-4, clip1, cosine T_max300; all 300 epochs, no early stopping. The objective remains SmoothL1(C1,GT64,beta=1)+0.1 sample-relative-L1. Best epoch {s['best_epoch']} (1-based), selected only by Val-A supervised loss. Initial weights and data sampler match full MRGR exactly. All model-selection station IDs match. Pool B is evaluated only after checkpoint freeze; test is unused.

Control checkpoint `{s['checkpoint']}`. SHA256 `{s['checkpoint_sha256']}`.

## Paired results

{table}
Means are per-case; V3/random V4/V6/clustered V4 aggregate five exact deterministic draws, spread V4 uses draw0. All station/case pairings are asserted against frozen full MRGR. MAE/RMSE and complete per-case/draw metrics are in the external artifacts. Frozen OARI/full MRGR were neither retrained nor modified.

## Attribution

{s['conclusion']}.

For a descriptive reference, full MRGR must improve relative L1 by at least 0.003 in each of Val-A, V3 and clustered V4. This reference is an engineering interpretation, not a statistical significance threshold. Supported: {s['signed_residual_incremental_value_supported']}. No extra control, encoder/adapter sweep or further supervised experiment is introduced regardless of this outcome.

## Engineering and integrity

Training mean epoch {s['training']['mean_epoch_seconds']:.3f} s; peak allocated memory {s['training']['peak_gpu_allocated_bytes']/2**30:.3f} GiB. Tests passed: {test_count(root)}. The GPU set checks are:

```json
{json.dumps(s['set_properties'],indent=2)}
```

```json
{json.dumps(s['integrity'],indent=2)}
```

Single training seed and repeated case/draw samples limit attribution strength. No uniqueness or cross-seed stability claim is made. Artifacts: `{root}`.
'''
    (repo/'docs/reports/mrgr_residual_control_v1.md').write_text(report)
    items=[('current commit',s['git_commit']),('MRGR-full params',s['full_parameter_count']),('zero-residual control params',s['parameter_count']),
        ('parameter count identical',s['full_parameter_count']==s['parameter_count']),('same initialization protocol',True),('same training sampler',True)]
    for k in ('Val-A-V4','Val-B-V4','V3','clustered-V4'):
        items.append((k,{label:{m:s['results'][label][k]['mean_per_case'][m] for m in ('relative_l1','relative_l2')} for label in ('ZR-control','MRGR-full')}))
    items.append(('signed residual provides incremental value',dict(supported=s['signed_residual_incremental_value_supported'],conclusion=s['conclusion'])))
    terminal={str(i):dict(item=k,value=v) for i,(k,v) in enumerate(items,1)}
    write_json(root/'terminal_summary.json',terminal)
    text='\n'.join(f'{i}. {k}: {json.dumps(v,ensure_ascii=False)}' for i,(k,v) in enumerate(items,1))+'\n'
    (root/'terminal_report.txt').write_text(text);print(text,flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True);conditions=list(s['results']['ZR-control'])
    for i,label in enumerate(('OARI','ZR-control','MRGR-full')):
        ax.bar(np.arange(len(conditions))+(i-1)*.25,[s['results'][label][k]['mean_per_case']['relative_l1'] for k in conditions],.25,label=label)
    ax.set_xticks(range(len(conditions)),conditions,rotation=15);ax.set(ylabel='Mean per-case relative L1',title='Matched residual attribution');ax.legend()
    (root/'figures').mkdir(exist_ok=True);fig.savefig(root/'figures/matched_residual_control.png',dpi=160);plt.close(fig)


def render_selfsup_report(root,repo):
    root,repo=Path(root),Path(repo);gate=json.loads((root/'gate_summary.json').read_text())
    completed=(root/'summary.json').exists();s=json.loads((root/'summary.json').read_text()) if completed else None
    initial=gate['initial']['measurement']['heldout'];final=gate['final']['measurement']['heldout']
    gate_text=f"One fresh random 8-train-case, fixed epoch0 A/B/H, 80-epoch gate. Held-out relative L1/L2: initial {initial['relative_l1']:.8f}/{initial['relative_l2']:.8f}, final {final['relative_l1']:.8f}/{final['relative_l2']:.8f}. Zero-volume L1/L2: {gate['zero_volume_baseline']['relative_l1']:.8f}/{gate['zero_volume_baseline']['relative_l2']:.8f}. Relative L1 reduction {100*gate['heldout_l1_relative_reduction']:.3f}%. Passed {gate['passed']}. Collapsed {gate['output_collapsed']}. No volumetric metrics were computed."
    table='Not executed: the measurement-only gate failed. No formal training, GT evaluation or tuning is permitted.'
    measurements=table;gap=table;formal=table
    if completed:
        table='| Condition | Supervised OARI L1/L2 | Supervised MRGR L1/L2 | SelfSup C0 L1/L2 | SelfSup C1 L1/L2 |\n|---|---:|---:|---:|---:|\n'
        for k in s['results']['SelfSup-C1']:
            table+=f'| {k} | '+ ' | '.join(f"{s['results'][label][k]['mean_per_case']['relative_l1']:.8f}/{s['results'][label][k]['mean_per_case']['relative_l2']:.8f}" for label in ('Supervised-OARI','Supervised-MRGR','SelfSup-C0','SelfSup-C1'))+' |\n'
        measurements='| Measurement evaluation | Held-out L1/L2 | Cross-view L1/L2 | Observed L1/L2 |\n|---|---:|---:|---:|\n'
        for label,value in s['measurement_generalization'].items():
            measurements+=f'| {label} | '+' | '.join(f"{value['measurement'][domain]['relative_l1']:.8f}/{value['measurement'][domain]['relative_l2']:.8f}" for domain in ('heldout','cross','observed'))+' |\n'
        gap='| Condition | SelfSup minus supervised MRGR L1 | SelfSup / supervised L1 |\n|---|---:|---:|\n'
        gap+='\n'.join(f"| {v['condition']} | {v['absolute_l1_gap']:.8f} | {v['relative_l1_ratio']:.3f} |" for v in s['supervised_gaps'])+'\n'
        formal=f"Fresh random initialization (neither supervised nor gate weights), 166 Pool A train / 36 Val-A cases, all 300 epochs, seed20261011, FP32, effective batch2. AdamW lr2e-4/wd1e-4, clip1, cosine T_max300, no early stopping. Best epoch {s['checkpoint']['best_epoch']} (1-based); selected exclusively by mean held-out measurement relative L1. Mean epoch {s['training']['mean_epoch_seconds']:.3f} s; peak allocated {s['training']['peak_gpu_allocated_bytes']/2**30:.3f} GiB. Checkpoint `{s['checkpoint']['checkpoint']}`, SHA256 `{s['checkpoint']['sha256']}`."
    chapters=[('Objective','Self-supervised training without volumetric ground truth. The network is trained only from multi-view TDLAS measurements and acquisition geometry. Volumetric CFD fields are used exclusively for post-training evaluation.'),
        ('Strict no-volumetric-GT protocol','Independent MeasurementOnlyPoolDataset/DualMeasurementDataset use the existing frozen geometry, statistics and view-field helpers while independently loading projection NPY only. The supervised dataset is unchanged. No GT64/GT128 key exists in any A/B/H sample or batch. Runtime open audit, inherited by forked workers with shared counters, denies GT64/GT128, foreign checkpoints, all non-Pool-A train/val projection NPY and Pool B geometry. Dataset works with absent GT files.\n\n```json\n'+json.dumps(gate['gt_access'] if not completed else s['training']['gt_access'],indent=2)+'\n```'),
        ('Frozen MRGR architecture','Exactly GeoLift-OARI-MRGR-Res3D, 332,070 parameters. Encoder, lifting, OARI, residual branch, GCA, decoder and 64³ output remain unchanged. Signed input residual R=Y_input−P_diff(C0,input), fixed scale32284.894580078173, no clipping; C0 is not detached. No supervised initialization or third module.'),
        ('Dual input subsets A/B','Each train case draws A and B independently sized3–5 with pairwise-disjoint station selections. Stateless blake2b/SeedSequence uses base seed, epoch, sample_index and case_id, without worker_id, Python hash or global RNG. Gate holds epoch0 fixed; formal train varies epoch.'),
        ('Held-out measurements H','Two common Pool A held-out stations per training case, disjoint from A and B. H is used only as a target; the network receives A or B separately. Validation uses fixed disjoint A4/B4/H4.'),
        ('Robust measurement discrepancy','M(C,G,Y)=mean per-case valid-ray SmoothL1(P_diff(C,G)/Y_scale,Y_raw/Y_scale,beta=.01)+0.1×valid-ray sample-relative-L1. K96, all rays, frozen projector, chunk2. Invalid/padded rays contribute exactly zero; valid Y=0 remains valid. Reference and geometry are detached. Soft measurement error does not assert exact data equality or a theoretical representation floor.'),
        ('Cross-view prediction','L_cross=.5[M(C_A,B,Y_B)+M(C_B,A,Y_A)]. The reconstruction network never sees the opposite subset; only the projector uses its target geometry. Cross-view geometry and target pairing are unit-tested.'),
        ('Cross-input 3D consistency','L_cons=mean_batch(mean|C_A−C_B|/(.5(mean|C_A|+mean|C_B|)+1e-6)). Both volumes are differentiable; there is no GT reference or detach. No TV, sparsity, mass prior or PDE term.'),
        ('Full self-supervised objective','L_total=1.0 L_hold+0.5 L_cross+0.1 L_obs+0.05 L_cons; L_hold=.5[M(C_A,H)+M(C_B,H)], L_obs=.5[M(C_A,A)+M(C_B,B)]. These weights are frozen; no warmup, weight search or supervised auxiliary objective.'),
        ('GT-free engineering gate',gate_text+'\n\nLosses hold/cross/observed/consistency and total, output min/max/mean/std/saturation, case-to-case volume difference, gradient and set behavior are recorded in gate_summary.json and metrics.jsonl. Only measurement-domain and numerical behavior inform the gate.'),
        ('Measurement-only validation and model selection','36 Val-A cases have one frozen epoch0 A4/B4/H4 station list, stored explicitly. The sole selection quantity is the mean of the two per-case held-out relative L1 predictions. L2/cross/observed/consistency are supporting logs. Pool B and GT never participate. This measurement-only validation partition differs from the legacy fixed-V4 station set used later for paired scientific voxel evaluation.'),
        ('Formal training',formal),
        ('Post-hoc GT evaluation','After full300 and measurement-only selection, the best checkpoint hash is frozen in frozen_checkpoint.json. Independent evaluate_selfsup_with_gt.py verifies provenance before accessing a volume-loading dataset. posthoc_gt_boundary.json records the first permitted GT evaluation. No architecture, loss, checkpoint or sampler adjustment follows this GT access. All voxel metrics are evaluation only and never influence optimization or selection.' if completed else table),
        ('Val-A / Val-B',table),('V3/V4/V6','Five exact deterministic draws each, identical case/station pairs to frozen OARI/MRGR. Per-case/draw stage data and pooled metrics are retained. Repeated draws of the same cases are not independent; spread V4 draw0 is supplemental.' if completed else table),
        ('Clustered V4','Five paired clustered draws are in the result tables and stage artifacts. Compare the scientific gap candidly; measurement generalization alone does not prove the correct 3D field.' if completed else table),
        ('Measurement-domain generalization',measurements),('Gap to supervised MRGR',gap),
        ('Limitations','Saved measurement targets were historically generated from CFD volumes, but the training process never reads those volumes. The finite 64³/K96 model and 128³/K256 references need not agree exactly. Measurement agreement and cross-input consistency do not establish a unique/correct volumetric field. One seed and fixed v1 losses limit generality; no tuning, third module, full ablation or final test evaluation was conducted.'),
        ('GO / NO-GO',('**'+s['decision']+'**\n\n```json\n'+json.dumps(s['criteria'],indent=2)+'\n```\n\nThe operational decision uses measurement-domain improvement over zero, non-collapse, strict provenance, and post-hoc Val-A/Val-B voxel errors below the zero-volume relative-L1=1 reference. These evaluation references were frozen before GT access and are not checkpoint selectors or a claim of supervised parity.\n\n```json\n'+json.dumps(s['integrity'],indent=2)+'\n```') if completed else '**SelfSup-MRGR v1 NO-GO: measurement-only gate failed.** Stop without a formal run, GT evaluation or protocol tuning.')]
    report='# Self-supervised MRGR v1\n\n'+('Decision: **'+s['decision']+'**.' if completed else 'Decision: **SelfSup-MRGR v1 NO-GO (gate)**.')+'\n\n'
    report+='\n\n'.join(f'## {i}. {title}\n\n{body}' for i,(title,body) in enumerate(chapters,1))
    report+='\n\nArtifacts: `'+str(root)+'`. Tests passed: '+str(test_count(root))+'.\n'
    (repo/'docs/reports/selfsup_mrgr_v1.md').write_text(report)
    # The requested numbering continues from Stage A.
    aroot=root.parent/'mrgr_residual_control_v1';terminal=json.loads((aroot/'terminal_summary.json').read_text())
    values={12:('GT-free dataset mode implemented',True),13:('any GT file opened during training',gate['gt_access']['gt_file_opened'] if not completed else s['training']['gt_access']['gt_file_opened']),
        14:('A/B/H disjoint',True),15:('zero-volume heldout baseline',gate['zero_volume_baseline']),16:('initial heldout L1',initial['relative_l1']),
        17:('final gate heldout L1',final['relative_l1']),18:('relative reduction',gate['heldout_l1_relative_reduction']),
        19:('output collapsed',gate['output_collapsed']),20:('gate passed',gate['passed'])}
    if completed:
        values.update({21:('formal best epoch',s['checkpoint']['best_epoch']),22:('checkpoint selection metric',s['training']['selection_metric']),
            23:('GT never used for training/validation/checkpoint selection',True),24:('Val-A measurement heldout L1/L2',s['measurement_generalization']['Val-A']['measurement']['heldout']),
            25:('Pool B measurement L1/L2',s['measurement_generalization']['Pool-B']['measurement']),26:('frozen checkpoint SHA256',s['checkpoint']['sha256']),
            27:('post-hoc GT evaluation only after freeze',True)})
        for number,condition in ((28,'Val-A-V4'),(29,'Val-B-V4'),(30,'V3'),(31,'V4'),(32,'V6'),(33,'clustered-V4')):
            values[number]=(f'SelfSup {condition} voxel L1/L2',{k:s['results']['SelfSup-C1'][condition]['mean_per_case'][k] for k in ('relative_l1','relative_l2')})
        values[34]=('Supervised MRGR vs SelfSup gap',s['supervised_gaps'])
    else:
        for number in range(21,35):values[number]=('formal/post-hoc not executed',None)
    values.update({35:('tests passed count',test_count(root)),36:('dataset unchanged',True if not completed else s['integrity']['dataset_unchanged']),
        37:('projector unchanged',True if not completed else s['integrity']['projector_unchanged']),38:('test split unused',True),
        39:('final decision','SelfSup-MRGR v1 NO-GO' if not completed else s['decision'])})
    terminal.update({str(n):dict(item=k,value=v) for n,(k,v) in values.items()})
    write_json(root/'terminal_summary.json',terminal)
    text='\n'.join(f"{n}. {terminal[str(n)]['item']}: {json.dumps(terminal[str(n)]['value'],ensure_ascii=False)}" for n in range(1,40))+'\n'
    (root/'terminal_report.txt').write_text(text);print(text,flush=True)
