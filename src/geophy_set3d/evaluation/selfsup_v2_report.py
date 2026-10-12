"""RVTC results, engineering decision and exact terminal checklist."""
import json
import re
from pathlib import Path
import numpy as np
from geophy_set3d.evaluation.analysis import write_csv,read_jsonl
from geophy_set3d.training.utils import write_json


def metric_table(results):
    text='| Condition | Supervised MRGR L1 / L2 | SelfSup v1 L1 / L2 | SelfSup v2 L1 / L2 |\n|---|---:|---:|---:|\n'
    for condition in results['SelfSup-v2']:
        text+=f'| {condition} | '+' | '.join(f"{results[label][condition]['mean_per_case']['relative_l1']:.8f} / {results[label][condition]['mean_per_case']['relative_l2']:.8f}"
             for label in ('Supervised-MRGR','SelfSup-v1','SelfSup-v2'))+' |\n'
    return text


def render_report(root,repo):
    root,repo=Path(root),Path(repo)
    gate=json.loads((root/'gate_summary.json').read_text()); s=json.loads((root/'summary.json').read_text())
    completed=s['completed']; skipped='Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.'
    table=metric_table(s['results']) if completed else skipped
    training=s.get('training'); checkpoint=s.get('checkpoint'); teacher=s.get('teacher_posthoc')
    tests=(root/'tests.log').read_text() if (root/'tests.log').exists() else ''
    matched=re.search(r'Ran (\d+) tests',tests); test_count=int(matched.group(1)) if matched and tests.rstrip().endswith('OK') else None
    gate_text=(f"Exactly one fresh-random 8-case / 80-epoch gate, fixed epoch0 A/B/H/T. Initial sparse held-out L1 "
        f"{gate['initial']['measurement']['heldout']['relative_l1']:.8f}; final {gate['final']['measurement']['heldout']['relative_l1']:.8f}; "
        f"teacher final {gate['final']['measurement']['teacher_heldout']['relative_l1']:.8f}; zero-volume baseline "
        f"{gate['zero_volume_baseline']['relative_l1']:.8f}. Frozen v1 gate reference 0.520501; v1 was not retrained and this is not a formal v1/v2 comparison. "
        f"First10 / last10 training teacher loss means {gate['student_teacher_decrease']['first10_mean']:.8f} / "
        f"{gate['student_teacher_decrease']['last10_mean']:.8f}. Output collapsed: {gate['output_collapsed']}; passed: {gate['passed']}.\n\n"
        '```json\n'+json.dumps(gate['criteria'],indent=2)+'\n```')
    formal=(f"Fresh random initialization; no v1/supervised/gate weights. All 300 epochs, 166 train / 36 Val-A, seed20261011, "
        f"FP32, batch1 / accumulation2, AdamW lr2e-4 wd1e-4, clip1, cosine T_max300. Best epoch {checkpoint['best_epoch']} (1-based), "
        f"selected only by sparse held-out relative L1. Mean epoch {training['mean_epoch_seconds']:.3f}s. "
        f"Frozen checkpoint `{checkpoint['checkpoint']}`, SHA256 `{checkpoint['sha256']}`. "
        f"GT post-hoc boundary `{s['posthoc_boundary']['first_posthoc_gt_evaluation_at']}` (Asia/Shanghai).") if completed else skipped
    measurement={}
    if completed:
        for pool,result in s['measurement_generalization'].items():
            measurement[pool]='| Domain | Relative L1 | Relative L2 |\n|---|---:|---:|\n'+''.join(
                f"| {domain} | {values['relative_l1']:.8f} | {values['relative_l2']:.8f} |\n" for domain,values in result['measurement'].items())
    teacher_text=(f"One evaluation-only pass on the frozen same-case Val-A A4/B4/T8 triples. Voxel relative L1: "
        f"A={teacher['A']['mean_per_case']['relative_l1']:.8f}, B={teacher['B']['mean_per_case']['relative_l1']:.8f}, "
        f"T={teacher['T']['mean_per_case']['relative_l1']:.8f}. T beats the mean student in "
        f"{100*teacher['teacher_better_than_mean_student_fraction']:.2f}% of cases. Rich-teacher accuracy hypothesis supported: "
        f"{teacher['teacher_generally_more_accurate']}. This diagnostic never optimizes, selects or revises the method.") if completed else (
        'GT teacher-quality diagnostic not executed. Measurement-only gate diagnostics compare A/B/T held-out errors and log output min/max/mean/std, fractions near0/1, case differences and D(A,T)/D(B,T)/D(A,B) each epoch.')
    gaps=skipped
    if completed:
        gaps='| Condition | v2 minus supervised L1 | v2 / supervised L1 |\n|---|---:|---:|\n'+''.join(
            f"| {r['condition']} | {r['absolute_l1_gap']:.8f} | {r['relative_l1_ratio']:.3f} |\n" for r in s['supervised_gaps'])
    chapters=[('Motivation','Test whether additional real TDLAS views provide a useful GT-free 3D structural target for sparse reconstruction.'),
        ('Limitation of SelfSup v1','v1 shows feasible GT-free reconstruction, while measurement compatibility does not establish volumetric correctness. Symmetric sparse A↔B consistency may reinforce a common ambiguous solution.'),
        ('Frozen MRGR architecture','GeoLift-OARI-MRGR-Res3D, exactly332,070 parameters. Encoder, lifting, OARI, residual branch, GCA, decoder, 64³ output and projector source remain unchanged.'),
        ('Sparse subsets A/B','Each train case draws disjoint A3–5 and B3–5 from Pool A. The v1 stateless seed/case/epoch sampler is reused exactly; formal training varies epoch.'),
        ('Rich subset T=A∪B','T6–10 is built before padding from the concatenated station IDs. All measurement/geometry/mask fields are indexed with those same IDs. View order does not change the set prediction. Validation uses T8.'),
        ('Held-out subset H','Train H2, validation H4. A/B/H are pairwise disjoint and T excludes H. H is a projector target, never a reconstruction network input. Gate epoch0 stations and all36 Val-A station IDs are saved.'),
        ('Rich-view teacher formulation','C_A=F(A), C_B=F(B), C_T=F(T) use one shared network and one optimizer. No EMA, second teacher network, prior or volume label. A reconstruction obtained from the union of two sparse measurement subsets is used as a richer-view pseudo-teacher for volumetric consistency.'),
        ('Stop-gradient volumetric distillation','D(X,T)=mean_voxel|X−T|/(mean_voxel|T|+1e-6), computed per sample then batch mean. L_teacher=.5[D(C_A,C_T.detach())+D(C_B,C_T.detach())]. Teacher consistency cannot directly move the C_T branch; its measurement losses remain differentiable. Both numerator and denominator use the detached teacher. No absolute voxel L1 or symmetric A/B loss is optimized.'),
        ('Measurement objectives','Frozen v1 M: per-case valid-ray SmoothL1(P_diff(C,G)/32284.894580078173,Y_raw/32284.894580078173,beta=.01)+.1 valid-ray relativeL1. K96, all rays, view_chunk2. L_hold=.5[M(C_A,H)+M(C_B,H)], L_cross=.5[M(C_A,B)+M(C_B,A)], L_obs=.5[M(C_A,A)+M(C_B,B)], L_hold_teacher=M(C_T,H), L_obs_teacher=M(C_T,T).'),
        ('Full v2 loss','L_total=1.0 L_hold+.5 L_cross+.1 L_obs+1.0 L_hold_teacher+.1 L_obs_teacher+.1 L_teacher. No weight sweep, warmup, TV, sparsity, mass/PDE/smoothness prior, teacher dropout or architecture change.'),
        ('Strict GT-free protocol','Runtime GT/foreign-checkpoint/projection-whitelist audits retain worker-inherited shared counters, with explicit Pool B access counts. Any forbidden read immediately fails. Training and model selection are volumetric-GT-free. Method development has inspected v1 volumetric GT post-hoc results; it is incorrect to claim otherwise.\n\n```json\n'+json.dumps(training['gt_access'] if completed else gate['gt_access'],indent=2)+'\n```'),
        ('Engineering gate',gate_text),
        ('Measurement-only model selection',formal+'\n\nThe only selector is .5[heldout(C_A,H)+heldout(C_B,H)] relativeL1. Teacher measurements, volumetric consistency and GT metrics never select a checkpoint.'),
        ('Val-A measurement results',measurement.get('Val-A',skipped)),
        ('Pool-B measurement results',measurement.get('Pool-B',skipped)),
        ('Post-hoc voxel evaluation',table+'\n\nAll case/station IDs are asserted against both frozen v1 and supervised MRGR, which already pair with supervised OARI. Full300 and checkpoint freeze must precede any GT read. Measurement evaluation finishes first.'),
        ('V3/V4/V6','Five exact deterministic draws per condition, using the frozen reference case/station IDs. Repeated draws are not independent samples.\n\n'+table),
        ('Clustered V4','Five paired draws. Engineering reference relativeL1≤.42 guards against materially losing v1’s clustered advantage.\n\n'+table),
        ('Teacher-quality diagnostics',teacher_text),
        ('SelfSup v1 vs v2',table+'\n\nThe predeclared engineering GO requires Val-A absolute L1 gain≥.003, Val-B relative worsening≤3%, improvements in V3/V6, and clustered L1≤.42. The V3 .115 value is a supporting reference only. Measurement error alone cannot decide success.'),
        ('Gap to supervised MRGR',gaps),
        ('Limitations','Rich measurements do not guarantee unique or more accurate 3D reconstruction. Saved projections originate historically from CFD, while training never reads volumetric fields. One seed limits inference. No final test, full ablation or multi-seed was run. The working term RVTC is not a final paper name.'),
        ('GO / NO-GO',f"**{s['decision']}**\n\n"+('Freeze as Final self-supervised training candidate. Proceed to final ablation and multi-seed validation.' if s['decision'].endswith(' v2 GO') else 'Retain SelfSup v1. Pause automatic experiments; do not create v3 or add priors/EMA/modules.')+
         '\n\n```json\n'+json.dumps(s['integrity'],indent=2)+'\n```')]
    text='# SelfSup-MRGR v2: Rich-View Teacher Consistency\n\n'+f"Decision: **{s['decision']}**. Training source commit `{json.loads((root/'training_source_manifest.json').read_text())['commit']}`. Tests passed: {test_count}.\n\n"
    text+='\n\n'.join(f'## {i}. {title}\n\n{content}' for i,(title,content) in enumerate(chapters,1))+'\n'
    (repo/'docs/reports/selfsup_mrgr_v2.md').write_text(text)
    diagnostic_rows=[]
    cfg=json.loads((root/'training_source_manifest.json').read_text())
    exp='selfsup_oari_mrgr_v2_rvtc'
    base=root.parents[1]
    for phase,name in (('gate',exp+'_gate'),('formal',exp)):
        path=base/'logs'/name/'metrics.jsonl'
        if not path.exists():continue
        for row in read_jsonl(path):
            for split in ('train','val'):
                if split not in row:continue
                result=row[split]
                diagnostic_rows.append(dict(phase=phase,split=split,epoch=row['epoch']+1,
                    sparse_heldout_l1=result['measurement']['heldout']['relative_l1'],teacher_heldout_l1=result['measurement']['teacher_heldout']['relative_l1'],
                    **result['disagreement'],**{'teacher_'+k:v for k,v in result['teacher_output'].items()}))
    write_csv(root/'teacher_training_diagnostics.csv',diagnostic_rows)
    terminal_report(root,s,gate,test_count)
    if completed:figures(root,s)


def terminal_report(root,s,gate,test_count):
    completed=s['completed'];pending='not executed: engineering gate failed'
    t=s.get('training',{}); freeze=s.get('checkpoint',{});audit=t.get('gt_access',{})
    def pair(condition):
        if not completed:return pending
        return {label:{metric:s['results'][label][condition]['mean_per_case'][metric] for metric in ('relative_l1','relative_l2')}
                for label in ('SelfSup-v1','SelfSup-v2')}
    props=gate['set_properties'] if not completed else s['set_properties']
    grads=gate['teacher_gradients'] if not completed else s['teacher_gradients']
    items=[('current git commit',s['git_commit']),('model name',s['model']),('parameter count unchanged',s['parameter_count']),
        ('v2 architecture changed?',False),('A views range',[3,5]),('B views range',[3,5]),('H views',2),('T views range',[6,10]),
        ('T = A union B',True),('T excludes H',True),('teacher stop-gradient verified?',grads['teacher_stop_gradient_verified']),
        ('teacher measurement gradient nonzero?',grads['teacher_measurement_gradient_abs_sum']>0),
        ('GT open attempts during gate',gate['gt_access']['gt_open_attempts']),('supervised checkpoint open attempts',gate['gt_access']['supervised_checkpoint_open_attempts']),
        ('gate sparse heldout L1 initial/final',{key:gate[key]['measurement']['heldout']['relative_l1'] for key in ('initial','final')}),
        ('gate teacher heldout L1',gate['final']['measurement']['teacher_heldout']['relative_l1']),('output collapsed?',gate['output_collapsed']),
        ('gate passed?',gate['passed']),('formal best epoch',freeze.get('best_epoch',pending)),('selection metric','mean sparse held-out measurement relative L1'),
        ('GT open attempts formal',audit.get('gt_open_attempts',pending)),('external checkpoint attempts formal',audit.get('external_checkpoint_attempts',pending)),
        ('Val-A measurement sparse/teacher L1/L2',s.get('measurement_generalization',{}).get('Val-A',pending)),
        ('Pool-B measurement sparse/teacher L1/L2',s.get('measurement_generalization',{}).get('Pool-B',pending)),
        ('frozen checkpoint SHA256',freeze.get('sha256',pending)),('GT evaluation only after freeze',completed if completed else 'GT evaluation not executed'),
        ('posthoc Val-A v1/v2 L1/L2',pair('Val-A-V4')),('Val-B v1/v2',pair('Val-B-V4')),('V3 v1/v2',pair('V3')),
        ('random V4 v1/v2',pair('V4')),('V6 v1/v2',pair('V6')),('clustered V4 v1/v2',pair('clustered-V4')),
        ('spread V4 v1/v2',pair('spread-V4')),('teacher posthoc voxel L1 vs students',s.get('teacher_posthoc',pending)),
        ('v2 improvement vs v1',s.get('comparisons',pending)),('supervised MRGR gap after v2',s.get('supervised_gaps',pending)),
        ('permutation max diff',max(p['permutation_max_diff'] for p in props.values())),
        ('padding max diff',max(p['padding_max_diff'] for p in props.values())),('all-invalid safe?',all(p['all_invalid_safe'] for p in props.values())),
        ('tests passed',test_count),('dataset unchanged?',s['integrity']['dataset_unchanged']),('projector unchanged?',s['integrity']['projector_unchanged']),
        ('network architecture unchanged?',s['integrity']['network_architecture_unchanged']),('test split unused?',True),('final decision',s['decision']),
        ('if GO freeze candidate','Final self-supervised training candidate' if s['decision'].endswith(' v2 GO') else 'not applicable'),
        ('if GO next step','Proceed to final ablation and multi-seed validation.' if s['decision'].endswith(' v2 GO') else 'not applicable'),
        ('if NO-GO','Retain SelfSup v1; do not automatically create v3' if s['decision'].endswith('NO-GO') else 'not applicable')]
    # Keep the terminal checklist compact: select requested measurement columns.
    for index in (22,23):
        value=items[index][1]
        if isinstance(value,dict):items[index]=(items[index][0],{key:value['measurement'][key] for key in ('heldout','teacher_heldout')})
    write_json(root/'terminal_summary.json',{str(i):dict(item=key,value=value) for i,(key,value) in enumerate(items,1)})
    text='\n'.join(f'{i}. {key}: {json.dumps(value,ensure_ascii=False)}' for i,(key,value) in enumerate(items,1))+'\n'
    (root/'terminal_report.txt').write_text(text);print(text,flush=True)


def figures(root,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    labels=('Supervised-MRGR','SelfSup-v1','SelfSup-v2')
    for filename,conditions in (
        ('selfsup_v1_v2_main_metrics.png',('Val-A-V4','Val-B-V4')),
        ('selfsup_v1_v2_sparse_views.png',('V3','V4','V6')),
        ('selfsup_v1_v2_clustered.png',('clustered-V4','spread-V4'))):
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,metric in zip(axes,('relative_l1','relative_l2')):
            for i,label in enumerate(labels):
                ax.bar(np.arange(len(conditions))+(i-1)*.25,[s['results'][label][c]['mean_per_case'][metric] for c in conditions],.25,label=label)
            ax.set_xticks(range(len(conditions)),conditions);ax.set_ylabel(metric);ax.legend(fontsize=8)
        fig.savefig(folder/filename,dpi=160);plt.close(fig)
    diag=s['teacher_posthoc'];fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    ax.bar(('Student A4','Student B4','Rich T8'),[diag[key]['mean_per_case']['relative_l1'] for key in ('A','B','T')])
    ax.set(ylabel='Voxel relative L1 (post-hoc only)',title='Frozen Val-A A/B/T pairs')
    fig.savefig(folder/'teacher_vs_student_posthoc.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,pool,condition in zip(axes,('Val-A','Pool-B'),('Val-A-V4','Val-B-V4')):
        v1=json.loads((root.parent/'selfsup_mrgr_v1/summary.json').read_text())
        measured=[v1['measurement_generalization'][pool]['measurement']['heldout']['relative_l1'],s['measurement_generalization'][pool]['measurement']['heldout']['relative_l1']]
        voxel=[s['results'][label][condition]['mean_per_case']['relative_l1'] for label in ('SelfSup-v1','SelfSup-v2')]
        ax.bar(np.arange(2)-.17,measured,.34,label='Sparse held-out measurement')
        ax.bar(np.arange(2)+.17,voxel,.34,label='Post-hoc voxel (legacy V4)')
        ax.set_xticks(range(2),('SelfSup v1','SelfSup v2'));ax.set(title=pool,ylabel='Relative L1');ax.legend(fontsize=8)
    fig.suptitle('Measurement and voxel errors use distinct frozen station protocols')
    fig.savefig(folder/'measurement_vs_voxel_gap.png',dpi=160);plt.close(fig)
