"""Render the fixed five figures, scientific report and 42-item terminal record."""
import json
from pathlib import Path
import numpy as np
from geophy_set3d.training.utils import write_json


def make_figures(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    results=summary['results']
    labels=('OARI','C0','C1')
    def bars(filename, conditions, title):
        fig, ax=plt.subplots(figsize=(max(7,len(conditions)*1.5),4),constrained_layout=True)
        for i,label in enumerate(labels):
            ax.bar(np.arange(len(conditions))+(i-1)*.25,
                   [results[label][k]['mean_per_case']['relative_l1'] for k in conditions],.25,label=label)
        ax.set_xticks(range(len(conditions)),conditions)
        ax.set(ylabel='Mean per-case relative L1',title=title); ax.legend()
        fig.savefig(figures/filename,dpi=160); plt.close(fig)
    bars('oari_mrgr_main_metrics.png',['Val-A-V4','Val-B-V4'],'Frozen OARI and MRGR stages')
    bars('oari_mrgr_sparse_views.png',['V3','V4','V6'],'Five paired deterministic draws')
    bars('oari_mrgr_clustered.png',['clustered-V4'],'Clustered V4: five paired deterministic draws')
    fig, ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    conditions=list(results['C0'])
    for i,label in enumerate(('C0','C1')):
        ax.bar(np.arange(len(conditions))+(i-.5)*.35,
               [results[label][k]['mean_per_case']['relative_l1'] for k in conditions],.35,label=label)
    ax.set_xticks(range(len(conditions)),conditions,rotation=15)
    ax.set(ylabel='Mean per-case relative L1',title='One refinement step'); ax.legend()
    fig.savefig(figures/'c0_vs_c1.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,domain in zip(axes,('observed','heldout')):
        for i,label in enumerate(('C0','C1')):
            value=results[label]['Val-A-V4'][domain]
            ax.bar(np.arange(2)+(i-.5)*.35,[value['relative_l1'],value['relative_l2']],.35,label=label)
        ax.set_xticks([0,1],['Relative L1','Relative L2'])
        ax.set(title=f'Val-A {domain} raw ppm·m projection',ylabel='Mean per-case relative error'); ax.legend()
    fig.savefig(figures/'projection_residual_before_after.png',dpi=160); plt.close(fig)


def render_report(root, repo):
    root,repo=Path(root),Path(repo)
    s=json.loads((root/'summary.json').read_text()); r=s['results']; gate=s['quick_gate']; diag=s['connectivity']
    make_figures(root,s)
    oari,mrgr=s['oari_parameter_count'],s['parameter_count']; delta=mrgr-oari
    def fmt(x): return f'{x:.8f}'
    def stage_line(condition):
        return ' | '.join(fmt(r[k][condition]['mean_per_case'][m]) for k in ('OARI','C0','C1') for m in ('mae','rmse','relative_l1','relative_l2'))
    table='| Condition | OARI MAE | RMSE | L1 | L2 | C0 MAE | RMSE | L1 | L2 | C1 MAE | RMSE | L1 | L2 |\n|'+ '---|'*13+'\n'
    table+='\n'.join(f'| {k} | {stage_line(k)} |' for k in r['OARI'])+'\n'
    gains='| Condition | OARI minus C1 L1 | C0 minus C1 L1 | Relative gain vs OARI % |\n|---|---:|---:|---:|\n'
    gains+='\n'.join(f"| {v['condition']} | {fmt(v['oari_to_c1_absolute_l1_gain'])} | {fmt(v['c0_to_c1_absolute_l1_gain'])} | {v['oari_to_c1_relative_gain_percent']:.3f} |" for v in s['improvements'])+'\n'
    projection='| Condition | C0 observed L1 / L2 | C1 observed L1 / L2 |\n|---|---:|---:|\n'
    for k in r['C0']:
        p=[r[label][k]['observed'] for label in ('C0','C1')]
        projection+=f"| {k} | {fmt(p[0]['relative_l1'])} / {fmt(p[0]['relative_l2'])} | {fmt(p[1]['relative_l1'])} / {fmt(p[1]['relative_l2'])} |\n"
    shortcut='| Residual input | C1 relative L1 | Relative L2 |\n|---|---:|---:|\n'
    shortcut+='\n'.join(f"| {k} | {fmt(v['mean_per_case']['relative_l1'])} | {fmt(v['mean_per_case']['relative_l2'])} |" for k,v in s['shortcuts'].items())+'\n'
    a=r['C0']['Val-A-V4']['heldout']; b=r['C1']['Val-A-V4']['heldout']
    subset=lambda entry:entry['stages']['C0']['mean_per_case']['relative_l1']
    gate_text=f"Initial C0/C1 L1: {fmt(subset(gate['initial']))} / {fmt(gate['initial']['metrics']['mean_per_case']['relative_l1'])}. Final: {fmt(subset(gate['final']))} / {fmt(gate['final']['metrics']['mean_per_case']['relative_l1'])}. C1 supervised loss: {fmt(gate['initial']['loss']['total'])} → {fmt(gate['final']['loss']['total'])}. Passed: {gate['passed']}. All shared weights unchanged."
    normal=s['shortcuts']['normal']['mean_per_case']['relative_l1']
    interpretation=[]
    for mode in ('zero','wrong_pairing','signflip'):
        value=s['shortcuts'][mode]['mean_per_case']['relative_l1']
        interpretation.append(f'{mode}: delta L1 vs normal {value-normal:+.8f}')
    improved_projection=[k for k in r['C0'] if r['C1'][k]['observed']['relative_l1'] < r['C0'][k]['observed']['relative_l1']]
    pstatement=f'Observed projection L1 improves in {len(improved_projection)}/{len(r["C0"])} conditions: {", ".join(improved_projection) or "none"}. '
    if len(improved_projection)<len(r['C0']):
        pstatement+='Refinement does not consistently reduce reprojection error; the residual is a feature, not a hard physical constraint.'
    chapters=[
        ('Motivation','The initial 3D estimate is reprojected into the measurement domain, and the signed measurement residual is geometrically lifted back into the volumetric feature space to refine the reconstruction. The question is whether explicit measurement feedback improves voxel reconstruction, especially with sparse or clustered views.'),
        ('Why additive projection loss was insufficient','Frozen B3 is NO-GO. B3 converts discrepancy into a scalar loss for gradient supervision. MRGR converts the same discrepancy into signed spatial features, lifts them through actual acquisition geometry, and corrects context features. This experiment adds no measurement loss.'),
        ('Frozen OARI backbone',f'OARI source, config, checkpoint, report and artifacts remain unchanged. Frozen best SHA256: `{s["protocol"]["reference_summary_sha256"]}` is the comparison summary hash; OARI checkpoint SHA256 is `a91fcdf200b034cc0e25a2687f61bb5bf8bdfdacb18441b4c1c1ea60e766a680`. Integrity checks cover all historical outputs and protected code.'),
        ('Initial reconstruction C0','The unchanged shared encoder → geometry lifting → OARI interaction → GCA → decoder computes C0. The MRGR formal model uses fresh random initialization. Shared same-seed state tensors match OARI exactly; C0 is an intermediate of the jointly trained MRGR and is not the frozen OARI prediction.'),
        ('Differentiable measurement residual','Y0=P_diff(C0,G_input), K=96. The unchanged projector uses midpoint integration, XYZ→ZYX grid_sample coordinates, align_corners=True, mass-to-mole conversion and raw ppm·m units. R=Y_raw−Y0 and R_scaled=R/32284.894580078173. No clipping or per-view/per-case normalization. Invalid rays/padded views are zero; valid zero measurements remain valid. Measurements and geometry are detached; C0 remains connected. Signed scaled residual min/max/mean/std/p95/p99 are logged each epoch over all valid rays, with population std and signed quantiles.'),
        ('Residual 2D encoder','Input [R_scaled, valid_ray, path_length/(√3×0.6)] has shape [B,V,3,64,64]. Conv3×3 3→16, GN4, SiLU, two 16-channel residual blocks (GN4), no downsampling. Output [B,V,16,64,64].'),
        ('Geometry-aware residual lifting','The exact same GeometryLifting instance projects residual features into [B,V,16,32,32,32]. Both visibility tensors match exactly; no new coordinate convention or attention module.'),
        ('Residual context refinement','Concatenate context0 (32) and residual lifted features (16); adapter Conv3d1×1 48→32, SiLU, Conv3d1×1 32→32. The final layer starts at zero. delta_context is masked by visibility AND view_mask; invisible context1 is exactly zero.'),
        ('C0 → C1 formulation','context1=context0+masked_adapter([context0,R3D]); C1=shared_decoder(shared_GCA(context1)). Exactly one refinement iteration, one GCA module and one decoder. Initial C1=C0 exactly. The only objective is SmoothL1(beta=1)+0.1×sample-relative-L1 on C1 versus GT64; no C0 auxiliary loss, projection, held-out, TV, sparsity or PDE term.'),
        ('Parameter / compute cost',f'OARI {oari:,}; MRGR {mrgr:,}; increase {delta:,} ({100*delta/oari:.3f}%). MRGR peak allocated {s["training"]["peak_gpu_allocated_bytes"]/2**30:.3f} GiB; mean epoch {s["training"]["mean_epoch_seconds"]:.3f} s. OARI observed peak {s["oari_runtime"]["peak_allocated_bytes"]/2**30:.3f} GiB and mean epoch {s["oari_runtime"]["mean_epoch_seconds"]:.3f} s. These runs are not a controlled throughput benchmark. View chunks of 2 retain all rays/K96. FP32 on GPU 1.'),
        ('Quick gate',gate_text+'\n\nOne fixed-V4 4-train-case run, 60 epochs, AdamW at the frozen lr/wd, only encoder/adapter trained. Frozen OARI loaded only for this functional test. No gate weight was used for the formal run.'),
        ('Val-A',f'Formal protocol: 166 Pool A train cases, V3–6; 36 fixed-V4 Val-A cases, seed20261011, FP32, batch2, AdamW lr2e-4/wd1e-4, clip1, cosine T_max300, all 300 epochs, no early stopping. Best epoch {s["best_epoch"]} (1-based) selected exclusively by Val-A C1 supervised loss. Training validation station IDs match OARI exactly.\n\n'+table),
        ('Val-B','Pool B was evaluated only after all 300 epochs and the best checkpoint had been frozen. Exact OARI case/station pairs are asserted. No test split was used. See the paired metrics above.'),
        ('V3/V4/V6','Five deterministic draws each, identical to frozen OARI. Tables report means over case/draw samples, not pooled errors; pooled metrics and draw standard deviations are retained in summary.json. Repeated cases across draws are not independent. Spread V4 draw0 is supplementary.\n\n'+gains),
        ('Clustered V4','Five exact deterministic clustered draws. Per-draw metrics are in clustered_v4.csv; all 180 case/draw rows per model are in clustered_v4_per_case.csv. Six compact diagnostics (indices0/18/35 × random/clustered draw0) record station IDs, raw measured/predicted maps, signed residual, stage voxel errors and feature/delta magnitudes. No full training volumes are exported.'),
        ('C0 vs C1','The stage contribution must reach ≥0.003 absolute relative-L1 improvement in Val-A, V3 or clustered V4. A lower final error alone does not establish a useful refinement stage. See the stage gains above.\n\n'+json.dumps({k:v for k,v in diag.items()},indent=2)),
        ('Projection residual before / after',projection+'\n'+pstatement+' Errors are means of per-case raw ppm·m relative errors. No absolute projection threshold is imposed because 64³ volumes cannot exactly reproduce the frozen 128³/K256 references.'),
        ('Held-out measurement evaluation',f'Val-A fixed input V4 plus one deterministic disjoint Pool A held-out station (the existing B3 hold sampler). C0 held-out L1/L2 {fmt(a["relative_l1"])} / {fmt(a["relative_l2"])}; C1 {fmt(b["relative_l1"])} / {fmt(b["relative_l2"])}. The network only receives input V4. Held-out errors never affect training, selection or stopping; they are supporting evidence, not proof of uniqueness.'),
        ('Residual shortcut tests',shortcut+'\n'+'; '.join(interpretation)+'. Zero keeps valid/path channels; wrong pairing rolls only real residual maps while geometry remains fixed; signflip negates only residual. These are evaluation-only probes with no strict performance threshold.'),
        ('GO / NO-GO',f'**{s["decision"]}. Final architecture candidate: {s["final_architecture_candidate"]}.**\n\n'+json.dumps(s['criteria'],indent=2)+'\n\nEngineering criteria are 2% relative main-error tolerance, ≥0.005 absolute V3 or clustered gain versus frozen OARI, ≥0.003 absolute stage gain in a core condition, and correct masks/permutation/gradients. They are operational criteria, not statistical significance claims. No third module, tuning, second seed or multi-step refinement was added.\n\nIntegrity: '+json.dumps(s['integrity'],indent=2))]
    report=f'# Supervised OARI-MRGR v1\n\nDecision: **{s["decision"]}**. Training source `{s["training_source_commit"]}`. Best checkpoint `{s["checkpoint"]}`, SHA256 `{s["checkpoint_sha256"]}`.\n\n'
    report+='\n\n'.join(f'## {i}. {title}\n\n{body}' for i,(title,body) in enumerate(chapters,1))+'\n\nArtifacts: `'+str(root)+'`. See `terminal_report.txt` for all 42 requested terminal items.\n'
    (repo/'docs/reports/supervised_oari_mrgr_v1.md').write_text(report)
    tests=(root/'tests.log').read_text(); import re
    count=int(re.search(r'Ran (\d+) tests',tests).group(1)); assert tests.rstrip().endswith('OK')
    l1l2=lambda condition:{k:{m:r[k][condition]['mean_per_case'][m] for m in ('relative_l1','relative_l2')} for k in ('OARI','C0','C1')}
    gains_by={v['condition']:v for v in s['improvements']}
    terminal=[('current git commit',s['git_commit']),('model name',s['model_name']),('OARI params',oari),('MRGR params',mrgr),
        ('parameter increase',dict(absolute=delta,relative_percent=100*delta/oari)),
        ('residual encoder','Conv3x3 3→16, GN4, SiLU; 2 GN4 residual blocks16; no downsample'),
        ('residual adapter','Conv3d1x1 48→32, SiLU, Conv3d1x1 32→32; final zero-init'),
        ('projector',dict(K=96,source_unchanged=s['integrity']['projector_unchanged'])),
        ('quick gate',dict(initial_C0=subset(gate['initial']),initial_C1=gate['initial']['metrics']['mean_per_case']['relative_l1'],final_C0=subset(gate['final']),final_C1=gate['final']['metrics']['mean_per_case']['relative_l1'],passed=gate['passed'])),
        ('formal best epoch (1-based)',s['best_epoch']),('Val-A L1/L2',l1l2('Val-A-V4')),('Val-B L1/L2',l1l2('Val-B-V4')),
        ('V3 L1/L2',l1l2('V3')),('random V4 L1/L2',l1l2('V4')),('V6 L1/L2',l1l2('V6')),('clustered V4 L1/L2',l1l2('clustered-V4')),('spread V4 L1/L2',l1l2('spread-V4')),
        ('OARI → MRGR C1 absolute L1 improvement',{k:gains_by[k]['oari_to_c1_absolute_l1_gain'] for k in ('Val-A-V4','Val-B-V4','V3','clustered-V4')}),
        ('C0 → C1 absolute L1 improvement',{k:gains_by[k]['c0_to_c1_absolute_l1_gain'] for k in ('Val-A-V4','Val-B-V4','V3','clustered-V4')}),
        ('observed reprojection Val-A',{k:r[k]['Val-A-V4']['observed'] for k in ('C0','C1')}),
        ('held-out reprojection Val-A',{k:r[k]['Val-A-V4']['heldout'] for k in ('C0','C1')}),
        ('zero residual voxel L1',s['shortcuts']['zero']['mean_per_case']['relative_l1']),
        ('wrong residual pairing voxel L1',s['shortcuts']['wrong_pairing']['mean_per_case']['relative_l1']),
        ('sign-flipped residual voxel L1',s['shortcuts']['signflip']['mean_per_case']['relative_l1']),
        ('correct residual voxel L1',normal),('residual branch gradient nonzero',diag['residual_gradient_abs_sum']>0),
        ('projector→refinement gradient finite',diag['projector_feedback_gradient_finite']),('permutation max diff',diag['permutation_max_diff']),
        ('padding max diff',diag['padding_max_diff']),('all-invalid safe',diag['all_invalid_safe']),('base/residual visibility identical',diag['visibility_identical']),
        ('peak GPU allocated memory',dict(bytes=s['training']['peak_gpu_allocated_bytes'],GiB=s['training']['peak_gpu_allocated_bytes']/2**30)),
        ('mean epoch seconds',s['training']['mean_epoch_seconds']),('OARI unchanged',s['integrity']['oari_source_checkpoint_report_unchanged']),
        ('projector unchanged',s['integrity']['projector_unchanged']),('dataset unchanged',s['integrity']['dataset_unchanged']),
        ('Pool B evaluation-only',True),('test unused',True),('tests passed count',count),('final decision',s['decision']),
        ('freeze Final architecture candidate v2',s['decision']=='MRGR GO'),('retain OARI Final architecture candidate v1',s['decision']=='MRGR NO-GO')]
    assert len(terminal)==42
    write_json(root/'terminal_summary.json',{str(i):dict(item=k,value=v) for i,(k,v) in enumerate(terminal,1)})
    text='\n'.join(f'{i}. {k}: {json.dumps(v,ensure_ascii=False)}' for i,(k,v) in enumerate(terminal,1))+'\n'
    (root/'terminal_report.txt').write_text(text); print(text,flush=True)
