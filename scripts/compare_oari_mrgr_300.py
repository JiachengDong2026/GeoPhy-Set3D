#!/usr/bin/env python3
"""Evaluate frozen OARI vs trained MRGR C0/C1, with exact case/station pairing."""
import copy
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest, source_snapshot
from geophy_set3d.data.measurement_sampling import attach_hold
from geophy_set3d.evaluation.protocols import evaluation_dataset, load_model
from geophy_set3d.evaluation.analysis import aggregate_draws, read_jsonl, write_csv
from geophy_set3d.evaluation.mrgr import evaluate_stages, connectivity_diagnostics
from geophy_set3d.training.utils import load_config, seed_everything, write_json
from geophy_set3d.training.comparison import assert_fair_oari_mrgr

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/oari_vs_mrgr_300')
REFERENCE = ROOT.parent/'b2_vs_oari_300'
METRICS = ('mae', 'rmse', 'relative_l1', 'relative_l2')
CONDITIONS = [('Val-A-V4', 4, 'reference', 'train', 1), ('Val-B-V4', 4, 'reference', 'ood', 1),
              ('V3', 3, 'random', 'train', 5), ('V4', 4, 'random', 'train', 5),
              ('V6', 6, 'random', 'train', 5), ('clustered-V4', 4, 'clustered', 'train', 5),
              ('spread-V4', 4, 'spread', 'train', 1)]


def integrity(config):
    before = json.loads((ROOT/'inventory_before.json').read_text())
    unchanged = lambda mapping, parent: all((parent/p).is_file() and digest(parent/p)==sha for p,sha in mapping.items())
    current = source_snapshot(config['dataset']['root'])
    old, protected = before['existing_artifacts'], before['protected_code']
    result = dict(dataset_files=len(current), dataset_unchanged=current == before['dataset'],
        all_historical_artifacts_unchanged=unchanged(old, ROOT.parent.parent),
        protected_code_unchanged=unchanged(protected, REPO),
        oari_source_checkpoint_report_unchanged=unchanged({p:s for p,s in protected.items() if 'oari' in p}, REPO) and
            unchanged({p:s for p,s in old.items() if 'supervised_oari_300' in p or 'b2_vs_oari_300' in p}, ROOT.parent.parent),
        b0_b1_b2_unchanged=unchanged({p:s for p,s in old.items() if any(k in p for k in ('b0','b1','b2'))}, ROOT.parent.parent),
        b3_artifacts_unchanged=unchanged({p:s for p,s in old.items() if 'b3' in p}, ROOT.parent.parent),
        projector_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/physics/')}, REPO),
        numerical_training_sources_unchanged=unchanged(json.loads((ROOT/'training_source_manifest.json').read_text())['sha256'], REPO),
        pool_b_evaluation_only=True, test_used=False)
    result['passed'] = all(v for k,v in result.items() if k.endswith('_unchanged'))
    write_json(ROOT/'integrity_summary.json', result)
    assert result['passed'], 'Protected sources/data/artifacts changed'
    return result


def paired(rows, reference):
    return [(r['case_id'], r['selected_station_ids']) for r in rows] == [(r['case_id'], r['selected_station_ids']) for r in reference]


def main():
    config = load_config('configs/train_supervised_oari_mrgr_300.yaml')
    oari_config = load_config('configs/train_supervised_oari_300.yaml')
    assert_fair_oari_mrgr(oari_config, config)
    seed_everything(20261011); torch.set_num_threads(2)
    base = Path(config['experiment']['output_root']); name = config['experiment']['name']; logs = base/'logs'/name
    training = json.loads((logs/'training_summary.json').read_text())
    assert training['completed_epochs']==300 and training['stopping_reason']=='max_epochs', 'Complete all epochs before Pool B evaluation'
    assert json.loads((logs/'validation_station_ids.json').read_text()) == json.loads((base/'logs/supervised_oari_300/validation_station_ids.json').read_text())
    checkpoint = base/'checkpoints'/name/'best.pt'; sha = digest(checkpoint)
    model, ckpt = load_model(checkpoint, config['training']['device'])
    assert ckpt['config'] == config
    oari = json.loads((REFERENCE/'summary.json').read_text())['models']['OARI-300']
    assert digest(oari['checkpoint']) == oari['checkpoint_sha256']
    output = ROOT/name; output.mkdir(exist_ok=True)
    protocol = dict(config=config, frozen_oari_config=oari_config, reference_summary_sha256=digest(REFERENCE/'summary.json'),
        checkpoint_sha256=sha, checkpoint_selection='minimum Val-A fixed V4 C1 supervised loss',
        formal_initialization='fresh random same-seed shared modules', refinement_iterations=1,
        projector_samples=96, projector_view_chunk=2, residual_scale_ppm_m=32284.894580078173,
        objective='supervised_loss(C1,GT64) only', residual_clamp=False, c0_detached=False,
        evaluation_seed=20261011, draw_ids=list(range(5)), diagnostics_indices=[0,18,35],
        test_used=False, pool_b_evaluation_only=True, heldout_training=False, retrain_oari=False)
    (ROOT/'effective_protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    results = {k: {} for k in ('OARI', 'C0', 'C1')}; main_rows=[]; stage_rows=[]; sparse_rows=[]; cluster_rows=[]; case_rows=[]; projection_rows=[]; hold_rows=[]
    for condition, views, selection, pool, draws in CONDITIONS:
        collected = {k: [] for k in ('C0','C1')}; allrows={k: [] for k in collected}
        for draw in range(draws):
            suffix = Path(condition)/f'draw{draw}' if draws==5 else Path(condition)
            refdir = REFERENCE/'supervised_oari_300'/suffix
            refs = read_jsonl(refdir/'per_case_metrics.jsonl')
            ds = evaluation_dataset(config, pool=pool, views=views, selection=selection, draw=draw)
            heldout = condition == 'Val-A-V4'
            if heldout:
                ds = attach_hold(ds)
            diagnostic = ROOT/'diagnostics' if condition in ('V4','clustered-V4') and draw==0 else None
            cached = output/suffix
            if all((cached/k/'metrics.json').exists() for k in collected):
                sums={k:json.loads((cached/k/'metrics.json').read_text()) for k in collected}
                rows={k:read_jsonl(cached/k/'per_case_metrics.jsonl') for k in collected}
            else:
                sums, rows = evaluate_stages(model, ds, config, cached, condition=condition, draw=draw, heldout=heldout, diagnostics=diagnostic)
            for stage in collected:
                assert paired(rows[stage], refs), f'Unpaired {condition} draw{draw}'
                collected[stage].append(sums[stage]); allrows[stage].extend(rows[stage])
                projection_rows.append(dict(condition=condition, draw_id=draw, stage=stage, **sums[stage]['observed']))
                if heldout:
                    hold_rows.append(dict(condition=condition, stage=stage, **sums[stage]['heldout']))
                if condition=='clustered-V4':
                    cluster_rows.append(dict(model=stage, draw_id=draw, **{k:sums[stage]['mean_per_case'][k] for k in METRICS}))
                    case_rows.extend(dict(model=stage, draw_id=draw, case_id=r['case_id'], station_ids=json.dumps(r['selected_station_ids']), **{k:r[k] for k in METRICS}) for r in rows[stage])
            if condition=='clustered-V4':
                refsummary=json.loads((refdir/'metrics.json').read_text())
                cluster_rows.append(dict(model='OARI', draw_id=draw, **{k:refsummary['mean_per_case'][k] for k in METRICS}))
                case_rows.extend(dict(model='OARI', draw_id=draw, case_id=r['case_id'], station_ids=json.dumps(r['selected_station_ids']), **{k:r[k] for k in METRICS}) for r in refs)
            print(f'{condition} draw{draw}: C0={sums["C0"]["mean_per_case"]["relative_l1"]:.8f} C1={sums["C1"]["mean_per_case"]["relative_l1"]:.8f}', flush=True)
        results['OARI'][condition] = oari['evaluations'][condition] if condition.startswith('Val-') else oari['spread'] if condition=='spread-V4' else oari['sparse'][condition]
        for stage in collected:
            result = collected[stage][0] if draws==1 else aggregate_draws(collected[stage], allrows[stage])
            if draws==5:
                result['observed']={k:float(np.mean([s['observed'][k] for s in collected[stage]])) for k in ('relative_l1','relative_l2')}
            results[stage][condition]=result
        for label in results:
            row=dict(condition=condition, model=label, draws=draws, **{k:results[label][condition]['mean_per_case'][k] for k in METRICS})
            stage_rows.append(row)
            if condition.startswith('Val-'): main_rows.append(row)
            else: sparse_rows.append(row)
    shortcuts={}; shortcut_rows=[]
    for mode in ('normal','zero','wrong_pairing','signflip'):
        if mode=='normal':
            s=results['C1']['Val-A-V4']
        else:
            s, _=evaluate_stages(model, evaluation_dataset(config), config, output/'shortcuts'/mode, condition='Val-A-V4', mode=mode)
            s=s['C1']
        shortcuts[mode]=s
        shortcut_rows.append(dict(residual_mode=mode, **{k:s['mean_per_case'][k] for k in METRICS}))
    diag=connectivity_diagnostics(model, evaluation_dataset(config), config)
    write_json(ROOT/'connectivity_diagnostics.json',diag)
    improvements=[]
    for condition in results['OARI']:
        a,b,c=[results[k][condition]['mean_per_case']['relative_l1'] for k in ('OARI','C0','C1')]
        improvements.append(dict(condition=condition, oari_to_c1_absolute_l1_gain=a-c, c0_to_c1_absolute_l1_gain=b-c, oari_to_c1_relative_gain_percent=100*(a-c)/a))
    criteria=dict(val_a_not_materially_worse=results['C1']['Val-A-V4']['mean_per_case']['relative_l1']<=results['OARI']['Val-A-V4']['mean_per_case']['relative_l1']*1.02,
                  val_b_not_materially_worse=results['C1']['Val-B-V4']['mean_per_case']['relative_l1']<=results['OARI']['Val-B-V4']['mean_per_case']['relative_l1']*1.02,
                  v3_or_clustered_gain=any(r['oari_to_c1_absolute_l1_gain']>=.005 for r in improvements if r['condition'] in ('V3','clustered-V4')),
                  refinement_contribution=any(r['c0_to_c1_absolute_l1_gain']>=.003 for r in improvements if r['condition'] in ('Val-A-V4','V3','clustered-V4')),
                  mask_permutation_gradient_connectivity=diag['passed'])
    for metric in METRICS:
        expected=training['best_val_metrics']['mean_per_case'][metric]; actual=results['C1']['Val-A-V4']['mean_per_case'][metric]
        assert abs(expected-actual)<=1e-7+abs(expected)*1e-5, 'Val-A checkpoint metrics mismatch'
    check=integrity(config); assert digest(checkpoint)==sha
    summary=dict(completed=True, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_source_commit=json.loads((ROOT/'training_source_manifest.json').read_text())['commit'],
        model_name='GeoLift-OARI-MRGR-Res3D', parameter_count=sum(p.numel() for p in model.parameters()),
        oari_parameter_count=oari['parameter_count'], checkpoint=str(checkpoint), checkpoint_sha256=sha,
        best_epoch=ckpt['epoch']+1, training=training, protocol=protocol, results=results,
        shortcuts=shortcuts, connectivity=diag, improvements=improvements, criteria=criteria,
        decision='MRGR GO' if all(criteria.values()) else 'MRGR NO-GO',
        final_architecture_candidate='v2' if all(criteria.values()) else 'v1 (OARI)',
        integrity=check, oari_runtime=json.loads((REFERENCE/'summary.json').read_text())['runtime']['OARI-300'],
        quick_gate=json.loads((base/'logs'/f'{name}_gate'/'gate_summary.json').read_text()),
        actual_case_station_pairings_identical=True, test_used=False, pool_b_evaluation_only=True)
    write_json(ROOT/'summary.json',summary)
    for file,rows in [('main_metrics',main_rows),('stage_metrics',stage_rows),('sparse_views',sparse_rows),('clustered_v4',cluster_rows),
                      ('clustered_v4_per_case',case_rows),('projection_before_after',projection_rows),('heldout_projection',hold_rows),('residual_shortcuts',shortcut_rows),('improvements',improvements)]:
        write_csv(ROOT/(file+'.csv'),rows)
    if summary['decision']=='MRGR GO':
        write_json(ROOT/'final_architecture_candidate_v2.json',dict(model=summary['model_name'], checkpoint=str(checkpoint),sha256=sha,source_commit=summary['training_source_commit'],criteria=criteria))
    from geophy_set3d.evaluation.mrgr_report import render_report
    render_report(ROOT,REPO)


if __name__=='__main__':
    main()
