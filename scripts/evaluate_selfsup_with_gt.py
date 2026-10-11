#!/usr/bin/env python3
"""Post-hoc scientific evaluation: volumetric GT is accessible only after freeze."""
import copy
import datetime
import json
import subprocess
import sys
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import DualMeasurementDataset
from geophy_set3d.evaluation.protocols import evaluation_dataset,load_model
from geophy_set3d.evaluation.analysis import aggregate_draws,read_jsonl,write_csv
from geophy_set3d.evaluation.mrgr import evaluate_stages
from geophy_set3d.evaluation.measurement_diagnostics import set_properties,feedback_gradients
from geophy_set3d.evaluation.research_integrity import check_integrity
from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric,zero_volume_baseline,move_dual_batch
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.utils import load_config,seed_everything,write_json

ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1')
REFERENCE=ROOT.parent/'oari_vs_mrgr_300'
REPO=Path(__file__).resolve().parents[1]
CONDITIONS=(('Val-A-V4',4,'reference','train',1),('Val-B-V4',4,'reference','ood',1),('V3',3,'random','train',5),
            ('V4',4,'random','train',5),('V6',6,'random','train',5),('clustered-V4',4,'clustered','train',5),('spread-V4',4,'spread','train',1))
METRICS=('mae','rmse','relative_l1','relative_l2')


def require_frozen(config):
    freeze=json.loads((ROOT/'frozen_checkpoint.json').read_text())
    training=json.loads((Path(config['experiment']['output_root'])/'logs'/config['experiment']['name']/'training_summary.json').read_text())
    if training['completed_epochs']!=300 or not training['completed'] or training['selection_metric']!='mean held-out measurement relative L1':
        raise RuntimeError('GT evaluation requires all 300 epochs and measurement-only selection')
    if freeze['sha256']!=digest(freeze['checkpoint']) or freeze['completed_epochs']!=300 or freeze['volumetric_gt_used']:
        raise RuntimeError('Selfsup checkpoint is not correctly frozen')
    if training['gt_access']['gt_open_attempts'] or training['gt_access']['supervised_checkpoint_open_attempts']:
        raise RuntimeError('GT-free provenance violation')
    return freeze,training


def main():
    config=load_config('configs/train_selfsup_oari_mrgr_v1.yaml');freeze,training=require_frozen(config)
    seed_everything(20261011);torch.set_num_threads(2)
    model,ckpt=load_model(freeze['checkpoint'],config['training']['device'])
    assert ckpt['config']==config and ckpt['volumetric_gt_used'] is False and ckpt['training_signal']=='TDLAS measurements only'
    assert sum(p.numel() for p in model.parameters())==332070
    # Read only projections first; this validation replay cannot select another checkpoint.
    val=make_dataset(config,validation=True);val_loader=make_loader(val,config)
    measurements={'Val-A':run_epoch(model,val_loader,config)}
    baseline=zero_volume_baseline(val_loader,config)
    assert abs(selection_metric(measurements['Val-A'])-training['best_val_heldout_relative_l1']) <= 1e-8+abs(training['best_val_heldout_relative_l1'])*1e-5
    poolb=DualMeasurementDataset(config['dataset']['root'],split='val',geometry_pool='ood',evaluation_only=True,
        a_range=(4,4),b_range=(4,4),hold_views=4,base_seed=20261011,normalization=config['normalization']['method'])
    measurements['Pool-B']=run_epoch(model,make_loader(poolb,config),config)
    batch=move_dual_batch(dual_measurement_collate([val[0]]),config['training']['device'])
    properties=set_properties(model,batch['A']);gradients=feedback_gradients(model,batch['A'],batch['H'])
    reference=json.loads((REFERENCE/'summary.json').read_text())
    assert digest(reference['checkpoint'])==reference['checkpoint_sha256']
    boundary_path=ROOT/'posthoc_gt_boundary.json'
    if not boundary_path.exists():
        write_json(boundary_path,dict(first_posthoc_gt_evaluation_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            checkpoint_hash_frozen_before_gt=True,sha256=freeze['sha256'],best_epoch_frozen=freeze['best_epoch'],
            completed_epochs_before_gt=300,checkpoint_selection_metric=training['selection_metric'],
            method_changes_after_gt_forbidden=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    # This is the first location permitted to construct a volume-loading dataset.
    results={'Supervised-OARI':copy.deepcopy(reference['results']['OARI']),
             'Supervised-MRGR':copy.deepcopy(reference['results']['C1']),'SelfSup-C0':{},'SelfSup-C1':{}}
    voxel_config=copy.deepcopy(config)
    voxel_config['loss']=load_config('configs/train_supervised_oari_mrgr_300.yaml')['loss']
    # This loss is a post-hoc diagnostic only, never an optimization/selection signal.
    output=ROOT/config['experiment']['name'];table=[];projection_rows=[]
    for condition,views,selection,pool,draws in CONDITIONS:
        collected={k:[] for k in ('C0','C1')};allrows={k:[] for k in collected}
        for draw in range(draws):
            suffix=Path(condition)/f'draw{draw}' if draws==5 else Path(condition)
            directory=output/suffix
            if all((directory/k/'metrics.json').exists() for k in collected):
                sums={k:json.loads((directory/k/'metrics.json').read_text()) for k in collected}
                rows={k:read_jsonl(directory/k/'per_case_metrics.jsonl') for k in collected}
            else:
                sums,rows=evaluate_stages(model,evaluation_dataset(config,pool=pool,views=views,selection=selection,draw=draw),voxel_config,
                    directory,condition=condition,draw=draw)
            refs=read_jsonl(REFERENCE/'supervised_oari_mrgr_300'/suffix/'C1/per_case_metrics.jsonl')
            for k in collected:
                assert [(r['case_id'],r['selected_station_ids']) for r in rows[k]]==[(r['case_id'],r['selected_station_ids']) for r in refs]
                collected[k].append(sums[k]);allrows[k].extend(rows[k])
                projection_rows.append(dict(condition=condition,draw_id=draw,stage=k,**sums[k]['observed']))
            print(f'post-hoc {condition} draw{draw}: voxel C1 L1={sums["C1"]["mean_per_case"]["relative_l1"]:.8f}',flush=True)
        for k in collected:
            results[f'SelfSup-{k}'][condition]=collected[k][0] if draws==1 else aggregate_draws(collected[k],allrows[k])
        for label in results:
            table.append(dict(condition=condition,model=label,draws=draws,**{k:results[label][condition]['mean_per_case'][k] for k in METRICS}))
    gaps=[dict(condition=k,supervised_l1=results['Supervised-MRGR'][k]['mean_per_case']['relative_l1'],
               selfsup_l1=results['SelfSup-C1'][k]['mean_per_case']['relative_l1'],
               absolute_l1_gap=results['SelfSup-C1'][k]['mean_per_case']['relative_l1']-results['Supervised-MRGR'][k]['mean_per_case']['relative_l1'],
               relative_l1_ratio=results['SelfSup-C1'][k]['mean_per_case']['relative_l1']/results['Supervised-MRGR'][k]['mean_per_case']['relative_l1']) for k in results['SelfSup-C1']]
    # Operational evaluation reference declared before GT access (see effective_protocol).
    criteria=dict(strict_gt_free_training=not training['gt_access']['gt_file_opened'],
        gate_passed=json.loads((ROOT/'gate_summary.json').read_text())['passed'],
        val_heldout_better_than_zero=selection_metric(measurements['Val-A'])<baseline['relative_l1'],
        unseen_geometry_heldout_better_than_zero=selection_metric(measurements['Pool-B'])<1.,
        posthoc_val_a_voxel_better_than_zero=results['SelfSup-C1']['Val-A-V4']['mean_per_case']['relative_l1']<1.,
        posthoc_val_b_voxel_better_than_zero=results['SelfSup-C1']['Val-B-V4']['mean_per_case']['relative_l1']<1.,
        output_not_collapsed=measurements['Val-A']['output']['std']>1e-6 and measurements['Val-A']['output']['case_to_case_mean_absolute_difference']>1e-6,
        set_and_feedback_checks=properties['passed'] and gradients['passed'])
    summary=dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model='GeoLift-OARI-MRGR-Res3D',training=training,checkpoint=freeze,posthoc_boundary=json.loads(boundary_path.read_text()),
        results=results,measurement_generalization=measurements,zero_volume_val_baseline=baseline,supervised_gaps=gaps,
        set_properties=properties,feedback_gradients=gradients,parameter_count=332070,
        volumetric_gt_used_for_training=False,volumetric_gt_used_for_selection=False,posthoc_gt_evaluation_only=True,
        criteria=criteria,decision='SelfSup-MRGR v1 GO' if all(criteria.values()) else 'SelfSup-MRGR v1 NO-GO',
        decision_note='Operational viability references, not parity with supervised MRGR, significance or uniqueness',
        integrity=check_integrity(ROOT,config))
    assert digest(freeze['checkpoint'])==freeze['sha256']
    write_json(ROOT/'summary.json',summary);write_csv(ROOT/'voxel_metrics_evaluation_only.csv',table)
    write_csv(ROOT/'projection_generalization.csv',projection_rows);write_csv(ROOT/'supervised_gap.csv',gaps)
    measurement_rows=[dict(pool=pool,domain=domain,**values) for pool,v in measurements.items() for domain,values in v['measurement'].items()]
    write_csv(ROOT/'measurement_generalization.csv',measurement_rows)
    from geophy_set3d.evaluation.selfsup_report import render_selfsup_report
    render_selfsup_report(ROOT,REPO)


if __name__=='__main__':main()
