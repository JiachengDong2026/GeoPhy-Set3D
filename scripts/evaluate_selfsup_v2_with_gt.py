#!/usr/bin/env python3
"""Frozen RVTC measurement evaluation followed by paired, evaluation-only GT metrics."""
import copy
import datetime
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.rich_measurement import RichMeasurementDataset,rich_measurement_collate
from geophy_set3d.evaluation.protocols import evaluation_dataset,load_model
from geophy_set3d.evaluation.analysis import aggregate_draws,read_jsonl,write_csv
from geophy_set3d.evaluation.mrgr import evaluate_stages
from geophy_set3d.evaluation.measurement_diagnostics import set_properties
from geophy_set3d.evaluation.selfsup_v2_integrity import check_integrity
from geophy_set3d.training.selfsup_v2 import (make_dataset,make_loader,run_epoch,selection_metric,
    move_rich_batch,audit_is_clean,teacher_gradient_diagnostic,SELECTION)
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import load_config,seed_everything,write_json

ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v2')
REFERENCE=ROOT.parent/'oari_vs_mrgr_300'
V1=ROOT.parent/'selfsup_mrgr_v1'
CONDITIONS=(('Val-A-V4',4,'reference','train',1),('Val-B-V4',4,'reference','ood',1),
            ('V3',3,'random','train',5),('V4',4,'random','train',5),('V6',6,'random','train',5),
            ('clustered-V4',4,'clustered','train',5),('spread-V4',4,'spread','train',1))
METRICS=('mae','rmse','relative_l1','relative_l2')


def require_frozen(config):
    freeze=json.loads((ROOT/'frozen_checkpoint.json').read_text())
    training=json.loads((Path(config['experiment']['output_root'])/'logs'/config['experiment']['name']/'training_summary.json').read_text())
    if training['completed_epochs']!=300 or not training['completed'] or training['selection_metric']!=SELECTION:
        raise RuntimeError('GT evaluation requires all 300 epochs and sparse measurement-only selection')
    if freeze['sha256']!=digest(freeze['checkpoint']) or freeze['completed_epochs']!=300 or freeze['volumetric_gt_used'] or freeze['selection_metric']!=SELECTION:
        raise RuntimeError('v2 checkpoint is not correctly frozen')
    if not audit_is_clean(training['gt_access']):raise RuntimeError('GT-free provenance violation')
    prior=training.get('prior_resume_audit')
    if prior and not audit_is_clean(prior):raise RuntimeError('Resume provenance violation')
    return freeze,training


def teacher_posthoc(model,dataset,config):
    """One diagnostic pass on the frozen Val-A A4/B4/T8 pairs; never selects weights."""
    metrics={key:VolumeMetrics() for key in ('A','B','T')}; rows=[]
    with torch.no_grad():
        for i in range(len(dataset)):
            raw=dataset[i]; batch=move_rich_batch(rich_measurement_collate([raw]),config['training']['device'])
            # Only this post-freeze diagnostic reads volumetric GT for these pairs.
            record=dataset.records[i]
            target=torch.from_numpy(np.load(Path(config['dataset']['root'])/record['gt_volume_64_path'],allow_pickle=False).copy())
            target=target.reshape(1,1,64,64,64).to(config['training']['device'])
            values={key:model(batch[key]) for key in metrics}
            for key in metrics:metrics[key].update(values[key],target,[record['case_id']])
            rows.append(dict(case_id=record['case_id'],pool='Val-A',evaluation_only=True,
                A_station_ids=raw['A']['input_station_indices'].tolist(),B_station_ids=raw['B']['input_station_indices'].tolist(),
                T_station_ids=raw['T']['input_station_indices'].tolist(),H_station_ids=raw['H']['input_station_indices'].tolist(),
                **{f'{key}_voxel_{m}':metrics[key].rows[-1][m] for key in metrics for m in ('relative_l1','relative_l2')}))
    result={key:value.compute() for key,value in metrics.items()}
    result['teacher_better_than_mean_student_fraction']=sum(row['T_voxel_relative_l1'] < .5*(row['A_voxel_relative_l1']+row['B_voxel_relative_l1']) for row in rows)/len(rows)
    result['teacher_generally_more_accurate']=(result['T']['mean_per_case']['relative_l1'] < .5*(result['A']['mean_per_case']['relative_l1']+result['B']['mean_per_case']['relative_l1']) and
                                              result['teacher_better_than_mean_student_fraction']>.5)
    return result,rows


def main():
    config=load_config('configs/train_selfsup_oari_mrgr_v2.yaml');freeze,training=require_frozen(config)
    seed_everything(20261011);torch.set_num_threads(2)
    model,ckpt=load_model(freeze['checkpoint'],config['training']['device'])
    assert ckpt['config']==config and ckpt['volumetric_gt_used'] is False and ckpt['training_signal']=='TDLAS measurements only'
    assert sum(p.numel() for p in model.parameters())==332070
    val=make_dataset(config,validation=True)
    protocol=json.loads((Path(config['experiment']['output_root'])/'logs'/config['experiment']['name']/'validation_station_ids.json').read_text())
    for i in range(len(val)):
        sample=val[i]
        assert {key:sample[key]['input_station_indices'].tolist() for key in ('A','B','H','T')}==protocol[sample['A']['case_id']]
    measurements={'Val-A':run_epoch(model,make_loader(val,config),config)}
    assert abs(selection_metric(measurements['Val-A'])-training['best_val_heldout_relative_l1'])<=1e-8+abs(training['best_val_heldout_relative_l1'])*1e-5
    poolb=RichMeasurementDataset(config['dataset']['root'],split='val',geometry_pool='ood',evaluation_only=True,
        a_range=(4,4),b_range=(4,4),hold_views=4,base_seed=20261011,normalization=config['normalization']['method'])
    measurements['Pool-B']=run_epoch(model,make_loader(poolb,config),config)
    measurement_rows=[dict(pool=pool,domain=domain,**values) for pool,result in measurements.items() for domain,values in result['measurement'].items()]
    write_csv(ROOT/'measurement_metrics.csv',measurement_rows)
    write_json(ROOT/'measurement_only_evaluation.json',dict(completed=True,checkpoint_sha256=freeze['sha256'],results=measurements,
        completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),volumetric_gt_read=False))
    batch=move_rich_batch(rich_measurement_collate([val[0]]),config['training']['device'])
    properties={key:set_properties(model,batch[key]) for key in ('A','B','T')}
    gradients=teacher_gradient_diagnostic(model,batch)
    reference=json.loads((REFERENCE/'summary.json').read_text());v1=json.loads((V1/'summary.json').read_text())
    boundary_path=ROOT/'posthoc_gt_boundary.json'
    if not boundary_path.exists():
        write_json(boundary_path,dict(first_posthoc_gt_evaluation_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            checkpoint_hash_frozen_before_gt=True,sha256=freeze['sha256'],best_epoch_frozen=freeze['best_epoch'],
            completed_epochs_before_gt=300,checkpoint_selection_metric=SELECTION,measurement_evaluation_completed_before_gt=True,
            method_changes_after_gt_forbidden=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    # First permitted volume-loading dataset follows the explicit boundary above.
    results={'Supervised-MRGR':copy.deepcopy(reference['results']['C1']),
             'SelfSup-v1':copy.deepcopy(v1['results']['SelfSup-C1']),'SelfSup-v2':{}}
    voxel_config=copy.deepcopy(config);voxel_config['loss']=load_config('configs/train_supervised_oari_mrgr_300.yaml')['loss']
    output=ROOT/config['experiment']['name'];table=[]
    for condition,views,selection,pool,draws in CONDITIONS:
        collected=[];allrows=[]
        for draw in range(draws):
            suffix=Path(condition)/f'draw{draw}' if draws==5 else Path(condition)
            directory=output/suffix
            if (directory/'C1/metrics.json').exists():
                summary=json.loads((directory/'C1/metrics.json').read_text());rows=read_jsonl(directory/'C1/per_case_metrics.jsonl')
            else:
                sums,stage_rows=evaluate_stages(model,evaluation_dataset(config,pool=pool,views=views,selection=selection,draw=draw),
                    voxel_config,directory,condition=condition,draw=draw)
                summary=sums['C1'];rows=stage_rows['C1']
            pairs=[(row['case_id'],row['selected_station_ids']) for row in rows]
            for path in (REFERENCE/'supervised_oari_mrgr_300'/suffix/'C1/per_case_metrics.jsonl',
                         V1/'selfsup_oari_mrgr_v1'/suffix/'C1/per_case_metrics.jsonl'):
                assert pairs==[(row['case_id'],row['selected_station_ids']) for row in read_jsonl(path)]
            collected.append(summary);allrows.extend(rows)
            print(f"v2 post-hoc {condition} draw{draw}: voxel L1={summary['mean_per_case']['relative_l1']:.8f}",flush=True)
        results['SelfSup-v2'][condition]=collected[0] if draws==1 else aggregate_draws(collected,allrows)
        for label in results:
            table.append(dict(condition=condition,model=label,draws=draws,**{metric:results[label][condition]['mean_per_case'][metric] for metric in METRICS}))
    teacher,teacher_rows=teacher_posthoc(model,val,config)
    write_csv(ROOT/'teacher_diagnostics.csv',teacher_rows)
    comparisons=[];gaps=[]
    for condition in results['SelfSup-v2']:
        old=results['SelfSup-v1'][condition]['mean_per_case'];new=results['SelfSup-v2'][condition]['mean_per_case'];sup=results['Supervised-MRGR'][condition]['mean_per_case']
        comparisons.append(dict(condition=condition,v1_l1=old['relative_l1'],v1_l2=old['relative_l2'],v2_l1=new['relative_l1'],v2_l2=new['relative_l2'],
            absolute_l1_improvement=old['relative_l1']-new['relative_l1'],relative_l1_improvement=1-new['relative_l1']/old['relative_l1']))
        gaps.append(dict(condition=condition,supervised_l1=sup['relative_l1'],v2_l1=new['relative_l1'],
                         absolute_l1_gap=new['relative_l1']-sup['relative_l1'],relative_l1_ratio=new['relative_l1']/sup['relative_l1']))
    by={row['condition']:row for row in comparisons}
    criteria=dict(val_a_improvement_ge_003=by['Val-A-V4']['absolute_l1_improvement']>=.003,
        val_b_worsening_le_3pct=by['Val-B-V4']['relative_l1_improvement']>=-.03,
        v3_improves=by['V3']['absolute_l1_improvement']>0,v6_improves=by['V6']['absolute_l1_improvement']>0,
        clustered_l1_le_042=by['clustered-V4']['v2_l1']<=.42,
        strict_gt_free_training=audit_is_clean(training['gt_access']),gate_passed=json.loads((ROOT/'gate_summary.json').read_text())['passed'],
        set_and_teacher_gradient_checks=all(p['passed'] for p in properties.values()) and gradients['passed'])
    summary=dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model='GeoLift-OARI-MRGR-Res3D',parameter_count=332070,architecture_changed=False,
        training=training,checkpoint=freeze,posthoc_boundary=json.loads(boundary_path.read_text()),results=results,
        measurement_generalization=measurements,teacher_posthoc=teacher,comparisons=comparisons,supervised_gaps=gaps,
        set_properties=properties,teacher_gradients=gradients,volumetric_gt_used_for_training=False,
        volumetric_gt_used_for_selection=False,posthoc_gt_evaluation_only=True,test_used=False,criteria=criteria,
        decision='SelfSup-MRGR v2 GO' if all(criteria.values()) else 'SelfSup-MRGR v2 NO-GO',
        decision_note='Single-seed engineering references; measurement error alone does not establish 3D correctness',
        integrity=check_integrity(ROOT,config,REPO,posthoc=True))
    assert digest(freeze['checkpoint'])==freeze['sha256']
    write_json(ROOT/'summary.json',summary);write_csv(ROOT/'voxel_metrics_posthoc.csv',table)
    write_csv(ROOT/'v1_vs_v2.csv',comparisons);write_csv(ROOT/'supervised_gap.csv',gaps)
    from geophy_set3d.evaluation.selfsup_v2_report import render_report
    render_report(ROOT,REPO)


if __name__=='__main__':main()
