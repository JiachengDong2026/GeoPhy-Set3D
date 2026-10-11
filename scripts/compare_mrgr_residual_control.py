#!/usr/bin/env python3
"""The sole supervised residual attribution comparison, with exact frozen pairs."""
import copy
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.protocols import evaluation_dataset,evaluate,load_model
from geophy_set3d.evaluation.analysis import aggregate_draws,read_jsonl,write_csv
from geophy_set3d.evaluation.measurement_diagnostics import set_properties
from geophy_set3d.evaluation.research_integrity import check_integrity
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import load_config,seed_everything,write_json,move_batch
from geophy_set3d.training.comparison import assert_fair_mrgr_zero_control

ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/mrgr_residual_control_v1')
REFERENCE=ROOT.parent/'oari_vs_mrgr_300'
REPO=Path(__file__).resolve().parents[1]
CONDITIONS=(('Val-A-V4',4,'reference','train',1),('Val-B-V4',4,'reference','ood',1),('V3',3,'random','train',5),
            ('V4',4,'random','train',5),('V6',6,'random','train',5),('clustered-V4',4,'clustered','train',5),('spread-V4',4,'spread','train',1))
METRICS=('mae','rmse','relative_l1','relative_l2')


def main():
    config=load_config('configs/train_supervised_oari_mrgr_zero_residual_control_300.yaml')
    full_config=load_config('configs/train_supervised_oari_mrgr_300.yaml');assert_fair_mrgr_zero_control(full_config,config)
    seed_everything(20261011);torch.set_num_threads(2)
    base=Path(config['experiment']['output_root']);name=config['experiment']['name'];logs=base/'logs'/name
    training=json.loads((logs/'training_summary.json').read_text())
    assert training['completed_epochs']==300 and training['stopping_reason']=='max_epochs'
    assert json.loads((logs/'validation_station_ids.json').read_text())==json.loads((base/'logs/supervised_oari_mrgr_300/validation_station_ids.json').read_text())
    checkpoint=base/'checkpoints'/name/'best.pt';sha=digest(checkpoint);model,ckpt=load_model(checkpoint,config['training']['device'])
    assert ckpt['config']==config and sum(p.numel() for p in model.parameters())==332070
    reference=json.loads((REFERENCE/'summary.json').read_text());assert digest(reference['checkpoint'])==reference['checkpoint_sha256']
    results={'OARI':copy.deepcopy(reference['results']['OARI']),'MRGR-full':copy.deepcopy(reference['results']['C1']),'ZR-control':{}}
    csv_rows=[];per_draw=[];output=ROOT/name
    for condition,views,selection,pool,draws in CONDITIONS:
        summaries=[];rows=[]
        for draw in range(draws):
            suffix=Path(condition)/f'draw{draw}' if draws==5 else Path(condition)
            directory=output/suffix
            if (directory/'metrics.json').exists():s=json.loads((directory/'metrics.json').read_text());r=read_jsonl(directory/'per_case_metrics.jsonl')
            else:s,r=evaluate(model,evaluation_dataset(config,pool=pool,views=views,selection=selection,draw=draw),config,directory,condition=condition,draw_id=draw)
            refs=read_jsonl(REFERENCE/'supervised_oari_mrgr_300'/suffix/'C1/per_case_metrics.jsonl')
            assert [(v['case_id'],v['selected_station_ids']) for v in r]==[(v['case_id'],v['selected_station_ids']) for v in refs]
            summaries.append(s);rows.extend(r)
            per_draw.append(dict(condition=condition,model='ZR-control',draw_id=draw,**{k:s['mean_per_case'][k] for k in METRICS}))
            print(f'{condition} draw{draw}: control L1={s["mean_per_case"]["relative_l1"]:.8f}',flush=True)
        results['ZR-control'][condition]=summaries[0] if draws==1 else aggregate_draws(summaries,rows)
        for label in results:csv_rows.append(dict(condition=condition,model=label,draws=draws,**{k:results[label][condition]['mean_per_case'][k] for k in METRICS}))
    for metric in METRICS:
        expected=training['best_val_metrics']['mean_per_case'][metric];actual=results['ZR-control']['Val-A-V4']['mean_per_case'][metric]
        assert abs(expected-actual)<=1e-7+abs(expected)*1e-5
    ds=evaluation_dataset(config);sample=ds[0];sample={k:v for k,v in sample.items() if not k.startswith('gt_')}
    properties=set_properties(model,move_batch(projection_pool_collate([sample]),config['training']['device']))
    gains=[]
    for condition in results['ZR-control']:
        control=results['ZR-control'][condition]['mean_per_case']['relative_l1'];full=results['MRGR-full'][condition]['mean_per_case']['relative_l1']
        gains.append(dict(condition=condition,control_l1=control,full_l1=full,full_absolute_l1_gain=control-full,full_relative_gain_percent=100*(control-full)/control))
    # Descriptive operational reference fixed before comparison, not a significance test.
    supports=all(v['full_absolute_l1_gain']>=.003 for v in gains if v['condition'] in ('Val-A-V4','V3','clustered-V4'))
    conclusion='signed measurement residual provides information beyond the added refinement capacity' if supports else 'the current experiment cannot cleanly attribute the gain to signed residual information alone'
    protocol=dict(config=config,full_config=full_config,parameter_count=332070,same_seed_initialization_exact=True,
        same_sampler=True,same_training_budget=True,real_residual_channel=False,projector_during_control_training=False,
        reference_checkpoint=reference['checkpoint'],reference_sha256=reference['checkpoint_sha256'],
        attribution_reference_absolute_l1_gain=.003,reference_note='descriptive engineering reference, not statistical significance',
        test_used=False,pool_b_evaluation_only=True,supervised_architecture_experiments_closed=True)
    (ROOT/'effective_protocol.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
    summary=dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model='GeoLift-OARI-MRGR-ZR-Control',
        checkpoint=str(checkpoint),checkpoint_sha256=sha,best_epoch=ckpt['epoch']+1,parameter_count=332070,full_parameter_count=332070,
        results=results,training=training,protocol=protocol,set_properties=properties,gains=gains,
        signed_residual_incremental_value_supported=supports,conclusion=conclusion,
        supervised_architecture_experiments_closed=True,integrity=check_integrity(ROOT,config))
    assert digest(checkpoint)==sha
    write_json(ROOT/'summary.json',summary);write_csv(ROOT/'main_metrics.csv',csv_rows);write_csv(ROOT/'per_draw.csv',per_draw);write_csv(ROOT/'gains.csv',gains)
    from geophy_set3d.evaluation.selfsup_report import render_control_report
    render_control_report(ROOT,REPO)


if __name__=='__main__':main()
