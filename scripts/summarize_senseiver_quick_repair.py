#!/usr/bin/env python3
"""Final read-only audit of completed A/C; no new model forwards or training."""
import csv
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_senseiver_single_repair import OUT,ART,CK,verify
from run_senseiver_quick_readout import settings,guard,verify as verify_readout
from geophy_set3d.training.senseiver_readout import ProbeGuard
from geophy_set3d.training.senseiver_single_repair import validate,exit_event,verify_resume
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.utils import write_json


def load(path):return json.loads(path.read_text())


def csv_file(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    c=validate(yaml.safe_load((REPO/'configs/ray_senseiver_single_repair_v1.yaml').read_text()));ds=make_dataset(c)
    a_config,source,layout=settings();a_guard=guard(a_config,source,ds)
    try:
        with a_guard:
            ra=verify_readout(a_config);assert sha256(a_config['readout']['source_checkpoint'])==a_config['readout']['checkpoint_sha256']
    finally:write_json(ART/'readout_final_access.json',a_guard.summary())
    a=load(OUT/'readout_complete.json');assert a['encoder_identity_before']==a['encoder_identity_after'] and a['encoder_grad_buffers_none']
    ledger=[json.loads(x) for x in (OUT/'readout_encoder_ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==32
    for i in range(16):assert [r['event'] for r in ledger if r['index']==i]==['started','returned']
    assert a['encoder_calls']==16 and a['encoder_updates']==0 and a['query_updates']==200
    assert len(a['query_history'])==201 and a['best_query_step']==min(a['query_history'],key=lambda r:r['fit_standardized_MSE'])['step']
    assert sha256(OUT/'latents.npz')==a['latents_sha256'] and sha256(OUT/'readout_stats.npz')==a['statistics_sha256']
    data={};audit=dict(stage_A_encoder_frozen=True,stage_A_calls=16,readout_fit_only_selection_verified=True,
        stage_A_registered_source_data_checkpoint_hashes_unchanged=True,stage_C={},posthoc_model_forwards=0,posthoc_optimizer_steps=0)
    initial_states={}
    for arm in ['control','treatment']:
        d=OUT/arm;ck=CK/arm;g=ProbeGuard(c['dataset']['root'],ck,ds.records,8)
        try:
            with g:
                reg,scale,inv=verify(c)
                assert [x['case_id'] for x in ra['fixed_inputs'][:8]]==reg['case_order']
                r={n:load(d/(n+'.json')) for n in ['history','health','events','complete','access','access_exit']}
                complete=r['complete'];history=r['history'];healths=r['health'];events=r['events']
                assert not (d/'engineering_error.json').exists()
                assert len(history)==complete['epochs'] and complete['updates']==4*len(history)<=320
                assert all(row['epoch']==i+1 and row['updates']==4*(i+1) and len(row['update_observations'])==4 for i,row in enumerate(history))
                digest=healths[0]['full_fixed_subset']['case_station_digest']
                assert all(row[k]['case_station_digest']==digest and row[k]['sample_count']==8 for row in history for k in ['train','eval'])
                state={}
                for h in healths[1:]:
                    state=exit_event(h['epoch'],history[:h['epoch']],h,scale['C_init'],state)
                    assert state==next(e for e in events if e['epoch']==h['epoch'])
                    assert h['updates']==4*h['epoch'] and h['chunk_check']['passed'] and h['chunk_check']['all_query_gradients_nonzero']
                    assert all(all(v>0 for v in row['gradient_l2'].values()) for row in h['diagnostic_cases'])
                assert state['reason']==complete['reason'] and state['reason'] is not None
                assert sha256(ck/'last.pt')==complete['last_sha256'] and sha256(ck/'epoch0.pt')==complete['epoch0_sha256']
                saved=torch.load(ck/'last.pt',map_location='cpu',weights_only=False)
                epoch0=torch.load(ck/'epoch0.pt',map_location='cpu',weights_only=False)
                assert state_hash(epoch0['model'])==complete['initial_hash']==reg['initialization'][arm]
                assert state_hash(saved['model'])==complete['model_state_hash']
                assert saved['history']==history and saved['health']==healths and saved['events']==events and saved['exit_state']==state
                assert saved['updates']==complete['updates'] and saved['scheduler']['last_epoch']==complete['epochs']
                assert all(int(v['step'])==complete['updates'] for v in saved['optimizer']['state'].values())
                try:verify_resume(saved,c,arm,complete['initial_hash'],sha256(OUT/'reconstruction_registration.json'),sha256(OUT/'reconstruction_source_manifest.json'))
                except ValueError as exc:assert 'stopped arm' in str(exc)
                else:raise AssertionError('Scientifically stopped checkpoint was resumable')
                groups=['encoder','latents','decoder']
                updates=[x for row in history for x in row['update_observations']]
                minimum_update={k:min(row['update_l2'][k] for row in updates) for k in groups}
                minimum_gradient={k:min(row['post_clip_gradient_l2'][k] for row in updates) for k in groups}
                assert all(v>0 for v in [*minimum_update.values(),*minimum_gradient.values()])
                initial_states[arm]=epoch0['model']
                item=dict(checkpoint_identity_verified=True,initial_state_verified=True,optimizer_scheduler_steps_verified=True,
                    all_4_updates_per_epoch=True,fixed_case_station_digests_verified=True,all_query_checks_passed=True,
                    exit_events_recomputed_exactly=True,scientific_resume_rejected=True,no_engineering_error=True,
                    minimum_update_l2=minimum_update,minimum_post_clip_gradient_l2=minimum_gradient)
                if arm=='treatment':
                    w=saved['model']['decoder.latent_condition.weight'];item['added_weight_final_l2']=float(w.double().norm());assert item['added_weight_final_l2']>0
                audit['stage_C'][arm]=item
                data[arm]=r
                for name in ['history','health','events']:
                    (ART/arm).mkdir(exist_ok=True)
                    with gzip.open(ART/arm/(name+'.json.gz'),'wb') as f:f.write((d/(name+'.json')).read_bytes())
                for name in ['complete','access','access_exit','environment']:shutil.copy2(d/(name+'.json'),ART/arm/(name+'.json'))
                del saved,epoch0
        finally:write_json(ART/f'{arm}_final_access.json',g.summary())
    assert all(torch.equal(v,initial_states['treatment'][k]) for k,v in initial_states['control'].items())
    assert set(initial_states['treatment'])-set(initial_states['control'])=={'decoder.latent_condition.weight'}
    assert not torch.count_nonzero(initial_states['treatment']['decoder.latent_condition.weight'])
    audit['shared_initial_tensors_elementwise_equal']=True
    first={arm:{k:v for k,v in row['health'][0]['full_fixed_subset'].items() if k!='seconds'} for arm,row in data.items()}
    assert first['control']==first['treatment']
    audit['epoch0_evaluation_identical_except_wall_seconds']=True
    paired_max=min(len(v['history']) for v in data.values());paired=[];scalar=[];health_rows=[]
    for arm,v in data.items():
        for r in v['history']:
            e=r['eval'];scalar.append(dict(arm=arm,epoch=r['epoch'],updates=r['updates'],H_L1=e['measurement']['heldout']['relative_l1'],H_L2=e['measurement']['heldout']['relative_l2'],
                cross_L1=e['measurement']['cross']['relative_l1'],observed_L1=e['measurement']['observed']['relative_l1'],measurement_smooth=e['loss_diagnostics']['measurement_smooth'],
                mean=e['output']['mean'],std=e['output']['std'],minimum=e['output']['min'],maximum=e['output']['max'],seconds=r['seconds'],peak_bytes=r['peak_gpu_allocated_bytes']))
        for h in v['health']:
            for r in h['diagnostic_cases']:
                health_rows.append(dict(arm=arm,epoch=h['epoch'],index=r['index'],**r['measurement_dependence'],**{f'gradient_{k}':x for k,x in r['gradient_l2'].items()}))
    for epoch in range(1,paired_max+1):
        values={arm:v['history'][epoch-1]['eval'] for arm,v in data.items()}
        paired.append(dict(epoch=epoch,updates=epoch*4,**{f'{arm}_{m}':v['measurement']['heldout'][f'relative_{m.lower()}'] for arm,v in values.items() for m in ['L1','L2']}))
    ht=data['treatment']['health'];last=ht[-1]
    conditions=dict(last3_H_below_090=len(ht)>=3 and all(h['full_fixed_subset']['measurement']['heldout']['relative_l1']<.9 for h in ht[-3:]),
        no_engineering_or_collapse=not any(e['reason']=='NEAR_ZERO_COLLAPSE' for e in data['treatment']['events']) and audit['stage_C']['treatment']['no_engineering_error'],
        distinguishable_case=max(r['relative_MA'] for r in last['case_differences'])>=.001,
        two_measurement_dependent_cases=sum(r['measurement_dependence']['relative_MA']>=.001 for r in last['diagnostic_cases'])>=2,
        all_groups_gradient_update_valid=all(x>0 for x in audit['stage_C']['treatment']['minimum_update_l2'].values()))
    decision=dict(status='GO_CANDIDATE' if all(conditions.values()) else 'REPAIR_FAILED_NO_GO',conditions=conditions,
        paired_epochs=paired_max,paired_updates=paired_max*4,actual_budget={arm:{k:v['complete'][k] for k in ['epochs','updates','reason','seconds','peak_gpu_allocated_bytes']} for arm,v in data.items()},
        readout_readable=a['readable'],stop_here=True,no_full_training=True,no_second_repair=True)
    write_json(ART/'final_integrity_audit.json',audit);write_json(ART/'decision.json',decision)
    csv_file(ART/'epoch_scalars.csv',scalar);csv_file(ART/'paired_epochs.csv',paired);csv_file(ART/'health_measurement_dependence.csv',health_rows);csv_file(ART/'readout_predictions.csv',a['predictions'])
    for source_name,target_name in [('quick_repair_tests_A.log','readout_tests.txt'),('quick_repair_tests_C.log','repair_tests.txt'),('quick_repair_regression.log','model_regression.txt'),('quick_repair_objective_regression.log','legacy_consistency_unseeded_failure.txt'),('quick_repair_objective_seeded.log','legacy_consistency_seeded_failure.txt')]:
        shutil.copy2(Path('/tmp')/source_name,ART/target_name)
    for name in ['control.log','treatment.log','readout.log','smoke.log','smoke_access.json']:
        shutil.copy2(OUT/name,ART/name)
    figdir=ART/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,head in zip(axes,['ridge','query']):
        for name,color in [('fit','tab:blue'),('check','tab:orange')]:
            p=[v for v in a['predictions'] if v['split']==name]
            ax.scatter([r['target'] for r in p],[r[head] for r in p],label=name,color=color)
        targets=[r['target'] for r in a['predictions']];ax.plot([min(targets),max(targets)],[min(targets),max(targets)],'k--',lw=1)
        ax.axhline(a['training_statistics']['target_mean'],ls=':',color='grey',label='fit-mean constant')
        ax.set_title(head);ax.set_xlabel('Target normalized measurement mean');ax.set_ylabel('Readout prediction');ax.legend()
    fig.savefig(figdir/'readout_scatter.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for arm,v in data.items():
        x=[0]+[r['epoch'] for r in v['history']];ev=[v['health'][0]['full_fixed_subset']]+[r['eval'] for r in v['history']]
        for ax,key in zip(axes[0],['relative_l1','relative_l2']):ax.plot(x,[r['measurement']['heldout'][key] for r in ev],label=arm);ax.set_ylabel('H '+key)
        axes[1,0].plot(x,[r['output']['mean'] for r in ev],label=arm)
        for i in range(3):axes[1,1].plot([h['epoch'] for h in v['health']],[h['diagnostic_cases'][i]['measurement_dependence']['relative_MA'] for h in v['health']],'.-' if arm=='control' else '.--',label=f'{arm} case{i}')
    axes[1,0].set_yscale('log');axes[1,0].set_ylabel('Output mean')
    axes[1,1].set_yscale('log');axes[1,1].axhline(.001,ls=':',color='k');axes[1,1].set_ylabel('normal-zero relative MA')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(figdir/'reconstruction_curves.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(decision=decision,audit=audit),indent=2))


if __name__=='__main__':main()
