#!/usr/bin/env python3
"""Read-only audit and plots of completed supervised diagnostic; no forwards/steps."""
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_senseiver_supervised_probe import ART,OUT,CK,config,verify
from geophy_set3d.training.senseiver_supervised_probe import (SupervisedProbeGuard,exit_event,verify_resume,
    volume_metrics,metrics_mean)
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import save_slices,write_json


def load(p):return json.loads(p.read_text())


def main():
    c=config();guard=SupervisedProbeGuard(c['dataset_root'],OUT,list(range(8)))
    rows=[];evalrows=[];interventions=[];stages={};audit={};figdir=ART/'figures';figdir.mkdir(exist_ok=True)
    with guard:
        reg=verify(c,list(range(8)))
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for stage in ['A','B']:
            d=OUT/stage
            if not (d/'complete.json').exists():
                assert stage=='B' and stages['A']['reason']!='PASSED'
                stages[stage]=dict(reason='NOT_RUN_A_DID_NOT_PASS',updates=0)
                write_json(ART/'B_not_run.json',stages[stage]);continue
            result=load(d/'complete.json');saved=torch.load(CK/stage/'last.pt',map_location='cpu',weights_only=False)
            init=torch.load(CK/stage/'epoch0.pt',map_location='cpu',weights_only=False)
            assert sha256(CK/stage/'last.pt')==result['last_sha256'] and sha256(CK/stage/'epoch0.pt')==result['epoch0_sha256']
            assert state_hash(init['model'])==reg['initial_hashes'][stage]==result['initial_hash']
            assert state_hash(saved['model'])==result['final_hash'] and result['initial_hash']!=result['final_hash']
            assert result['reason']==saved['exit_state']['reason']
            history=saved['history'];ev=saved['evaluations'];events=saved['events']
            assert result['updates']==len(history)==saved['updates']<=c['stages'][stage]['max_updates']
            assert all(int(x['step'])==len(history) for x in saved['optimizer']['state'].values())
            assert saved['registration_sha256']==sha256(OUT/'registration.json')
            for n,value in [('history',history),('evaluations',ev),('events',events)]:
                assert load(d/(n+'.json'))==value
                with gzip.open(ART/stage/(n+'.json.gz'),'rt') as f:assert json.load(f)==value
            assert all(r['updates']==r['step']==j+1 and r['queries_per_case']==64**3 and r['stations']==c['station_ids'] and r['lr']==2e-4 for j,r in enumerate(history))
            effective=c['stages'][stage]['effective_batch'];count=len(c['stages'][stage]['indices'])
            assert all(r['indices']==list(range(((j*effective)%count),((j*effective)%count)+effective)) for j,r in enumerate(history))
            assert all(all(r[k][g]>0 for g in ['encoder','latents','decoder']) for r in history for k in ['gradient_l2','update_l2','post_clip_gradient_l2'])
            assert all(e['complete_query_finite'] and e['query_count_per_case']==64**3 for e in ev)
            assert not result['access']['denied_attempts']
            permitted={reg['samples'][i]['case_id'] for i in c['stages'][stage]['indices']}
            assert all(any(name in item['path'] for name in permitted) for item in result['access']['reads'] if item['category'] in ['train_GT64','projection'])
            try:verify_resume(saved,stage,c,saved['registration_sha256'])
            except ValueError as exc:assert 'stopped' in str(exc)
            else:raise AssertionError('Scientific stop was resumable')
            recomputed={}
            for event in events:
                if 'engineering' in event:continue
                e=next(x for x in ev if x['updates']==event['updates'])
                elapsed=c['stages'][stage]['max_seconds'] if event['reason']=='BUDGET_EXHAUSTED_UNRESOLVED' and event['updates']<c['stages'][stage]['max_updates'] else 0.
                check=exit_event(stage,event['updates'],[x['loss_nmse'] for x in history[:event['updates']]],e,c['stages'][stage],recomputed,elapsed,result['references']['mean_template']['mean']['nmse'])
                assert check==event,(check,event);recomputed=check
            with np.load(d/'final_volumes.npz') as a:
                p=torch.from_numpy(a['prediction']);g=torch.from_numpy(a['gt'])
            assert p.shape==g.shape==(count,1,64,64,64)
            metrics=[volume_metrics(p[i],g[i]) for i in range(count)]
            for i,m in enumerate(metrics):
                for name in ['nmse','relative_l1','relative_l2','mean','peak','spatial_std','total_ratio']:
                    assert abs(m[name]-ev[-1]['cases'][i][name])<1e-9,(stage,i,name)
            for i in range(min(count,3)):
                save_slices(figdir/f'{stage}_case{i}_slices.png',g[i],p[i],f'{stage} fixed train index {i}')
            minimum={k:{g:min(r[k][g] for r in history) for g in ['encoder','latents','decoder']} for k in ['gradient_l2','update_l2']}
            audit[stage]=dict(registered_fresh_initial_verified=True,checkpoint_and_optimizer_identity_verified=True,
                fixed_case_order_and_stations_verified=True,all_full_query_and_gradients_updates_valid=True,
                exit_events_recomputed=True,scientific_resume_rejected=True,derived_volumes_match_metrics=True,
                no_denied_access=True,minima=minimum)
            stages[stage]=dict(reason=result['reason'],updates=result['updates'],elapsed_seconds=result['elapsed_seconds'],
                mean=ev[-1]['mean'],checkpoint=result['checkpoint'],checkpoint_sha256=result['last_sha256'],
                initial_hash=result['initial_hash'],training_head=result['training_head'],peak_gpu_bytes=result['peak_gpu_bytes'])
            for r in history:rows.append(dict(stage=stage,updates=r['updates'],loss_nmse=r['loss_nmse'],seconds=r['seconds'],elapsed_seconds=r['elapsed_seconds'],peak_gpu_bytes=r['peak_gpu_bytes'],**{f'{k}_{g}':v for k in ['gradient_l2','update_l2'] for g,v in r[k].items()}))
            for e in ev:
                for r in e['cases']:evalrows.append(dict(stage=stage,updates=e['updates'],index=r['index'],case_id=r['case_id'],**{k:v for k,v in r.items() if isinstance(v,(float,int)) and k not in ['index']},**{f'plume_{k}':v for k,v in r['plume'].items()}))
                for r in e.get('interventions',[]):interventions.append(dict(stage=stage,updates=e['updates'],**r))
            axes[0].plot([r['updates'] for r in history],[r['loss_nmse'] for r in history],alpha=.6,label=f'{stage} train')
            axes[0].plot([e['updates'] for e in ev],[e['mean']['nmse'] for e in ev],'o-',label=f'{stage} full eval')
            axes[1].plot([e['updates'] for e in ev],[e['mean']['relative_l1'] for e in ev],'o-',label=stage)
            for name in ['zero','uniform']+(['mean_template'] if stage=='B' else []):
                axes[0].axhline(result['references'][name]['mean']['nmse'],ls=':',label=f'{stage} {name}')
            shutil.copy2(OUT/f'{stage}.log',ART/f'{stage}.log')
        for ax,label in zip(axes,['Per-sample normalized MSE','3D relative L1']):
            ax.set_xlabel('Optimizer updates');ax.set_ylabel(label);ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(figdir/'learning_curves.png',dpi=150);plt.close(fig)
        if interventions:
            fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
            for i in range(3):
                rr=[r for r in interventions if r['index']==i];ax.plot([r['updates'] for r in rr],[r['relative_MA'] for r in rr],'.-',label=f'case{i}')
            ax.axhline(.001,color='k',ls=':');ax.set_yscale('log');ax.set_xlabel('Updates');ax.set_ylabel('normal-zero relative MA');ax.legend();fig.savefig(figdir/'measurement_dependence.png',dpi=150);plt.close(fig)
        for name,records in [('train_scalars',rows),('evaluation_scalars',evalrows),('measurement_interventions',interventions)]:
            if records:
                with (ART/(name+'.csv')).open('w') as f:
                    writer=csv.DictWriter(f,list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(records)
        if stages['A']['reason']!='PASSED':decision='A_NOT_PASSED_NO_CONDITIONAL_CLAIM'
        elif stages['B']['reason']=='PASSED':decision='SUPERVISED_SMALL_SAMPLE_LEARNABILITY_PASSED'
        else:decision='A_PASSED_B_NOT_PASSED'
        result=dict(decision=decision,stages=stages,stop_here=True,no_full_training=True,no_additional_repair=True,
                    not_a_selfsupervised_baseline=True,posthoc_model_forwards=0,posthoc_optimizer_steps=0)
        write_json(ART/'decision.json',result);write_json(ART/'final_integrity_audit.json',audit)
    write_json(ART/'final_access.json',guard.summary())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
