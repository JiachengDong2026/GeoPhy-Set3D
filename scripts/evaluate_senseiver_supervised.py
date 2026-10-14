#!/usr/bin/env python3
"""Freeze Val-selected supervised candidate, then GT-free 900 outputs and separate GT."""
import argparse,ast,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np,pandas as pd,torch,yaml
import train_ray_senseiver_supervised as train
from run_senseiver_supervised_probe import load_samples,evaluate
from geophy_set3d.training.senseiver_supervised_formal import FormalGT64Guard,datasets
from geophy_set3d.training.senseiver_supervised_probe import fresh
from geophy_set3d.training.ray_senseiver import sha256,tensor_identity
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import write_json,move_batch,seed_everything
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard,METRICS,GROUPS
from geophy_set3d.evaluation.ray_senseiver import summarize
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.models import build_model
from geophy_set3d.data.collate import projection_pool_collate
from diagnose_clustered_one_swap import dataset,target_gt
OUT=train.OUT;ART=train.ART;EV=OUT/'evaluation'
MRGR=dict(checkpoint='/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_oari_mrgr_300/best.pt',sha256='4c9a39fd39432988cf5971b9a657a4cf22180650d3490bf65f3a8a7aeb04905c',config='configs/train_supervised_oari_mrgr_300.yaml')
DEEP=REPO/'docs/reports/artifacts/deepsets_selfsup_v1'

def freeze(c):
    complete=json.loads((OUT/'science/complete.json').read_text());EV.mkdir(exist_ok=True)
    assert complete['reason'] not in ['FAILED_TO_LEARN','STALLED_WITHOUT_DEPENDENCE']
    reg=train.verify(c);path=Path(complete['best_checkpoint']);assert sha256(path)==complete['best_sha256']
    with FormalGT64Guard(c['dataset_root'],OUT) as guard:
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['config']==c and saved['registration_sha256']==sha256(OUT/'registration.json')
        model=fresh(reg['scale']).to(c['device']);model.load_state_dict(saved['model']);model.eval();_,val=datasets(c);replay=train.validate(model,val,c['device'])
        assert np.isclose(replay['mean']['nmse'],complete['best_nmse'],rtol=1e-5,atol=1e-8)
        hist=json.loads((OUT/'science/history.json').read_text());assert complete['best_nmse']==min(r['val']['mean']['nmse'] for r in hist)
        _,samples,targets=load_samples(c,[0,1,2]);health,_=evaluate(model,samples,targets,c,'B',saved['updates'],c['device'],True)
        step=saved['steps'][-1];validgroups=all(x>0 for key in ['gradient_l2','update_l2'] for x in step[key].values())
        effective=(replay['mean']['nmse']<.98 and saved['history'][-1]['train']['mean']['nmse']<.98 and sum(r['relative_MA']>=.001 for r in health['interventions'])>=2 and all(r['peak']>0 and r['spatial_std']>0 for r in replay['cases']) and validgroups)
        result=dict(effective_learning=effective,checkpoint=str(path),sha256=sha256(path),epoch=complete['best_epoch'],updates=saved['updates'],completed_epochs=complete['epochs'],total_updates=complete['updates'],reason=complete['reason'],model_hash=state_hash(model.state_dict()),selector='Val mean per-case voxel NMSE only',selector_replay=replay,health=health,valid_gradient_updates=validgroups,supervised_train_GT64=True,supervised_Val_GT64_selection=True,not_pure_selfsup=True,access=guard.summary(),registration_sha256=sha256(OUT/'registration.json'))
        write_json(EV/'frozen_best.json',result);write_json(ART/'frozen_best.json',result)
    if not effective:print('No effective model; no 900 evaluation');return
    records=json.loads((REPO/c['comparison']['manifest']).read_text());assert len(records)==900
    assert sha256(REPO/c['comparison']['manifest'])==reg['comparison_manifest_sha256']
    assert sha256(MRGR['checkpoint'])==MRGR['sha256']
    p=dict(config=c,models=dict(Senseiver=dict(checkpoint=str(path),sha256=result['sha256']),SupervisedMRGR=MRGR),manifest_sha256=reg['comparison_manifest_sha256'],sources=reg['source_sha256']|{f:sha256(REPO/f) for f in ['scripts/evaluate_senseiver_supervised.py','src/geophy_set3d/evaluation/selfsup_c0c1_diagnostic.py','src/geophy_set3d/training/metrics.py',MRGR['config']]},references={str(p.relative_to(REPO)):sha256(p) for p in [DEEP/'paired_900.csv',DEEP/'deepsets_900.csv']},no_900_selection=True,tie_epsilon=1e-8)
    write_json(EV/'protocol.json',p);write_json(EV/'station_subsets.json',records);write_json(ART/'evaluation_protocol.json',p)
    print(json.dumps(dict(effective=effective,best_epoch=result['epoch'],val=replay['mean'])))

def verify(c,weights=True):
    p=json.loads((EV/'protocol.json').read_text());f=json.loads((EV/'frozen_best.json').read_text());assert f['effective_learning']
    assert subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/ray_senseiver_supervised_v1/evaluation_protocol.json'],cwd=REPO)==(EV/'protocol.json').read_bytes()
    for path,h in p['sources'].items():assert sha256(REPO/path)==h
    for path,h in p['references'].items():assert sha256(REPO/path)==h
    assert sha256(EV/'station_subsets.json')==p['manifest_sha256']
    if weights:
        for v in p['models'].values():assert sha256(v['checkpoint'])==v['sha256']
    return p,f,json.loads((EV/'station_subsets.json').read_text())

def token(r):return f"case{r['case_index']}_{r['geometry_id']}"

def model_for(name,c,p,f,device):
    v=p['models'][name];ck=torch.load(v['checkpoint'],map_location='cpu',weights_only=False)
    if name=='Senseiver':
        reg=json.loads((OUT/'registration.json').read_text());model=fresh(reg['scale'])
    else:model=build_model(yaml.safe_load((REPO/v['config']).read_text()))
    model.load_state_dict(ck['model']);return model.to(device).eval()

def infer(c):
    p,f,records=verify(c);device=c['device'];ref=pd.read_csv(DEEP/'deepsets_900.csv').set_index(['case_id','group','draw']);(EV/'case_images').mkdir(exist_ok=True);(EV/'prediction_rows').mkdir(exist_ok=True)
    checkpoints=[v['checkpoint'] for v in p['models'].values()]
    for name in ['Senseiver','SupervisedMRGR']:
        with DiagnosticReadGuard(c['dataset_root'],checkpoints,EV,'inference') as guard:
            ds=dataset(dict(dataset=c['dataset_root'],original_seed=c['seed']))
            model=model_for(name,c,p,f,device);before=state_hash(model.state_dict());started=time.perf_counter()
            with torch.no_grad():
                for index in range(36):
                    source,projections=ds._projections(index)
                    for r in [x for x in records if x['case_index']==index]:
                        tok=name+'_'+token(r);rowfile=EV/'prediction_rows'/(tok+'.json')
                        if rowfile.exists():
                            saved=json.loads(rowfile.read_text());assert saved['record']==r and saved['checkpoint_sha256']==p['models'][name]['sha256'] and sha256(saved['prediction_file'])==saved['prediction_sha256'];continue
                        assert source['case_id']==r['case_id'];empty=np.empty(0,dtype=np.int64)
                        b=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['station_ids']),empty)]),device)
                        h=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['H_station_ids']),empty)]),device)
                        identity=dict(input_identity_sha256=tensor_identity(b['input_projection_raw'],b['input_station_indices'],b['input_station_origin_m'],b['input_path_length_m']),H_mask_sha256=tensor_identity(h['view_mask'],h['input_valid_ray']),complete_input_sha256=tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k],torch.Tensor)]),complete_H_sha256=tensor_identity(*[h[k] for k in sorted(h) if isinstance(h[k],torch.Tensor)]))
                        old=ref.loc[(r['case_id'],r['group'],r['draw'])]
                        for key in identity:assert old[key]==identity[key],key
                        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter();prediction=model(b);torch.cuda.synchronize(device);seconds=time.perf_counter()-start
                        assert prediction.shape==(1,1,64,64,64) and torch.isfinite(prediction).all()
                        metrics={}
                        for label,bb in [('observed',b),('heldout',h)]:
                            pp=project_input(prediction,bb);valid=bb['input_valid_ray'][0,:,0].bool()&bb['view_mask'][0,:,None,None]
                            mm=relative_projection_metrics(pp[0,:,0],bb['input_projection_raw'][0,:,0],valid);metrics.update({label+'_'+k:v for k,v in mm.items()})
                        dest=EV/'case_images'/(tok+'.npz');tmp=dest.with_suffix('.tmp')
                        with tmp.open('wb') as fd:np.savez(fd,prediction=prediction[0,0].cpu().numpy())
                        os.replace(tmp,dest)
                        write_json(rowfile,dict(record=r,model=name,checkpoint_sha256=p['models'][name]['sha256'],prediction_file=str(dest),prediction_sha256=sha256(dest),forward_seconds=seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),**identity,**metrics))
                    print(f'{name} saved {index+1}/36 cases; {time.perf_counter()-started:.1f}s',flush=True)
            assert state_hash(model.state_dict())==before and all(x.grad is None for x in model.parameters());assert guard.counts['val_gt64_opens']==0
            write_json(EV/(name+'_inference_access.json'),guard.summary());write_json(ART/(name+'_inference_access.json'),guard.summary());del model;torch.cuda.empty_cache()
    files=sorted((EV/'prediction_rows').glob('*.json'));assert len(files)==1800
    write_json(EV/'predictions_complete.json',dict(rows=1800,models=2,all_saved_before_posthoc_GT=True,protocol_sha256=sha256(EV/'protocol.json'),prediction_rows_sha256={x.name:sha256(x) for x in files}))
    write_json(ART/'predictions_complete.json',json.loads((EV/'predictions_complete.json').read_text()))

def gt(c):
    p,f,records=verify(c,False);seal=json.loads((EV/'predictions_complete.json').read_text());assert seal['rows']==1800 and seal['protocol_sha256']==sha256(EV/'protocol.json')
    frames={}
    with DiagnosticReadGuard(c['dataset_root'],[],EV,'gt') as guard:
        ds=dataset(dict(dataset=c['dataset_root'],original_seed=c['seed']))
        for name in ['Senseiver','SupervisedMRGR']:
            rows=[]
            for index in range(36):
                g=target_gt(ds,index)
                for r in [x for x in records if x['case_index']==index]:
                    path=EV/'prediction_rows'/(name+'_'+token(r)+'.json');assert sha256(path)==seal['prediction_rows_sha256'][path.name];row=json.loads(path.read_text());assert sha256(row['prediction_file'])==row['prediction_sha256']
                    with np.load(row['prediction_file'],allow_pickle=False) as a:prediction=torch.from_numpy(a['prediction'].copy())[None,None]
                    m=VolumeMetrics();m.update(prediction,g,[r['case_id']]);rows.append(dict(**r,**{k:v for k,v in row.items() if k not in ['record','prediction_file']},**{k:v for k,v in m.rows[0].items() if k!='case_id'}))
            frames[name]=pd.DataFrame(rows);frames[name].to_csv(ART/(name.lower()+'_900.csv'),index=False)
        write_json(ART/'evaluation_GT_access.json',guard.summary())
    historical=pd.read_csv(DEEP/'paired_900.csv').set_index(['case_id','group','draw']);deep=pd.read_csv(DEEP/'deepsets_900.csv').set_index(['case_id','group','draw']);ours=[]
    indices={k:fr.set_index(['case_id','group','draw']) for k,fr in frames.items()}
    for r in records:
        key=(r['case_id'],r['group'],r['draw']);h=historical.loc[key];d=deep.loc[key];s=indices['Senseiver'].loc[key];m=indices['SupervisedMRGR'].loc[key]
        for row in [d,s,m]:
            for name in ['station_ids','H_station_ids']:
                v=ast.literal_eval(row[name]) if isinstance(row[name],str) else row[name];assert v==r[name]
            assert row['case_index']==r['case_index'] and row['geometry_id']==r['geometry_id']
        for k in ['complete_input_sha256','complete_H_sha256']:assert d[k]==s[k]==m[k]
        for k in ['input_identity_sha256','H_mask_sha256']:assert d[k]==h[k]==s[k]==m[k]
        assert d['checkpoint_sha256']==h['deepsets_checkpoint_sha256'] and d['prediction_sha256']==h['deepsets_prediction_sha256']
        row=dict(**r,senseiver_checkpoint_sha256=s['checkpoint_sha256'],supervised_mrgr_checkpoint_sha256=m['checkpoint_sha256'],ours_checkpoint_sha256=h['ours_checkpoint_sha256'],deepsets_checkpoint_sha256=d['checkpoint_sha256'])
        for metric in METRICS:
            row.update({n+'_'+metric:float(v) for n,v in [('Senseiver',s[metric]),('SupervisedMRGR',m[metric]),('Ours',h['Ours_'+metric]),('DeepSets',d[metric])]})
        ours.append(row)
    paired=pd.DataFrame(ours);assert len(paired)==900;paired.to_csv(ART/'paired_900.csv',index=False)
    names=['Senseiver','SupervisedMRGR','Ours','DeepSets'];cols=[n+'_'+m for n in names for m in METRICS];cases=paired.groupby(['case_id','case_index','group'],as_index=False)[cols].mean();assert len(cases)==180;cases.to_csv(ART/'case_group_180.csv',index=False)
    ix=np.random.default_rng(c['comparison']['bootstrap_seed']).integers(0,36,(10000,36));statistics=[];summary=[]
    for group in GROUPS:
        cf=cases[cases.group==group].sort_values('case_id');assert len(cf)==36
        for metric in METRICS:
            for name in names:
                a=cf[name+'_'+metric].to_numpy();summary.append(dict(group=group,model=name,metric=metric,mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())))
            for name in names[1:]:statistics.append(dict(group=group,metric=metric,left='Senseiver',right=name,**summarize(cf['Senseiver_'+metric].to_numpy(),cf[name+'_'+metric].to_numpy(),ix,1e-8)))
    pd.DataFrame(statistics).to_csv(ART/'paired_statistics.csv',index=False);pd.DataFrame(summary).to_csv(ART/'group_summary.csv',index=False)
    write_json(ART/'evaluation_complete.json',dict(rows=900,models=4,cases=36,case_groups=180,paired_sha256=sha256(ART/'paired_900.csv'),best_sha256=f['sha256'],no_epoch_reselection=True,bootstrap_samples=10000,bootstrap_seed=c['comparison']['bootstrap_seed']))

def timing(c):
    p,f,records=verify(c);device=c['device'];selected=[r for r in records if r['case_index'] in [0,18,35] and r['group'] in ['G1','G3','G5'] and r['draw']==0];assert len(selected)==9
    refs=json.loads((DEEP/'protocol.json').read_text()) if (DEEP/'protocol.json').exists() else None
    # Historical timing for self-supervised references is retained; rerun the two supervised models with identical timing method.
    with DiagnosticReadGuard(c['dataset_root'],[x['checkpoint'] for x in p['models'].values()],EV,'inference') as guard:
        models={n:model_for(n,c,p,f,device) for n in p['models']};ds=dataset(dict(dataset=c['dataset_root'],original_seed=c['seed']));rows=[]
        resident={n:sum(x.numel()*x.element_size() for x in list(m.parameters())+list(m.buffers())) for n,m in models.items()};gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
        with torch.no_grad():
            for r in selected:
                source,pr=ds._projections(r['case_index']);b=move_batch(projection_pool_collate([ds._measurement_sample(source,pr,np.asarray(r['station_ids']),np.empty(0,dtype=np.int64))]),device)
                for m in models.values():
                    for _ in range(10):m(b)
                for repeat in range(30):
                    names=list(models);names=names[repeat%2:]+names[:repeat%2]
                    for name in names:
                        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);t=time.perf_counter();pred=models[name](b);torch.cuda.synchronize(device)
                        rows.append(dict(model=name,record=r,repeat=repeat,seconds=time.perf_counter()-t,peak_bytes_excluding_other_model=torch.cuda.max_memory_allocated(device)-sum(v for n,v in resident.items() if n!=name)));del pred
        write_json(ART/'timing.json',dict(device=device,shared_gpu=True,precision='FP32',batch=1,warmup=10,repeats=30,inputs=9,gpu_before=gpu_before,parameters={n:sum(x.numel() for x in m.parameters()) for n,m in models.items()},rows=rows,reference_selfsup_timing_source='docs/reports/artifacts/deepsets_selfsup_v1/timing.json; historical shared GPU, not directly controlled wall-clock comparison',access=guard.summary()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','infer','gt','timing']);a=ap.parse_args();torch.set_num_threads(2);c=train.settings();seed_everything(c['seed']);globals()[a.phase](c)
if __name__=='__main__':main()
