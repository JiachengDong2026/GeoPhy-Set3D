"""Uniform predictions, all-method sealing, separate GT scoring and case statistics."""
from common import *
import argparse
import time
import copy
import numpy as np
import torch
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from geophy_set3d.evaluation.research import volume_metrics
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.reconstruction.classical import geometry_tensor,project
from report_historical_reconstruction_v1 import metrics as shape_metrics
from module_models_v2 import model_for
from classical import METHODS,task_directory

SCOPE=None
def score_dir(split):return OUT/'evaluation'/(SCOPE if split=='val' and SCOPE else split)


def training_methods():
    models=[]
    for m in verified(OUT/'model_manifest.json')['models']:
        models.append(dict(key=m['key'],kind='neural',run=m['run'],checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],best_epoch=m['best_epoch'],arm='multiscale',config=m['config'],parameters=m['parameters'],seed=m['seed']))
    for arm in ('single_scale','no_feedback','no_consistency','multiscale'):
        for seed in (20261011,20261019):
            if arm=='multiscale' and seed==20261011:continue
            target=OUT/'training'/f'{arm}_seed{seed}'
            summary=read(target/'summary.json');p=verified(target/'protocol.json')
            assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900
            assert not p['subset_smoke'] and p['seed']==seed
            assert all(summary['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
            log=rows(target/'metrics.jsonl');assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
            chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
            assert chosen['epoch']==summary['best_epoch'] and chosen['val']['measurement']['heldout']['relative_l1']==summary['best_val_H']
            ck=target/'checkpoints/best.pt';assert sha(ck)==summary['best_checkpoint_sha256']
            models.append(dict(key=f'{arm}_seed{seed}',kind='neural',run=str(target),checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=summary['best_epoch'],arm=arm,config=p['config'],parameters=p['parameters'],seed=seed))
    if (OUT/'module_ablations_v2/registration.json').exists():
        from module_registry_v2 import methods as module_methods
        models.extend(module_methods())
    return models


def target_for(m,split):
    return task_directory(split,m['key'],None) if m['kind']!='neural' else OUT/'evaluation'/split/m['key']


def predict_neural(model_info,split):
    if model_info['kind']=='sart':
        from sart import work,method
        assert model_info==method()
        return work(split)
    backend();m=model_info;reg=verified(OUT/'registration.json');settings=verified(OUT/'layouts.json')[split];cases=reg['cases'][split]
    if split=='test':
        lock=verified(OUT/'final_lock.json');assert m in lock['methods'];verify_hashes(lock['sources']);verify_hashes(lock['data_identity']);verify_hashes(lock['evidence'])
    target=target_for(m,split);target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():assert sha(target/'manifest.jsonl')==read(target/'complete.json')['manifest_sha256'];return
    records=rows(target/'manifest.jsonl') if (target/'manifest.jsonl').exists() else []
    known={(x['condition'],x['draw'],x['case_id']) for x in records};assert len(known)==len(records)
    for x in records:assert sha(target/x['prediction'])==x['sha256']
    started=time.monotonic()
    with AccessGuard(splits=[split],checkpoints=[m['checkpoint']]) as guard:
        assert sha(m['checkpoint'])==m['checkpoint_sha256'];ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==m['best_epoch'] and not ck['volumetric_gt_used']
        assert ck['config']==m['config'] and ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json')
        model=model_for(m['config'],m['arm'],'cuda:0');model.load_state_dict(ck['model'],strict=True);model.eval()
        assert sum(p.numel() for p in model.parameters())==m['parameters'];del ck
        with torch.no_grad():
            for s in settings:
                ds=dataset(split,s);assert [x['case_id'] for x in ds.records]==cases
                for i,case in enumerate(cases):
                    if (s['condition'],s['draw'],case) in known:continue
                    sample=ds[i];batch=move_batch(projection_pool_collate([sample]),'cuda:0');inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
                    assert sample['input_station_indices'].tolist()==s['input_ids'][case]
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();st=model(inputs,return_stages=True);torch.cuda.synchronize();seconds=time.perf_counter()-t
                    fields={k:st[k][0,0].cpu().numpy() for k in ('C0','C1')}
                    assert all(v.shape==(64,)*3 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in fields.values())
                    path=target/'predictions'/f'{s["condition"]}_d{s["draw"]}_{i:03d}.npz';path.parent.mkdir(exist_ok=True)
                    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**fields);temp.replace(path)
                    records.append(dict(condition=s['condition'],draw=s['draw'],case_id=case,input_ids=s['input_ids'][case],heldout_ids=s['heldout_ids'][case],
                        prediction=str(path.relative_to(target)),sha256=sha(path),reconstruct_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
                    jsonl(target/'manifest.jsonl',records)
                    del st,fields,batch,inputs
                print(f'{split} {m["key"]} {s["condition"]} draw{s["draw"]}: {len(records)}/{len(cases)*17}',flush=True)
        audit=guard.summary();assert not audit['denied']
    assert len(records)==len(cases)*17
    write(target/'complete.json',dict(completed_at=now(),reconstructions=len(records),fields=2*len(records),manifest_sha256=sha(target/'manifest.jsonl'),checkpoint_sha256=m['checkpoint_sha256'],audit=audit,seconds=time.monotonic()-started))


def expected_keys(reg,split):
    settings=verified(OUT/'layouts.json')[split]
    return {(s['condition'],s['draw'],case) for s in settings for case in reg['cases'][split]}


def validate_rows(cases,settings,rr):
    expected={(s['condition'],s['draw'],c) for s in settings for c in cases}
    assert len(rr)==len(expected)
    assert {(r['condition'],r['draw'],r['case_id']) for r in rr}==expected
    table={(s['condition'],s['draw']):s for s in settings}
    for r in rr:
        s=table[r['condition'],r['draw']];c=r['case_id']
        assert r['input_ids']==s['input_ids'][c] and r['heldout_ids']==s['heldout_ids'][c]
        assert not set(r['input_ids']) & set(r['heldout_ids'])


def verify_predictions(m,split,*,arrays=False):
    target=target_for(m,split);b=read(target/'complete.json');assert sha(target/'manifest.jsonl')==b['manifest_sha256']
    assert not b['audit']['denied'] and not any(k.startswith('GT64') for k in b['audit']['counts'])
    if m['kind']=='neural':assert b['checkpoint_sha256']==m['checkpoint_sha256'] and sha(m['checkpoint'])==m['checkpoint_sha256']
    else:
        expected_config=m['config'] if 'config' in m else verified(OUT/'classical/selected.json')['selected'][m['key']]['config']
        assert b['config']==expected_config and b['method']==m['key'] and b['split']==split
    rr=rows(target/'manifest.jsonl');reg=verified(OUT/'registration.json');expected=expected_keys(reg,split)
    assert len(rr)==len(expected)==b['reconstructions']
    assert b['fields']==len(rr)*(2 if m['kind']=='neural' else 1)
    assert {(r['condition'],r['draw'],r['case_id']) for r in rr}==expected
    validate_rows(reg['cases'][split],verified(OUT/'layouts.json')[split],rr)
    settings={(s['condition'],s['draw']):s for s in verified(OUT/'layouts.json')[split]}
    for x in rr:
        s=settings[x['condition'],x['draw']];case=x['case_id']
        assert x['input_ids']==s['input_ids'][case] and x['heldout_ids']==s['heldout_ids'][case] and not set(x['input_ids'])&set(x['heldout_ids'])
        path=target/x['prediction'];assert sha(path)==x['sha256']
        if m['kind']!='neural':assert x['config']==expected_config and 0<=x['iterations']<=expected_config['iterations']
        if arrays:
            with np.load(path,allow_pickle=False) as z:
                assert set(z.files)==({'C0','C1'} if m['kind']=='neural' else {'C1'})
                for k in z.files:
                    v=z[k];assert v.shape==(64,)*3 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1
    return target,rr


def seal(split,methods):
    # A running controller may have cached the original method list. Apply the
    # explicit pre-test SART amendment here without interrupting its training.
    if split=='val' and SCOPE is None and (OUT/'sart/registration.json').exists():
        from sart import wait_development
        added=wait_development()
        if not any(m['key']=='SART' for m in methods):methods=[*methods,added]
    if split=='test':assert methods==verified(OUT/'final_lock.json')['methods']
    manifests={};counts={}
    with AccessGuard() as guard:
        for m in methods:
            target,rr=verify_predictions(m,split,arrays=True);manifests[str(target/'manifest.jsonl')]=sha(target/'manifest.jsonl');counts[m['key']]=len(rr)
            if split=='test':assert not any(x.get('solver_status') in ('line_search_failed','nonfinite') for x in rr),'Unresolved numerical failure blocks final GT scoring'
    if split=='test':assert len(methods)==len(verified(OUT/'final_lock.json')['methods'])
    freeze(score_dir(split)/'all_predictions_sealed.json',dict(sealed_at=now(),methods=methods,manifests=manifests,counts=counts,
        cases=len(verified(OUT/'registration.json')['cases'][split]),layouts=17,all_predictions_saved_before_GT=True,audit=guard.summary(),
        final_lock_sha256=sha(OUT/'final_lock.json') if split=='test' else None))


def safe_shape(p,g):
    if p.sum()>0:return shape_metrics(p,g)
    # No removal of zero-output failures: L1 remains 1, other defined quantities retained.
    r=shape_metrics(np.maximum(p,np.finfo(np.float64).tiny),g)
    for k in ('centroid_distance_mm','centroid_delta_xyz_mm','spread_ratio_xyz','background_predicted_fraction'):r[k]=None
    r.update(relative_l1=1.,peak_ratio=0.,gt_peak_location_ratio=0.,mass_ratio=0.)
    return r


def score(split,device='cuda:0'):
    backend();boundary=verified(score_dir(split)/'all_predictions_sealed.json');reg=verified(OUT/'registration.json')
    if split=='test':
        lock=verified(OUT/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['data_identity']);verify_hashes(lock['evidence']);assert sha(OUT/'final_lock.json')==boundary['final_lock_sha256']
        assert boundary['methods']==lock['methods']
        for info in lock['gt64_stat_only'].values():
            stat=Path(info['path']).stat();assert stat.st_size==info['size'] and stat.st_mtime_ns==info['mtime_ns']
    verify_hashes(boundary['manifests']);work=[]
    for m in boundary['methods']:
        target,rr=verify_predictions(m,split);work.extend((m,target,r) for r in rr)
    out=score_dir(split)
    assert not (out/'metrics.jsonl').exists(),'Preserve previous scoring; investigate rather than overwrite'
    gt={};gt_hashes={};source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};scored=[];first_gt=None
    with AccessGuard(splits=[split],gt_splits=[split]) as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        for m,target,x in work:
            case=x['case_id']
            if case not in gt:
                if first_gt is None:first_gt=now()
                gt[case]=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
                gt_hashes[str(DATA/source[case]['gt_volume_64_path'])]=sha(DATA/source[case]['gt_volume_64_path'])
            p=np.load(DATA/source[case]['train_projection_path'],mmap_mode='r',allow_pickle=False)
            with np.load(target/x['prediction'],allow_pickle=False) as fields:
                for stage in fields.files:
                    c=fields[stage].astype(np.float64);shape=safe_shape(c,gt[case]);shape['absolute_peak_ratio_error']=abs(shape['peak_ratio']-1)
                    measurement={}
                    with torch.no_grad():
                        for label,ids in [('observed',x['input_ids']),('heldout',x['heldout_ids'])]:
                            g=geometry_tensor({k:v[ids] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')},device)
                            pred=project(torch.as_tensor(c,dtype=torch.float32,device=device),g);y=torch.as_tensor(np.array(p[ids]),device=device)
                            measurement[label]=relative_projection_metrics(pred,y,g['valid_ray'])
                    scored.append(dict(method=m['key'],stage=stage,case_id=case,condition=x['condition'],draw=x['draw'],volume=volume_metrics(c,gt[case]),morphology=shape,measurement=measurement,
                        reconstruct_seconds=x['reconstruct_seconds'],peak_allocated_bytes=x['peak_allocated_bytes'],device=x.get('device','cuda'),process_peak_rss_bytes=x.get('process_peak_rss_bytes'),solver_status=x.get('solver_status'),iterations=x.get('iterations')))
            if len(scored)%144==0:print(f'{split}: scored {len(scored)} fields',flush=True)
    jsonl(out/'metrics.jsonl',scored)
    write(out/'gt_access.json',dict(first_GT64_at=first_gt,seal_at=boundary['sealed_at'],seal_sha256=sha(out/'all_predictions_sealed.json'),unique_GT64_files=len(gt),GT64_sha256_after_seal=gt_hashes,gt128_read=False,audit=guard.summary(),scored_at=now()))
    print(f'{split}: scoring completed, {len(scored)} fields',flush=True)


def scalar_metrics(x):
    result={}
    for group in ('volume','morphology'):
        for k,v in x[group].items():
            if v is None or isinstance(v,(float,int)):result[group+'.'+k]=v
    for label,values in x['measurement'].items():
        for k,v in values.items():result['measurement.'+label+'.'+k]=v
    result['reconstruct_seconds']=x['reconstruct_seconds'];result['peak_allocated_bytes']=x['peak_allocated_bytes']
    result['process_peak_rss_bytes']=x.get('process_peak_rss_bytes')
    return result


def summarize(split,extra=None):
    reg=verified(OUT/'registration.json');rr=rows(score_dir(split)/'metrics.jsonl');cases=reg['cases'][split]
    if extra:rr+=extra
    lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    assert len(lookup)==len(rr)
    methods=sorted({(r['method'],r['stage']) for r in rr});panels=[];case_rows=[]
    for condition in ('Val-A-V4','spread-V4','V3','V6','clustered-V4'):
        selections=[[0]] if condition in ('Val-A-V4','spread-V4') else [[d] for d in range(5)]+[list(range(5))]
        for draws in selections:
            for method,stage in methods:
                values=[]
                for case in cases:
                    metrics=[scalar_metrics(lookup[method,stage,condition,d,case]) for d in draws];v={}
                    for k in metrics[0]:
                        vals=[a[k] for a in metrics];v[k]=None if any(x is None for x in vals) else float(np.mean(vals))
                    values.append(v);case_rows.append(dict(method=method,stage=stage,condition=condition,draws=draws,case_id=case,metrics=v))
                stats={}
                for k in values[0]:
                    a=np.array([v[k] for v in values if v[k] is not None],dtype=float)
                    stats[k]=dict(valid_cases=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,max=float(a.max()) if len(a) else None)
                panels.append(dict(method=method,stage=stage,condition=condition,draws=draws,metrics=stats))
    paired=[];rng=np.random.default_rng(reg['statistics']['seed']);indices=rng.integers(0,len(cases),(reg['statistics']['resamples'],len(cases)))
    comparisons=[m for m in methods if m[1]=='C1' and m[0]!='MRGR'];count=len(comparisons)
    for condition,draws in [('Val-A-V4',[0]),('spread-V4',[0]),('V3',list(range(5))),('V6',list(range(5))),('clustered-V4',list(range(5)))]:
        def vector(method,stage,key):return np.array([np.mean([lookup[method,stage,condition,d,c]['volume'][key] for d in draws]) for c in cases])
        for method,stage in comparisons:
            for metric in ('relative_l1','relative_l2'):
                a=vector('MRGR','C1',metric);b=vector(method,stage,metric);delta=a-b;bs=delta[indices].mean(1)
                paired.append(dict(condition=condition,draws=draws,reference=method,metric=metric,primary=condition=='Val-A-V4' and metric=='relative_l1',mean_MRGR=float(a.mean()),mean_reference=float(b.mean()),delta=float(delta.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),
                    simultaneous_ci=np.quantile(bs,[.025/count,1-.025/count]).tolist(),comparisons=count,case_differences=dict(zip(cases,delta.tolist())),wins=int((delta<-1e-8).sum()),ties=int((np.abs(delta)<=1e-8).sum()),losses=int((delta>1e-8).sum())))
    rates=[]
    for method in sorted({r['method'] for r in rr}):
        final=[r for r in rr if r['method']==method and r['stage']=='C1'];failed=sum(r.get('solver_status') in ('line_search_failed','nonfinite') for r in final)
        rates.append(dict(method=method,reconstructions=len(final),expected=len(cases)*17,missing=len(cases)*17-len(final),numerical_failures=failed,failure_rate=failed/len(final),budget_exhausted=sum(r.get('solver_status')=='budget_exhausted' for r in final),objective_stable=sum(r.get('solver_status')=='objective_stable' for r in final)))
    jsonl(score_dir(split)/'case_panels.jsonl',case_rows);write(score_dir(split)/'summary.json',dict(at=now(),split=split,cases=len(cases),panels=panels,paired=paired,statistics=reg['statistics'],failure_rates=rates,failures=[dict(method=r['method'],case_id=r['case_id'],condition=r['condition'],draw=r['draw'],status=r['solver_status']) for r in rr if r.get('solver_status') in ('line_search_failed','nonfinite')]))
    return rr


def historical_val_metrics():
    out=[]
    for key,base,metrics,manifest in [('MRGR',OLD_EVAL,OLD_EVAL/'per_case_metrics.jsonl',OLD_EVAL/'prediction_manifest.jsonl'),
        ('LR4E4',ROOT/'outputs/research/mrgr_autonomous_optimization_v2/frozen_formal_evaluation'/LR.name,
         ROOT/'outputs/research/mrgr_autonomous_optimization_v2/frozen_formal_evaluation'/LR.name/'per_case_metrics.jsonl',
         ROOT/'outputs/research/mrgr_autonomous_optimization_v2/frozen_formal_evaluation'/LR.name/'prediction_manifest.jsonl')]:
        mm={(x['condition'],x['draw'],x['case_id']):x for x in rows(manifest)}
        for x in rows(metrics):
            y=copy.deepcopy(x);meta=mm[x['condition'],x['draw'],x['case_id']];y['method']=key
            y['measurement']=meta['measurement'];y['reconstruct_seconds']=meta.get('reconstruct_s',meta.get('reconstruct_seconds'));y['peak_allocated_bytes']=None
            y['morphology']['absolute_peak_ratio_error']=abs(y['morphology']['peak_ratio']-1)
            # Historical measurement fields describe C1, so do not falsely label them as C0.
            if y['stage']=='C0':y['measurement']={label:{k:None for k in values} for label,values in meta['measurement'].items()}
            out.append(y)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','seal','score','summarize']);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--key');p.add_argument('--methods');p.add_argument('--scope');a=p.parse_args();SCOPE=a.scope
    methods=read(a.methods) if a.methods else (verified(OUT/'final_lock.json')['methods'] if a.split=='test' else training_methods())
    if a.action=='predict':predict_neural(next(m for m in methods if m['key']==a.key),a.split)
    elif a.action=='seal':seal(a.split,methods)
    elif a.action=='score':score(a.split)
    else:summarize(a.split,historical_val_metrics() if a.split=='val' else None)
