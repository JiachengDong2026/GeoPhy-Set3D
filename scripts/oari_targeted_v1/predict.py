"""Measurement-only prediction, frozen token hook reuse, and matched GPU timing."""
from stage import *
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
import time


def methods():
    out=[]
    for arm,seed in TASKS:
        target=BASE/'training'/f'{arm}_seed{seed}';p=verified(target/'protocol.json');s=read(target/'summary.json')
        assert s['completed'] and s['epochs']==300 and s['updates']==24900
        assert all(s['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        assert not read(target/'boundary_audit.json')['denied']
        assert read(target/'environment.json')['initial_model_sha256']==p['initial_model_sha256']
        assert sha(target/'checkpoints/last.pt')==s['last_checkpoint_sha256']
        log=rows(target/'metrics.jsonl');assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        if target.name=='uniform_plain_set_interaction_seed20261019' and (BASE/'recovery/allow_original_epoch_commit_resume.json').exists():
            recovery=verified(BASE/'recovery/allow_original_epoch_commit_resume.json')
            assert sha(recovery['preserved_snapshot'])==recovery['checkpoint_sha256']
            commit=torch.load(recovery['preserved_snapshot'],map_location='cpu',weights_only=False)
            assert log[:commit['epoch']]==commit['metrics'] and commit['epoch']==4 and commit['updates']==332
            events=rows(target/'events.jsonl')
            assert any(e.get('event')=='resume' and e.get('epoch')==4 and e.get('updates')==332 and
                       e.get('rng_restored') and e.get('optimizer_scheduler_restored') for e in events)
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert s['best_epoch']==chosen['epoch'] and s['best_val_H']==chosen['val']['measurement']['heldout']['relative_l1']
        ck=target/'checkpoints/best.pt';assert sha(ck)==s['best_checkpoint_sha256']
        out.append(dict(key=target.name,arm=arm,seed=seed,config=p['config'],parameters=p['parameters'],
            checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=s['best_epoch'],run=str(target)))
    return out


def load_model(m):
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['epoch']==m['best_epoch'] and not ck['volumetric_gt_used'] and ck['config']==m['config']
    if 'run' in m:assert ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json')
    model=model_for(m['config'],m['arm'],'cuda:0').eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==m['parameters']
    return model


def batch_for(ds,i):
    b=move_batch(projection_pool_collate([ds[i]]),'cuda:0')
    return {k:v for k,v in b.items() if not k.startswith('hold_')}


def output_fields(stages):
    out={k:stages[k][0,0].cpu().numpy().copy() for k in ('C0','C1')}
    assert all(v.shape==(64,)*3 and v.dtype==np.float32 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in out.values())
    return out


def save_prediction(root,r,ff):
    path=root/'predictions'/f"{r['condition']}_d{r['draw']}_{r['case_index']:03d}.npz"
    path.parent.mkdir(exist_ok=True)
    assert not path.exists(),'Unexpected uncommitted prediction: preserve and investigate before replay'
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**ff);temp.replace(path)
    return dict(r,prediction=str(path),prediction_sha256=sha(path),reused=False)


def trained(key):
    backend();m=next(x for x in methods() if x['key']==key)
    root=BASE/'evaluation'/key;root.mkdir(parents=True,exist_ok=True)
    settings=verified(BASE/'layouts.json');cases=verified(BASE/'cases.json')
    rr=rows(root/'manifest.jsonl') if (root/'manifest.jsonl').exists() else []
    known={(r['case_id'],r['condition'],r['draw']) for r in rr};assert len(known)==len(rr)
    with AccessGuard(splits=['test'],checkpoints=[m['checkpoint']],output=BASE) as guard,torch.inference_mode():
        for r in rr:assert sha(r['prediction'])==r['prediction_sha256']
        model=load_model(m);state=canonical_worker.probe.state_hash(model.state_dict())
        for s in settings:
            ds=dataset('test',s);assert [r['case_id'] for r in ds.records]==cases
            for i,case in enumerate(cases):
                if (case,s['condition'],s['draw']) in known:continue
                batch=batch_for(ds,i);assert batch['input_station_indices'][0].tolist()==s['input_ids'][case]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                stages=model(batch,return_stages=True);torch.cuda.synchronize();seconds=time.perf_counter()-t
                ff=output_fields(stages)
                r=dict(case_id=case,case_index=i,condition=s['condition'],draw=s['draw'],input_ids=s['input_ids'][case],
                    input_sha256=input_hash(batch),method=key,checkpoint_sha256=m['checkpoint_sha256'],
                    seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],forward_calls=1)
                rr.append(save_prediction(root,r,ff));jsonl(root/'manifest.jsonl',rr)
                del stages,batch,ff
            print(key,s['condition'],s['draw'],len(rr),'/210',flush=True)
        assert len(rr)==210 and canonical_worker.probe.state_hash(model.state_dict())==state
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()) and not guard.denied
    freeze(root/'complete.json',dict(at=now(),method=m,count=210,manifest_sha256=sha(root/'manifest.jsonl'),
        eval=True,no_grad=True,parameter_state_unchanged=True,audit=guard.summary()))


def hook_module():
    registered=verified(BASE.parent/'mrgr_oari_mechanism_v2/protocol.json')['sources']
    for relative in ('scripts/oari_mechanism_v2/runtime.py','scripts/oari_mechanism_v2/interventions.py'):
        path=PROJECT/relative
        assert sha(path)==registered[str(path)],'Frozen intervention differs from the registered OARI v2 implementation'
    previous=sys.modules['runtime']
    try:
        r=load_unique('targeted_exact_oariv2_runtime',PROJECT/'scripts/oari_mechanism_v2/runtime.py')
        sys.modules['runtime']=r
        return load_unique('targeted_exact_oariv2_interventions',PROJECT/'scripts/oari_mechanism_v2/interventions.py')
    finally:sys.modules['runtime']=previous


def frozen():
    backend();m=next(x for x in verified(BASE/'references.json') if x['key']=='uniform_fusion_seed20261019')
    m=dict(m,arm='uniform');root=BASE/'frozen';root.mkdir(exist_ok=True)
    settings=[s for s in verified(BASE/'layouts.json') if s['draw']==0];cases=verified(BASE/'cases.json')
    frozen_protocol=dict(model=m,seed=20261019,training_allowed=False,return_stages=True,
        condition_draws=[dict(condition=s['condition'],draw=s['draw']) for s in settings],
        cases_sha256=sha(BASE/'cases.json'),layouts_sha256=sha(BASE/'layouts.json'),
        parent_protocol_sha256=sha(BASE/'protocol.json'),planned_predictions=70,engineering_noop_forwards=2,
        intervention='OARI v2 obs_token: remove five input contributions to first observability affine; retain learned bias and SiLU',
        sources={str(p):sha(p) for p in (HERE/'predict.py',PROJECT/'scripts/oari_mechanism_v2/interventions.py',
                                        PROJECT/'scripts/oari_mechanism_v2/runtime.py')},
        GT64_before_global_prediction_seal_forbidden=True,GT128_forbidden=True)
    if (root/'protocol.json').exists():assert verified(root/'protocol.json')==frozen_protocol
    else:freeze(root/'protocol.json',frozen_protocol)
    baseline={(r['case_id'],r['condition'],r['draw']):r for r in rows(Path(m['prediction_root'])/'manifest.jsonl')}
    # Search registered mechanism manifests; never assume that filename alone
    # identifies an intervention checkpoint. No suitable seed19 record was found.
    candidates=[]
    for p in BASE.parent.glob('*oari*/prediction_manifest.json'):
        for r in read(p):
            if r.get('checkpoint_sha256')==m['checkpoint_sha256'] and r.get('group')=='obs_token':
                candidates.append(dict(manifest=str(p),record=r))
    freeze(root/'reuse_search.json',dict(at=now(),checkpoint_sha256=m['checkpoint_sha256'],candidates=candidates,
        decision='generate missing matched outputs' if not candidates else 'stop and audit discovered reusable outputs'))
    assert not candidates,'Existing seed19 obs_token outputs require integration, never duplicate'
    intervention=hook_module().Intervention
    rr=rows(root/'manifest.jsonl') if (root/'manifest.jsonl').exists() else []
    known={(r['case_id'],r['condition'],r['draw']) for r in rr}
    with AccessGuard(splits=['test'],checkpoints=[m['checkpoint']],output=BASE) as guard,torch.inference_mode():
        for r in baseline.values():guard.allowed[str((Path(m['prediction_root'])/r['prediction']).resolve())]='historical_sealed_prediction'
        for r in rr:assert sha(r['prediction'])==r['prediction_sha256']
        model=load_model(m);state=canonical_worker.probe.state_hash(model.state_dict())
        for s in settings:
            ds=dataset('test',s)
            for i,case in enumerate(cases):
                if (case,s['condition'],0) in known:continue
                b=batch_for(ds,i);ih=input_hash(b);ref=baseline[case,s['condition'],0]
                assert ref['input_ids']==s['input_ids'][case] and b['input_station_indices'][0].tolist()==s['input_ids'][case]
                if i==0:
                    st=model(b,return_stages=True);current=output_fields(st)
                    with np.load(Path(m['prediction_root'])/ref['prediction'],allow_pickle=False) as z:
                        noop={}
                        for k in ('C0','C1'):
                            a=current[k].astype(float);v=z[k].astype(float)
                            mx=float(np.abs(a-v).max());rel=float(np.abs(a-v).mean()/(np.abs(v).mean()+1e-6))
                            assert mx<=1e-5 and rel<=1e-4;noop[k]=dict(max_abs=mx,relative=rel)
                    freeze(root/f"engineering_{s['condition']}.json",dict(at=now(),case=case,baseline_comparison=noop,extra_noop_forwards=1))
                    del st,current
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                with intervention(model,'obs_token') as hook:
                    stages=model(b,return_stages=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-t
                assert hook.calls==hook.linear_calls==1 and input_hash(b)==ih
                ff=output_fields(stages)
                r=dict(case_id=case,case_index=i,condition=s['condition'],draw=0,input_ids=s['input_ids'][case],input_sha256=ih,
                    method='obs_token_seed20261019',checkpoint_sha256=m['checkpoint_sha256'],seconds=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],forward_calls=1,
                    hook_audit=hook.audit,hook_calls=hook.calls,first_linear_calls=hook.linear_calls,
                    exact_intervention_source_sha256=sha(PROJECT/'scripts/oari_mechanism_v2/interventions.py'))
                rr.append(save_prediction(root,r,ff));jsonl(root/'manifest.jsonl',rr)
                del stages,b,ff
            print('frozen obs_token seed19',s['condition'],len(rr),'/70',flush=True)
        assert len(rr)==70 and canonical_worker.probe.state_hash(model.state_dict())==state
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert sha(m['checkpoint'])==m['checkpoint_sha256'] and not guard.denied
    freeze(root/'complete.json',dict(at=now(),method=m,count=70,manifest_sha256=sha(root/'manifest.jsonl'),
        formal_forwards=70,extra_engineering_forwards=2,parameter_state_unchanged=True,eval=True,no_grad=True,audit=guard.summary()))


def benchmark():
    backend();references=verified(BASE/'references.json');allmodels=references+methods()
    records=[];setting=verified(HISTORY/'layouts.json')['val'][0]
    with AccessGuard(splits=['val'],checkpoints=[m['checkpoint'] for m in allmodels],output=BASE) as guard,torch.inference_mode():
        ds=dataset('val',setting)
        for method in allmodels:
            m=dict(method)
            if m['key'].startswith('uniform_fusion'):m['arm']='uniform'
            model=load_model(m);values=[];peaks=[]
            for index in (0,17,34):
                batch=batch_for(ds,index)
                model(batch,return_stages=True);torch.cuda.synchronize()
                for repeat in range(3):
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    st=model(batch,return_stages=True);torch.cuda.synchronize()
                    values.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated());del st
            records.append(dict(key=m['key'],parameters=sum(p.numel() for p in model.parameters()),
                mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),sd_seconds=float(np.std(values,ddof=1)),
                peak_allocated_bytes=max(peaks),timings=values,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
            del model;torch.cuda.empty_cache()
        assert not guard.denied
    freeze(BASE/'benchmark.json',dict(at=now(),protocol='same exclusiveGPU, validation cases0/17/34 fixedV4, one warmup each +3 measured forwards, batch1 FP32/TF32off return_stagesTrue',
        records=records,audit=guard.summary()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['frozen','trained','benchmark']);p.add_argument('--key');a=p.parse_args()
    if a.action=='trained':trained(a.key)
    else:{'frozen':frozen,'benchmark':benchmark}[a.action]()
