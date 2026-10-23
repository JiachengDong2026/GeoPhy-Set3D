"""Reuse the formal measurement-only prediction and matched timing implementation."""
from common_stage import *
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
import time


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
    p=argparse.ArgumentParser();p.add_argument('action',choices=['trained','benchmark']);p.add_argument('--key');a=p.parse_args()
    if a.action=='trained':trained(a.key)
    else:benchmark()
