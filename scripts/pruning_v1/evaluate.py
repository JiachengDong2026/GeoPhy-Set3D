"""Development-only adapter and matched actual64³ inference cost measurement."""
from runtime import *
import argparse
import math
import time
import numpy as np
import torch
import evaluation as engine
from pruned_models import model_for
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch

engine.OUT=BASE;engine.AccessGuard=DevGuard;engine.model_for=model_for;engine.SCOPE=None

def benchmark():
    if (BASE/'benchmark.json').exists():return verified(BASE/'benchmark.json')
    backend();methods=[m for m in verified(BASE/'reuse_methods.json') if m['arm']!='no_observability']+verified(BASE/'methods.json')
    setting=verified(BASE/'layouts.json')['val'][0];result=[]
    with DevGuard(splits=['val'],checkpoints=[m['checkpoint'] for m in methods]) as guard:
        sample=dataset('val',setting)[0];b=move_batch(projection_pool_collate([sample]),'cuda:0');b={k:v for k,v in b.items() if not k.startswith('hold_')}
        for info in methods:
            assert sha(info['checkpoint'])==info['checkpoint_sha256']
            ck=torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
            m=model_for(info['config'],info['arm'],'cuda:0').eval();m.load_state_dict(ck['model'],strict=True);del ck
            count=[0];handles=[]
            def hook(module,args,out):
                if isinstance(module,torch.nn.Linear):count[0]+=out.numel()*module.in_features
                else:count[0]+=out.numel()*math.prod(module.kernel_size)*module.in_channels//module.groups
            for module in m.modules():
                if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.Conv3d)):handles.append(module.register_forward_hook(hook))
            with torch.no_grad():m(b)
            for handle in handles:handle.remove()
            with torch.no_grad():
                for _ in range(10):m(b)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(40):
                    torch.cuda.synchronize();start=time.perf_counter();out=m(b);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    assert out.shape==(1,1,64,64,64) and torch.isfinite(out).all()
            result.append(dict(method=info['key'],arm=info['arm'],seed=info['seed'],parameters=sum(p.numel() for p in m.parameters()),
                learned_conv_linear_MACs=count[0],median_seconds=float(np.median(times)),p90_seconds=float(np.quantile(times,.9)),times_seconds=times,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),warmup=10,repeats=40))
            del m,out;torch.cuda.empty_cache()
            print('Matched benchmark',info['key'],result[-1]['median_seconds'],flush=True)
        assert not guard.denied
    freeze(BASE/'benchmark.json',dict(at=now(),results=result,audit=guard.summary(),case_id=sample['case_id'],condition=setting['condition'],
        boundary='B1 V4 actual64³ end-to-end network C0+C1 including internal P(C0); excludes loader, transfer, external projection and file I/O; synchronized CUDA wall time',
        MAC_scope='executed Conv2d/Conv3d/Linear multiply-accumulates only; excludes attention matmuls, interpolation/lifting, normalization, eigensolver and physics projector; not total FLOPs',
        precision='FP32; TF32 disabled; CPU2 threads; same GPU and entry for all A/B/C/D both seeds'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','score','benchmark']);p.add_argument('--key');args=p.parse_args()
    verify_hashes(verified(BASE/'design.json')['sources'])
    if args.action=='benchmark':return benchmark()
    methods=verified(BASE/'methods.json')
    if args.action=='predict':engine.predict_neural(next(m for m in methods if m['key']==args.key),'val')
    else:
        seal=BASE/'evaluation/val/all_predictions_sealed.json'
        if seal.exists():assert verified(seal)['methods']==methods
        else:engine.seal('val',methods)
        if (BASE/'evaluation/val/metrics.jsonl').exists():
            assert read(BASE/'evaluation/val/gt_access.json')['unique_GT64_files']==36
        else:engine.score('val')

if __name__=='__main__':main()
