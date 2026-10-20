"""Same-device, same-input timing; development projections only, no GT."""
from runtime import *
import time
import numpy as np
import torch
from uniform_models import model_for
from module_models_v2 import model_for as old_model
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch

def main():
    path=BASE/'benchmark.json'
    if path.exists():assert verified(path)['passed'];return
    backend();assert verified(BASE/'training_complete.json')['complete']
    reused=verified(BASE/'reuse_methods.json')
    selected=[next(m for m in reused if m['key']==k) for k in ('MRGR','uniform_fusion_seed20261011','uniform_pruned_observability_seed20261011')]
    for arm in ('uniform_feedback_oari','uniform_no_feedback','uniform_zero_residual','uniform_no_ray_relations','uniform_no_consistency','uniform_single_scale'):
        if arm=='uniform_feedback_oari':
            available=[s for s in NEW_SEEDS if (BASE/'training'/f'{arm}_seed{s}'/'summary.json').exists()]
            if not available:continue
            seed=available[0]
        else:seed=ABLATION_SEEDS[0]
        target=BASE/'training'/f'{arm}_seed{seed}'
        if not (target/'summary.json').exists():continue
        p=verified(target/'protocol.json')
        selected.append(dict(key=target.name,arm=arm,config=p['config'],checkpoint=str(target/'checkpoints/best.pt')))
    results=[];shapes={}
    with AccessGuard(splits=['val'],checkpoints=[m['checkpoint'] for m in selected],output=BASE) as guard:
        sample=dataset('val',verified(HISTORY/'layouts.json')['val'][0])[0]
        batch=move_batch(projection_pool_collate([sample]),'cuda:0')
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        for m in selected:
            c=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
            factory=old_model if m['key']=='MRGR' else model_for
            arm='multiscale' if m['key']=='MRGR' else 'uniform' if m['key']=='uniform_fusion_seed20261011' else m['arm']
            if arm=='uniform_pruned_observability':
                import importlib.util
                spec=importlib.util.spec_from_file_location('benchmark_pruned_frozen',PROJECT/'scripts/pruning_v1/pruned_models.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                # Exact original dispatch registry; no forward implementation change.
                module.ARMS=('pruned_observability','uniform_pruned_observability')
                factory=module.model_for
            model=factory(m['config'],arm,'cuda:0').eval();model.load_state_dict(c['model'],strict=True)
            calls=[];hooks=[]
            if arm=='uniform':
                for name in ('encoder','lifting','interaction','fusion','decoder','residual_encoder','residual_adapter'):
                    def capture(module,inputs,outputs,name=name):
                        def shape(v):
                            if isinstance(v,torch.Tensor):return list(v.shape)
                            if isinstance(v,(tuple,list)):return [shape(x) for x in v]
                            return type(v).__name__
                        calls.append(dict(module=name,inputs=shape(inputs),outputs=shape(outputs)))
                    hooks.append(getattr(model,name).register_forward_hook(capture))
            with torch.no_grad():model(batch,return_stages=True)
            for h in hooks:h.remove()
            if calls:shapes[m['key']]=calls
            with torch.no_grad():
                for _ in range(3):model(batch)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();seconds=[]
                for _ in range(10):
                    torch.cuda.synchronize();t=time.perf_counter();model(batch);torch.cuda.synchronize()
                    seconds.append(time.perf_counter()-t)
            results.append(dict(method=m['key'],parameters=sum(p.numel() for p in model.parameters()),
                checkpoint_sha256=sha(m['checkpoint']),latency_seconds=seconds,median_seconds=float(np.median(seconds)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),warmups=3,repetitions=10))
            del model,c;torch.cuda.empty_cache()
        assert not guard.denied
    freeze(path,dict(at=now(),passed=True,source_sha256=sha(__file__),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        batch=1,views=4,case=sample['case_id'],condition='Val-A-V4 development case0',
        timing_scope='synchronized C0+C1 forward; excludes loading and output I/O; common single-device session',
        dtype='FP32, TF32 disabled',results=results,main_tensor_shapes=shapes,audit=guard.summary()))

if __name__=='__main__':main()
