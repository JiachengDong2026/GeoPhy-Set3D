"""Mechanism/mask/gradient and four-arm cross-process CUDA recovery verification."""
from common import *
import copy
import math
import numpy as np
import torch
from module_models_v2 import ARMS,model_for,UniformFusion
from module_registry_v2 import BASE
from module_train_v2 import register
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch


def compare(a,b):
    if isinstance(a,torch.Tensor):
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
        else:assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):assert a.keys()==b.keys();[compare(a[k],b[k]) for k in a]
    elif isinstance(a,(tuple,list)):assert len(a)==len(b);[compare(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
    else:assert a==b,(a,b)


def main():
    backend();reg=verified(BASE/'registration.json');verify_hashes(reg['protocol_hashes'])
    config=verified(OUT/'model_manifest.json')['models'][0]['config'];setting=verified(OUT/'layouts.json')['val'][0]
    results=[]
    with AccessGuard(splits=['val']) as guard:
        ds=dataset('val',setting);sample=ds[0];batch=projection_pool_collate([sample]);batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        reference=model_for(config,'multiscale','cpu');reference_state={k:v.clone() for k,v in reference.state_dict().items()};del reference
        for arm in ARMS:
            model=model_for(config,arm,'cpu');state=model.state_dict();assert all(torch.equal(v,reference_state[k]) for k,v in state.items())
            if arm!='uniform_fusion':assert state.keys()==reference_state.keys()
            else:assert all(k.startswith('fusion.') for k in reference_state.keys()-state.keys())
            # Invariance is tested directly even when output gates start at zero.
            if arm in ('no_ray_relations','no_observability'):
                bias=model.interaction.pairwise_bias;torch.nn.init.normal_(bias[-1].weight,std=.1)
                x=torch.randn(5,4,4,8);changed=x.clone()
                if arm=='no_ray_relations':changed[...,:3]=torch.randn_like(changed[...,:3])*3
                else:changed[...,3:]=torch.randn_like(changed[...,3:])*3
                torch.testing.assert_close(bias(x),bias(changed),rtol=0,atol=0)
                if arm=='no_observability':
                    emb=model.interaction.observability_embedding;torch.testing.assert_close(emb(torch.randn(7,5)),emb(torch.randn(7,5)),rtol=0,atol=0)
                # Reset exact freshly initialized model after probing learned gate.
                del model;model=model_for(config,arm,'cpu')
            if arm=='uniform_fusion':
                f=torch.randn(1,4,32,2,2,2);geo=torch.randn(1,4,4,2,2,2);vis=torch.ones(1,4,2,2,2,dtype=torch.bool);mask=torch.tensor([[True,True,False,False]])
                fused,w=model.fusion(f,geo,vis,mask);torch.testing.assert_close(fused,f[:,:2].mean(1))
                dirty=f.clone();dirty[:,2:]=float('nan');same,_=model.fusion(dirty,geo,vis,mask);torch.testing.assert_close(fused,same)
                empty,_=model.fusion(f,geo,vis,torch.zeros_like(mask));assert torch.equal(empty,torch.zeros_like(empty))
            model=model.cuda().train();b=move_batch(batch,'cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
            for step in range(2):
                opt.zero_grad();st=model(b,return_stages=True)
                assert all(st[k].shape==(1,1,64,64,64) and torch.isfinite(st[k]).all() and st[k].min()>=0 and st[k].max()<=1 for k in ('C0','C1'))
                if arm=='zero_residual':assert not st['residual_input'][:,:,0].any() and st['measurement_prediction_Y0'] is None
                loss=st['C1'].square().mean();loss.backward()
                assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.encoder.parameters())
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
                if step==1:
                    assert model.decoder.context.context8[0].weight.grad.abs().sum()>0
                    if arm=='zero_residual':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.residual_adapter.parameters())
                opt.step()
            # Actual view reordering remains equivariant, masks still honored.
            model.eval();permutation=torch.arange(4-1,-1,-1,device='cuda:0')
            permuted={k:(v[:,permutation] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==4 and k.startswith(('input_','view_mask')) else v) for k,v in b.items()}
            with torch.no_grad():a=model(b);c=model(permuted)
            torch.testing.assert_close(a,c,rtol=2e-4,atol=2e-6)
            results.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),retained_initial_tensors_exact=True,
                shape_range_gradient=True,view_permutation_passed=True,mechanism_invariance_passed=True))
            del model,st,loss,a,c,opt,b,permuted;torch.cuda.empty_cache()
        audit=guard.summary();assert not audit['denied']
    # Separate real training processes use train.py's authoritative complete epoch
    # checkpoints, four loader workers, optimizer/scheduler and every RNG stream.
    resumes=[]
    for arm in ARMS:
        for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
            target=register(arm,20261011,smoke='module_v2_'+name)
            log=BASE/'resume_logs'/f'{arm}_{name}_stop{stop}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as f:
                p=subprocess.run([sys.executable,'scripts/paper_final_v1/module_train_v2.py','train','--target',str(target),'--stop-after',str(stop)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            assert p.returncode==0,log
        prefix=OUT/'resume_smoke'/f'{arm}_seed20261011'
        a=torch.load(prefix/'module_v2_uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(prefix/'module_v2_resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
        for k in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):compare(a[k],b[k])
        for ra,rb in zip(a['metrics'],b['metrics']):
            for k in ('epoch','learning_rate','loader_rng_sha256'):compare(ra[k],rb[k])
            for split in ('train','val'):compare({k:v for k,v in ra[split].items() if k!='seconds'},{k:v for k,v in rb[split].items() if k!='seconds'})
        resumes.append(dict(arm=arm,passed=True,epochs=2,workers=4,train_subset=4,val_subset=2,float_rtol=1e-4,float_atol=1e-6,integer_and_RNG_exact=True))
    freeze(BASE/'checks.json',dict(at=now(),passed=True,architecture=results,resume=resumes,audit=audit,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),source_sha256=sha(__file__)))
    print('All module mechanisms, real GPU gradients and four-arm recovery checks PASSED',flush=True)


if __name__=='__main__':main()
