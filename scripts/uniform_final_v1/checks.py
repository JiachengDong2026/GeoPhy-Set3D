"""Engineering gates: masked fusion, retained state, physics, gradients and resume."""
from runtime import *
import argparse
import math
import numpy as np
import torch
import worker
from uniform_models import model_for
from module_models_v2 import model_for as historical_model, UniformFusion
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training import selfsup
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input

def same(a,b):
    if isinstance(a,torch.Tensor):
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
        else:assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
    else:assert a==b,(a,b)

def cpu():
    out=BASE/'checks/cpu.json'
    if out.exists():assert verified(out)['passed'];return
    backend();config=verified(BASE/'training/uniform_seed20261020/protocol.json')['config']
    setting=verified(HISTORY/'layouts.json')['val'][0]
    results=[];state_audit={}
    with TrainGuard() as guard:
        batch=projection_pool_collate([dataset('val',setting)[0]])
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        historical=historical_model(config,'uniform_fusion','cpu').eval()
        control=model_for(config,'uniform','cpu').eval()
        assert historical.state_dict().keys()==control.state_dict().keys()
        assert all(torch.equal(v,control.state_dict()[k]) for k,v in historical.state_dict().items())
        with torch.no_grad():
            reference=historical(batch,return_stages=True)
            current=control(batch,return_stages=True)
            for k in reference:torch.testing.assert_close(reference[k],current[k],rtol=0,atol=0)
        for arm in ARMS:
            model=model_for(config,arm,'cpu').eval()
            state_audit[arm]=[dict(name=n,shape=list(p.shape),numel=p.numel(),requires_grad=p.requires_grad) for n,p in model.named_parameters()]
            if arm=='uniform_feedback_oari':
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in control.state_dict().items())
                assert all(k.startswith('feedback_interaction.') for k in model.state_dict().keys()-control.state_dict().keys())
                assert all(torch.equal(v,model.feedback_interaction.state_dict()[k]) for k,v in model.interaction.state_dict().items())
                with torch.no_grad():
                    control.residual_adapter[2].weight.normal_(std=.01)
                    model.residual_adapter[2].weight.copy_(control.residual_adapter[2].weight)
                    model.feedback_bypass=True
                    a=control(batch,return_stages=True);b=model(batch,return_stages=True)
                    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
                model=model_for(config,arm,'cpu').eval()
            with torch.no_grad():
                stages=model(batch,return_stages=True)
                assert all(stages[k].shape==(1,1,64,64,64) and torch.isfinite(stages[k]).all() and stages[k].min()>=0 and stages[k].max()<=1 for k in ('C0','C1'))
                if arm not in ('uniform_no_feedback','uniform_zero_residual'):
                    torch.testing.assert_close(stages['measurement_prediction_Y0'],project_input(stages['C0'],batch),rtol=0,atol=0)
                if arm!='uniform_no_feedback':
                    torch.testing.assert_close(stages['C0'],stages['C1'],rtol=0,atol=0)
                if arm=='uniform_zero_residual':assert not stages['residual_input'][:,:,0].any()
                if arm=='uniform_feedback_oari':
                    zero=model(batch,return_stages=True,residual_mode='zero')
                    assert not zero['residual_input'][:,:,0].any()
                    assert stages['residual_input'][:,:,0].abs().sum()>0
                if arm=='uniform_no_feedback':assert not hasattr(model,'residual_encoder') and not hasattr(model,'residual_adapter')
            results.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),grid64=True,
                original_physics_checked=arm not in ('uniform_no_feedback','uniform_zero_residual'),
                projection_intentionally_omitted=arm in ('uniform_no_feedback','uniform_zero_residual')))
            del model
        f=UniformFusion()
        features=torch.randn(2,4,3,2,2,2,requires_grad=True)
        visible=torch.rand(2,4,2,2,2)>.4;mask=torch.tensor([[1,1,1,0],[1,0,1,0]],dtype=torch.bool)
        valid=visible & mask[:,:,None,None,None]
        bad=torch.where(valid[:,:,None],features,float('nan'))
        fused,w=f(bad,None,visible,mask)
        torch.testing.assert_close(fused,(torch.where(valid[:,:,None],features,0)*w[:,:,None]).sum(1))
        perm=torch.tensor([3,1,0,2]);p,_=f(bad[:,perm],None,visible[:,perm],mask[:,perm])
        torch.testing.assert_close(p,fused)
        zero,weights=f(bad,None,torch.zeros_like(visible),mask)
        assert torch.equal(zero,torch.zeros_like(zero)) and not weights.any()
        fused.sum().backward();assert torch.isfinite(features.grad).all()
        assert not guard.denied
    source=next(r for r in rows(DATA/'source_index.jsonl') if r['split']=='test')
    with TrainGuard() as probes:
        for path in (DATA/source['train_projection_path'],DATA/source['gt_volume_64_path'],Path(source['source_target_path'])):
            try:
                with path.open('rb'):pass
            except PermissionError:pass
            else:raise AssertionError('Training access guard failed')
        assert len(probes.denied)==3
    freeze(BASE/'parameter_inventory.json',state_audit)
    freeze(out,dict(passed=True,at=now(),results=results,historical_uniform_bit_exact=True,
        feedback_nonzero_adapter_bypass_bit_exact=True,masked_fusion_nan_permutation_all_invalid_gradient=True,
        audit=guard.summary(),boundary_probes=probes.denied,sources={str(Path(__file__)):sha(__file__)}))
    print('CPU engineering gates passed for7arms',flush=True)

def gpu():
    out=BASE/'checks/gpu.json'
    if out.exists():assert verified(out)['passed'];return
    backend();verify_hashes(verified(BASE/'design.json')['training_sources'])
    results=[]
    with TrainGuard() as guard:
        for arm in ARMS:
            target=worker.register(arm,NEW_SEEDS[0],smoke='gradient_config')
            config=verified(target/'protocol.json')['config']
            import models
            models.install_loss(config)
            ds=selfsup.make_dataset(config)
            batch=selfsup.move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),'cuda:0')
            model=model_for(config,arm,'cuda:0').train()
            opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                loss,parts,volumes,projections=selfsup.objective(model,batch)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if step==1 and arm=='uniform_feedback_oari':
                    assert model.feedback_interaction.token_projection.weight.grad.abs().sum()>0
                if arm=='uniform_zero_residual':
                    assert not model.residual_encoder[0].weight.grad[:,0].any()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();del loss,parts,volumes,projections
            results.append(dict(arm=arm,full_objective_backward=True,steps=2))
            del model,opt,batch;torch.cuda.empty_cache()
        assert not guard.denied
    resume=[]
    for arm in ARMS:
        for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
            target=worker.register(arm,NEW_SEEDS[0],smoke=name)
            log=BASE/'checks'/f'{arm}_{name}_{stop}.log'
            with log.open('x') as f:
                r=subprocess.run(args('worker.py','--target',target,'--stop-after',stop),stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT)
            assert r.returncode==0,log
        prefix=BASE/'resume_smoke'/f'{arm}_seed{NEW_SEEDS[0]}'
        a=torch.load(prefix/'uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(prefix/'resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
        for key in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):same(a[key],b[key])
        for ra,rb in zip(a['metrics'],b['metrics']):
            for key in ('epoch','learning_rate','loader_rng_sha256'):same(ra[key],rb[key])
            for split in ('train','val'):same({k:v for k,v in ra[split].items() if k!='seconds'},{k:v for k,v in rb[split].items() if k!='seconds'})
        resume.append(dict(arm=arm,passed=True,float_rtol=1e-4,float_atol=1e-6,integer_rng_exact=True))
        print('GPU objective/recovery passed',arm,flush=True)
    freeze(out,dict(passed=True,at=now(),objective=results,resume=resume,audit=guard.summary(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),sources={str(Path(__file__)):sha(__file__)},
        tolerance_reason='CUDA grid_sample backward nondeterministic atomics; RNG and integer states exact'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','gpu']);o=parser.parse_args()
    try:cpu() if o.mode=='cpu' else gpu()
    except BaseException:
        import traceback
        append(BASE/'checks/failures.jsonl',dict(at=now(),mode=o.mode,traceback=traceback.format_exc()))
        raise
