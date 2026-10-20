"""CPU equivalence and CUDA objective/recovery checks for a short pilot."""
from runtime import *
import argparse
import math
import traceback
import numpy as np
import torch
from pilot_models import model_for
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from geophy_set3d.training import selfsup
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
import worker

def same(a,b):
    if isinstance(a,torch.Tensor):
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
        else:assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):assert a.keys()==b.keys();[same(a[k],b[k]) for k in a]
    elif isinstance(a,(tuple,list)):assert len(a)==len(b);[same(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
    else:assert a==b,(a,b)

def cpu():
    if (BASE/'cpu_checks.json').exists():assert verified(BASE/'cpu_checks.json')['passed'];return
    backend();config=verified(BASE/'model_manifest.json')['models'][0]['config'];setting=verified(BASE/'layouts.json')['val'][0]
    results=[]
    with DevGuard(splits=['val']) as guard:
        batch=projection_pool_collate([dataset('val',setting)[0]]);batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        original=model_for(config,'baseline','cpu').eval();state={k:v.clone() for k,v in original.state_dict().items()}
        with torch.no_grad():reference=original(batch,return_stages=True)
        for arm in ARMS[1:]:
            model=model_for(config,arm,'cpu').eval();initial=model.state_dict()
            assert all(torch.equal(v,initial[k]) for k,v in state.items())
            assert all(k.startswith('feedback_interaction.') for k in initial.keys()-state.keys())
            assert all(torch.equal(v,model.feedback_interaction.state_dict()[k]) for k,v in model.interaction.state_dict().items())
            # Bypassing only the additional OARI must reproduce every original
            # stage, even after making the originally zero output adapter active.
            if arm=='feedback_oari':
                with torch.no_grad():
                    original.residual_adapter[2].weight.normal_(std=.01)
                    model.residual_adapter[2].weight.copy_(original.residual_adapter[2].weight)
                    model.feedback_bypass=True;ref=original(batch,return_stages=True);out=model(batch,return_stages=True)
                    for k in ref:torch.testing.assert_close(out[k],ref[k],rtol=0,atol=0)
                model=model_for(config,arm,'cpu').eval()
            with torch.no_grad():
                out=model(batch,return_stages=True)
                for key in ('C0','C1'):assert out[key].shape==(1,1,64,64,64) and torch.isfinite(out[key]).all() and out[key].min()>=0 and out[key].max()<=1
                torch.testing.assert_close(out['C0'],reference['C0'],rtol=0,atol=0)
                torch.testing.assert_close(out['C1'],out['C0'],rtol=0,atol=0)
                torch.testing.assert_close(out['measurement_prediction_Y0'],project_input(out['C0'],batch),rtol=0,atol=0)
                if arm.endswith('zero_residual'):assert not out['residual_input'][:,:,0].any()
                else:assert out['residual_input'][:,:,0].abs().sum()>0
                perm=torch.tensor([3,1,0,2]);reordered={k:(v[:,perm] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==4 and k.startswith(('input_','view_mask')) else v) for k,v in batch.items()}
                torch.testing.assert_close(model(reordered),out['C1'],rtol=2e-4,atol=2e-6)
            results.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),retained_tensors_exact=True,
                grid64_range=True,initial_C0_C1_equal=True,permutation=True,original_physics=True))
            print('CPU checks passed',arm,flush=True)
        assert not guard.denied
    # Explicit forbidden-file probes are separate from ordinary access audits.
    r=next(r for r in rows(DATA/'source_index.jsonl') if r['split']=='test')
    with DevGuard() as g:
        for p in (DATA/r['gt_volume_64_path'],DATA/r['train_projection_path'],Path(r['source_target_path'])):
            try:
                with p.open('rb'):pass
            except PermissionError:pass
            else:raise AssertionError('boundary failed')
        assert len(g.denied)==3
    freeze(BASE/'cpu_checks.json',dict(passed=True,at=now(),results=results,nonzero_adapter_bypass_bit_exact=True,audit=guard.summary(),forbidden_before_bytes=g.denied))

def gpu():
    if (BASE/'gpu_checks.json').exists():assert verified(BASE/'gpu_checks.json')['passed'];return
    backend();verify_hashes(verified(BASE/'design.json')['sources'])
    config=verified(BASE/'model_manifest.json')['models'][0]['config'];results=[]
    with DevGuard(splits=['train','val']) as guard:
        ds=selfsup.make_dataset(config);batch=selfsup.move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),'cuda:0')
        for arm in ARMS:
            model=model_for(config,arm,'cuda:0').train();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            for step in range(2):
                opt.zero_grad(set_to_none=True);loss,parts,vols,projs=selfsup.objective(model,batch);loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if step==1:
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.residual_encoder.parameters())
                    if arm!='baseline':assert model.feedback_interaction.token_projection.weight.grad.abs().sum()>0
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();del loss,parts,vols,projs
            results.append(dict(arm=arm,full_objective_backward=True,steps=2));del model,opt;torch.cuda.empty_cache()
        assert not guard.denied
    resumes=[]
    for arm in ARMS[1:]:
        for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
            target=worker.register(arm,smoke=name);log=BASE/'checks'/f'{arm}_{name}_{stop}.log';log.parent.mkdir(exist_ok=True)
            with log.open('x') as f:result=subprocess.run(args('worker.py','--target',target,'--stop-after',stop),stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT)
            assert result.returncode==0,log
        prefix=BASE/'resume_smoke'/f'{arm}_seed{SEED}'
        a=torch.load(prefix/'uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(prefix/'resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
        for key in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):same(a[key],b[key])
        for ra,rb in zip(a['metrics'],b['metrics']):
            for key in ('epoch','learning_rate','loader_rng_sha256'):same(ra[key],rb[key])
            for split in ('train','val'):same({k:v for k,v in ra[split].items() if k!='seconds'},{k:v for k,v in rb[split].items() if k!='seconds'})
        resumes.append(dict(arm=arm,passed=True,epochs=2,workers=4,float_rtol=1e-4,float_atol=1e-6,RNG_integer_exact=True))
        print('GPU recovery passed',arm,flush=True)
    freeze(BASE/'gpu_checks.json',dict(passed=True,at=now(),gradients=results,resume=resumes,audit=guard.summary(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args()
    try:cpu() if a.mode=='cpu' else gpu()
    except BaseException:
        append(BASE/'check_failures.jsonl',dict(at=now(),mode=a.mode,traceback=traceback.format_exc()));raise
