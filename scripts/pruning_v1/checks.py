"""Fail-closed unit, actual64³ physics/gradient and cross-process recovery checks."""
from runtime import *
import copy
import math
import traceback
import numpy as np
import torch
from unittest.mock import patch
from pruned_models import model_for,UniformFusion,PrunedMRGR
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D,project_input,Y_SCALE
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from geophy_set3d.training import selfsup
from geophy_set3d.data.measurement_only import dual_measurement_collate
from module_checks_v2 import compare
import train_worker as worker

def uniform_unit():
    f=torch.randn(2,4,32,2,2,2,requires_grad=True);geo=torch.randn(2,4,4,2,2,2)
    vis=torch.ones(2,4,2,2,2,dtype=torch.bool);mask=torch.tensor([[True,True,False,False],[True,False,True,False]])
    vis[0,1,0,0,0]=False;valid=vis & mask[:,:,None,None,None]
    dirty=torch.where(valid[:,:,None],f,torch.full_like(f,float('nan')))
    fusion=UniformFusion();out,w=fusion(dirty,geo,vis,mask)
    expected=torch.where(valid[:,:,None],f,0).sum(1)/valid.sum(1).clamp_min(1)[:,None]
    torch.testing.assert_close(out,expected);assert out.shape==(2,32,2,2,2) and torch.isfinite(out).all()
    perm=torch.tensor([3,1,0,2]);other,_=fusion(dirty[:,perm],geo[:,perm],vis[:,perm],mask[:,perm]);torch.testing.assert_close(out,other)
    empty,weights=fusion(dirty,geo,vis,torch.zeros_like(mask));assert not empty.any() and not weights.any()
    out.sum().backward();assert torch.isfinite(f.grad).all() and not f.grad.masked_select(~valid[:,:,None].expand_as(f)).any()
    assert not list(fusion.parameters())
    return dict(permutation=True,invalid_NaN_excluded=True,all_invalid_exact_zero=True,gradient=True,shape=list(out.shape))

def boundaries():
    rr=rows(DATA/'source_index.jsonl');test=next(r for r in rr if r['split']=='test')
    paths=[DATA/test['gt_volume_64_path'],DATA/test['train_projection_path'],Path(test['source_target_path'])]
    with DevGuard(splits=['train','val']) as g:
        for p in paths:
            try:
                with p.open('rb'):pass
            except PermissionError:pass
            else:raise AssertionError('boundary failed '+str(p))
        assert len(g.denied)==3
    return dict(passed=True,blocked_before_file_bytes=g.denied)

def actual(config):
    setting=verified(BASE/'layouts.json')['val'][0];results=[]
    assert PrunedMRGR.forward is GeoLiftOARIMRGRRes3D.forward
    with DevGuard(splits=['train','val']) as guard:
        batch=projection_pool_collate([dataset('val',setting)[0]])
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')};b=move_batch(batch,'cuda:0')
        for arm in ARMS:
            m=model_for(config,arm,'cuda:0');assert not hasattr(m,'observability') and not hasattr(m.interaction,'observability_embedding')
            assert not any('observability' in k for k in m.state_dict())
            ref=model_for(config,'uniform_fusion' if arm.startswith('uniform') else 'multiscale','cuda:0')
            # Independent full-width algebraic reference: zero appended5 inputs
            # AND zero8 embedded outputs, while retaining original biases.
            class ZeroDescriptor(torch.nn.Module):
                def forward(self,directions,mask,count):return directions.new_zeros((len(directions),5))
            ref.observability=ZeroDescriptor()
            handle=ref.interaction.observability_embedding.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
            m.eval();ref.eval()
            with torch.no_grad():
                with patch('torch.linalg.eigvalsh',side_effect=AssertionError('deleted eigensolver called')):
                    st=m(b,return_stages=True)
                sr=ref(b,return_stages=True)
                for k in st:
                    if isinstance(st[k],torch.Tensor):torch.testing.assert_close(st[k],sr[k],rtol=3e-4,atol=2e-5)
                for k in ('C0','C1'):assert st[k].shape==(1,1,64,64,64) and torch.isfinite(st[k]).all() and st[k].min()>=0 and st[k].max()<=1
                torch.testing.assert_close(st['measurement_prediction_Y0'],project_input(st['C0'],b),rtol=0,atol=0)
                valid=b['input_valid_ray'].bool() & b['view_mask'][:,:,None,None,None]
                torch.testing.assert_close(st['measurement_residual'],torch.where(valid,b['input_projection_raw']-st['measurement_prediction_Y0'],0),rtol=0,atol=0)
                perm=torch.arange(3,-1,-1,device='cuda:0')
                reordered={k:(v[:,perm] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==4 and k.startswith(('input_','view_mask')) else v) for k,v in b.items()}
                torch.testing.assert_close(m(reordered),st['C1'],rtol=2e-4,atol=2e-6)
            handle.remove();del ref,sr,st
            # Pair relations are active after the intentionally zero initial gate.
            bias=m.interaction.pairwise_bias;original=bias[-1].weight.detach().clone()
            with torch.no_grad():
                bias[-1].weight.normal_(std=.1);x=torch.randn(5,4,4,3,device='cuda:0');a=bias(x);c=bias(x+1)
                assert (a-c).abs().max()>1e-6;bias[-1].weight.copy_(original)
            opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4);m.train()
            dual=selfsup.move_dual_batch(dual_measurement_collate([selfsup.make_dataset(config)[0],selfsup.make_dataset(config)[1]]),'cuda:0')
            gradients=[]
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                with patch('torch.linalg.eigvalsh',side_effect=AssertionError('deleted eigensolver called')):
                    loss,parts,volumes,projections=selfsup.objective(m,dual);loss.backward()
                assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.encoder.parameters())
                if step==1:
                    assert m.decoder.context.context8[0].weight.grad.abs().sum()>0
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.residual_encoder.parameters())
                    assert m.interaction.pairwise_bias[0].weight.grad.abs().sum()>0
                gradients.append(dict(step=step,loss=float(loss),gradient_norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))))
                opt.step();del loss,parts,volumes,projections
            results.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),actual_grid=[64,64,64],
                full_width_reference_matches=True,feedback_source_inherited=True,physics_and_signed_residual_exact=True,
                descriptor_attributes_parameters_eigensolver_absent=True,view_permutation=True,ray_relations_active=True,full_objective_backward=gradients))
            del m,opt,dual;torch.cuda.empty_cache();print('Actual64 physics and gradients passed',arm,flush=True)
        assert not guard.denied
    return results,guard.summary()

def recovery():
    results=[]
    for arm in ARMS:
        for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
            target=worker.engine.register(arm,SEEDS[0],smoke=name)
            path=BASE/'checks'/f'{arm}_{name}_stop{stop}.log'
            with path.open('x') as f:
                result=subprocess.run([sys.executable,*job_args('train_worker.py','--target',target,'--stop-after',stop)],stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT)
            assert result.returncode==0,path
        root=BASE/'resume_smoke'/f'{arm}_seed{SEEDS[0]}'
        a=torch.load(root/'uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(root/'resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
        for k in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):compare(a[k],b[k])
        for ra,rb in zip(a['metrics'],b['metrics']):
            for k in ('epoch','learning_rate','loader_rng_sha256'):compare(ra[k],rb[k])
            for split in ('train','val'):compare({k:v for k,v in ra[split].items() if k!='seconds'},{k:v for k,v in rb[split].items() if k!='seconds'})
        results.append(dict(arm=arm,passed=True,workers=4,subset_train=4,subset_val=2,epochs=2,
            float_rtol=1e-4,float_atol=1e-6,integer_RNG_exact=True,restored=['model','best_model','optimizer','scheduler','rng','loader_rng','epoch_sampling','updates','best_epoch','best_score']))
        print('Cross-process recovery passed',arm,flush=True)
    return results

def main():
    backend();BASE.joinpath('checks').mkdir(exist_ok=True)
    if (BASE/'checks/passed.json').exists():assert verified(BASE/'checks/passed.json')['passed'];return
    try:
        verify_hashes(verified(BASE/'design.json')['sources'])
        unit=uniform_unit();boundary=boundaries();config=verified(BASE/'model_manifest.json')['models'][0]['config']
        shapes,audit=actual(config);resumes=recovery()
        freeze(BASE/'checks/passed.json',dict(at=now(),passed=True,unit=unit,boundary=boundary,architecture=shapes,resume=resumes,audit=audit,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
    except BaseException:
        append(BASE/'checks/failures.jsonl',dict(at=now(),traceback=traceback.format_exc(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')));raise

if __name__=='__main__':main()
