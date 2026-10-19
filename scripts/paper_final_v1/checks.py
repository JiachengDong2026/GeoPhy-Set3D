"""Meaningful boundary, gradient, exact-projector and interruption tests."""
from common import *
import tempfile
import time
import numpy as np
import torch
sys.path.insert(0,str(ROOT/'tests'))
from test_classical_reconstruction import phantom_geometry
from geophy_set3d.reconstruction.classical import project,geometry_tensor,variational
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from solver import CachedProjector,solve
from models import model_for


def main(device):
    backend();results={};g=geometry_tensor(phantom_geometry(2),'cpu')
    torch.manual_seed(123);c=(torch.rand(8,8,8)*.04).requires_grad_(True)
    cached=CachedProjector(g);a=project(c,g);b=cached(c)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga,=torch.autograd.grad(a.sum(),c,retain_graph=True);gb,=torch.autograd.grad(b.sum(),c)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    results['cached_forward_and_vjp_exact']=True
    # Directional finite differences test the true nonlinear mapping, with FP64.
    gd={k:v.double() if v.dtype!=torch.bool else v for k,v in g.items()};cd=c.detach().double().requires_grad_(True);pd=CachedProjector(gd)
    v=torch.randn_like(cd);loss=pd(cd).square().sum()/1e12;grad,=torch.autograd.grad(loss,cd);eps=1e-6
    fd=(pd(cd.detach()+eps*v).square().sum()-pd(cd.detach()-eps*v).square().sum())/(2*eps*1e12)
    torch.testing.assert_close((grad*v).sum(),fd,rtol=1e-5,atol=1e-9);results['nonlinear_gradient_finite_difference']=True
    cfg=dict(iterations=30,minimum_iterations=30,tolerance=1e-6,patience=8,initial_step=1000.,step_bounds=[1e-5,1e6],max_backtracks=20,armijo=1e-4,resolution=8,**{'lambda':1e-4},epsilon=1e-4)
    y=cached(c.detach());valid=g['valid_ray']
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        for method in ('NNLS-NL','Tikhonov','TV'):
            whole,stat=solve(cached,valid,y,method,cfg)
            part=Path(temp)/(method+'.pt');solve(cached,valid,y,method,cfg,save=part,stop_after=11)
            resumed,rs=solve(cached,valid,y,method,cfg,resume=part)
            torch.testing.assert_close(whole,resumed,rtol=0,atol=0);assert stat==rs
            assert torch.isfinite(whole).all() and whole.min()>=0 and whole.max()<=1 and whole.abs().sum()>0
            assert stat['curve'][-1]['objective']<stat['curve'][0]['objective']
            if method!='NNLS-NL':
                old,old_stats=variational(g,y,method,cfg,32284.894580078173,n=8)
                torch.testing.assert_close(old,whole,rtol=0,atol=0)
            results[method+'_updates_and_exact_resume']=True
        zero,zs=solve(cached,valid,y*0,'NNLS-NL',dict(cfg,minimum_iterations=1,patience=2))
        assert zs['status']=='objective_stable' and zs['iterations']<30 and zero.abs().sum()==0
        results['actual_early_stop']=True
    # Deliberately forbidden attempts are isolated from real experimental access logs.
    test=next(r for r in rows(DATA/'source_index.jsonl') if r['split']=='test')
    guard=AccessGuard()
    with guard:
        for p in (DATA/test['gt_volume_64_path'],DATA/test['train_projection_path'],Path(test['source_target_path'])):
            try:
                with p.open('rb'):pass
            except PermissionError:continue
            raise AssertionError('Boundary failed: '+str(p))
    assert len(guard.denied)==3;results['test_GT64_test_measurements_GT128_blocked']=True
    # Real-batch architecture/gradient checks, before any costly 300-epoch job.
    config=verified(OUT/'model_manifest.json')['models'][0]['config']
    setting=verified(OUT/'layouts.json')['val'][0]
    with AccessGuard(splits=['val']) as access:
        ds=dataset('val',setting);sample=ds[0];batch=move_batch(projection_pool_collate([sample]),device)
        inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
        counts={}
        for arm in ('multiscale','single_scale','no_feedback','no_consistency'):
            m=model_for(config,arm,device).train();st=m(inputs,return_stages=True);counts[arm]=sum(x.numel() for x in m.parameters())
            assert st['C1'].shape==(1,1,64,64,64) and torch.isfinite(st['C1']).all()
            st['C1'].square().mean().backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
            if arm=='no_feedback':assert not hasattr(m,'residual_encoder') and not hasattr(m,'residual_adapter');assert torch.equal(st['C0'],st['C1'])
            elif arm=='single_scale':
                assert m.decoder.context.output.weight.grad.abs().sum()>0
                # Zero-start output gates intentionally yield zero internal gradients at update0.
                opt=torch.optim.AdamW(m.parameters(),lr=2e-4);opt.step();opt.zero_grad();m(inputs).square().mean().backward()
                assert m.decoder.context.context8[0].weight.grad.abs().sum()>0
            del m,st
        assert counts['multiscale']==counts['single_scale']==counts['no_consistency']==344678
        results['model_parameters']=counts
        # Measure actual 64³ 4-view baseline cost and cached exactness.
        with np.load(DATA/'geometry/train_geometry.npz') as z:gg=geometry_tensor({k:z[k][setting['input_ids'][sample['case_id']]] for k in ('station_origin_m','ray_direction','t_near','path_length_m','valid_ray')},device)
        pc=CachedProjector(gg);p=torch.rand((64,)*3,device=device)*.03
        torch.testing.assert_close(pc(p),project(p,gg),rtol=0,atol=0)
        ys=batch['input_projection_raw'][0,:,0]
        torch.cuda.synchronize();t=time.perf_counter()
        out,stats=solve(pc,gg['valid_ray'],ys,'NNLS-NL',dict(cfg,resolution=64,iterations=20,minimum_iterations=20))
        torch.cuda.synchronize();results['real_64_smoke']=dict(seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),status=stats['status'],iterations=stats['iterations'])
    results.update(at=now(),device=device,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),access=access.summary(),passed=True)
    write(OUT/'numerical_checks.json',results);print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'cuda:0')
