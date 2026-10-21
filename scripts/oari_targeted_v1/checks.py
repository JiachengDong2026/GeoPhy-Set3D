"""Engineering gates for a strict interaction-only replacement."""
from stage import *
from geophy_set3d.models.set_interaction import GeometryConditionedSetInteraction
from geophy_set3d.models.geolift_plain_set_mrgr import PlainSetStage0
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training import selfsup


def cpu():
    backend();results=[]
    with AccessGuard(splits=['val'],output=BASE) as guard:
        setting=verified(HISTORY/'layouts.json')['val'][0]
        batch=projection_pool_collate([dataset('val',setting)[0]])
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        for seed in (20261011,20261019):
            cfg=verified(BASE/'training'/f'{PLAIN}_seed{seed}'/'protocol.json')['config']
            full=model_for(cfg,'uniform','cpu').eval();plain=model_for(cfg,PLAIN,'cpu').eval()
            a,b=full.state_dict(),plain.state_dict()
            shared=[k for k in a if not k.startswith('interaction.')]
            assert shared and all(k in b and torch.equal(a[k],b[k]) for k in shared)
            assert all(k in a for k in b if not k.startswith('interaction.'))
            for k,v in plain.interaction.state_dict().items():
                if not k.startswith('token_projection.'):assert torch.equal(v,full.interaction.state_dict()[k])
            assert plain.interaction.token_projection.in_features==48 and full.interaction.token_projection.in_features==56
            assert type(plain).forward is GeoLiftOARIMRGRRes3D.forward
            assert not hasattr(plain,'ray_geometry') and not hasattr(plain,'observability')
            captured=[]
            def hook(module,args,out):captured.append(tuple(x.detach().clone() for x in out))
            for model in (full,plain):
                h=model.lifting.register_forward_hook(hook)
                with torch.no_grad():
                    st=model(batch,return_stages=True)
                    for s in ('C0','C1'):
                        assert st[s].shape==(1,1,64,64,64) and torch.isfinite(st[s]).all() and st[s].min()>=0 and st[s].max()<=1
                h.remove()
            # Lifting is called twice per forward; compare the first calls only.
            assert len(captured)==4
            for x,y in zip(captured[0],captured[2]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            with torch.no_grad():
                torch.testing.assert_close(full.geometry_descriptor(batch),plain.geometry_descriptor(batch),atol=0,rtol=0)
                p,info=PlainSetStage0.forward(plain,batch,return_attention=True)
                assert info['interaction_kind']=='plain_set' and 'pairwise_bias' not in info
                assert info['interaction_attention'].shape[:4]==(1,4,4,4)
                invalid=dict(batch,view_mask=torch.zeros_like(batch['view_mask']))
                zero=plain(invalid,return_stages=True)
                assert all(torch.isfinite(zero[s]).all() and zero[s].min()>=0 and zero[s].max()<=1 for s in ('C0','C1'))
            results.append(dict(seed=seed,parameters=sum(x.numel() for x in plain.parameters()),shared_initialization_exact=True,
                first_lifting_bit_exact=True,geometry_bit_exact=True,unchanged_MRGR_forward=True,all_invalid_full_forward_finite=True))
        m=GeometryConditionedSetInteraction()
        f=torch.randn(2,4,32,2,2,2,requires_grad=True);g=torch.randn(2,4,4,2,2,2)
        vis=torch.rand(2,4,2,2,2)>.3;vm=torch.tensor([[1,1,1,0],[0,0,0,0]],dtype=torch.bool)
        valid=vis&vm[:,:,None,None,None]
        bad=torch.where(valid[:,:,None],f,float('nan'));badg=torch.where(valid[:,:,None],g,float('nan'))
        out,att=m(bad,badg,vis,vm,True)
        assert torch.isfinite(out).all() and torch.isfinite(att).all() and not out[1].any() and not att[1].any()
        permutation=torch.tensor([3,1,0,2]);other=m(bad[:,permutation],badg[:,permutation],vis[:,permutation],vm[:,permutation])
        torch.testing.assert_close(other,out[:,permutation],atol=1e-6,rtol=1e-5)
        padded=m(torch.cat((bad,torch.full_like(bad[:,:1],float('nan'))),1),
            torch.cat((badg,torch.full_like(badg[:,:1],float('nan'))),1),
            torch.cat((vis,torch.zeros_like(vis[:,:1])),1),torch.cat((vm,torch.zeros_like(vm[:,:1])),1))
        torch.testing.assert_close(padded[:,:4],out,atol=1e-6,rtol=1e-5);assert not padded[:,4].any()
        out.square().mean().backward();assert torch.isfinite(f.grad).all()
        assert not guard.denied
    freeze(BASE/'checks/cpu.json',dict(at=now(),passed=True,models=results,
        mask_nan_safe=True,permutation_equivariance=True,padding_invariance=True,all_invalid_interaction_zero=True,finite_backward=True,audit=guard.summary()))
    print('CPU engineering PASS',results,flush=True)


def gpu():
    backend();assert verified(BASE/'checks/cpu.json')['passed'];results=[]
    with AccessGuard(splits=['train','val'],output=BASE) as guard:
        for arm,seed in (TASKS[0],TASKS[1]):
            cfg=verified(BASE/'training'/f'{arm}_seed{seed}'/'protocol.json')['config']
            ds=selfsup.make_dataset(cfg);ds.set_epoch(0)
            batch=selfsup.move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),'cuda:0')
            m=model_for(cfg,arm,'cuda:0').train();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
            loss_values=[];torch.cuda.reset_peak_memory_stats()
            for i in range(2):
                opt.zero_grad(set_to_none=True)
                loss,parts,volumes,projections=selfsup.objective(m,batch)
                assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                assert m.interaction.qkv.weight.grad.abs().sum()>0
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                assert all(v.min()>=0 and v.max()<=1 and torch.isfinite(v).all() for v in volumes)
                loss_values.append(float(loss.detach()))
                del loss,parts,volumes,projections
            results.append(dict(arm=arm,losses=loss_values,steps=2,single_repeated_batch=True,formal_state_reused=False,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del m,opt,batch;torch.cuda.empty_cache()
        assert not guard.denied
    freeze(BASE/'checks/gpu.json',dict(at=now(),passed=True,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],results=results,audit=guard.summary()))
    print('GPU engineering PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['cpu','gpu']);a=p.parse_args();{'cpu':cpu,'gpu':gpu}[a.kind]()
