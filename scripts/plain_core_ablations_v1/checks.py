"""Per-run structure, initialization, real gradient and checkpoint engineering gates."""
from common_stage import *
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training import selfsup
from geophy_set3d.models.lifting import GeometryLifting
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D


def cpu():
    backend();records=[]
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    with TrainGuard() as guard:
        setting=verified(HISTORY/'layouts.json')['val'][0]
        batch={k:v for k,v in projection_pool_collate([dataset('val',setting)[0]]).items() if not k.startswith('hold_')}
        for arm,seed in TASKS:
            p=verified(BASE/'training'/f'{arm}_seed{seed}'/'protocol.json');cfg=p['config']
            full=model_for(cfg,PLAIN,'cpu').eval();rng=torch.get_rng_state().clone();m=model_for(cfg,arm,'cpu').eval()
            assert torch.equal(rng,torch.get_rng_state())
            a,b=full.state_dict(),m.state_dict();assert all(torch.equal(v,a[k]) for k,v in b.items())
            assert type(m.interaction) is type(full.interaction) and type(m.fusion) is type(full.fusion)
            assert type(m.encoder) is type(full.encoder)
            assert not hasattr(m,'observability') and not hasattr(m,'ray_geometry')
            assert cfg['loss']==next(x for x in verified(BASE/'references.json') if x['seed']==seed)['config']['loss']
            if arm==GEOMETRY:
                assert type(m.lifting) is legacy_lifting.PlanarExtrusionLifting
                assert type(m).forward is GeoLiftOARIMRGRRes3D.forward
            elif arm==SINGLE:
                assert type(m.decoder.context) is SingleScaleContext and type(m.lifting) is GeometryLifting
                assert type(m).forward is GeoLiftOARIMRGRRes3D.forward
                spatial=[];h=m.decoder.context.local16.register_forward_pre_hook(lambda mod,args:spatial.append(list(args[0].shape[-3:])))
            else:
                assert not hasattr(m,'residual_encoder') and not hasattr(m,'residual_adapter')
                assert type(m.lifting) is GeometryLifting and type(m.decoder.context) is type(full.decoder.context)
            calls=[];lh=m.lifting.register_forward_hook(lambda mod,args,out:calls.append((args[0].shape[2],out[0].detach().clone())))
            with torch.no_grad():out=m(batch,return_stages=True)
            lh.remove()
            assert all(out[k].shape==(1,1,64,64,64) and torch.isfinite(out[k]).all() and out[k].min()>=0 and out[k].max()<=1 for k in ('C0','C1'))
            assert [x[0] for x in calls]==([32] if arm==NO_PRF else [32,16])
            if arm==GEOMETRY:
                for _,v in calls:torch.testing.assert_close(v[:,:,:,0],v[:,:,:,31],atol=0,rtol=0)
            if arm==SINGLE:h.remove();assert spatial==[[32,32,32],[32,32,32]]
            if arm==NO_PRF:assert torch.equal(out['C0'],out['C1']) and set(out)=={'C0','C1'}
            records.append(dict(key=f'{arm}_seed{seed}',parameters=sum(x.numel() for x in m.parameters()),
                shared_initialization_bit_exact=True,CPU_RNG_exact=True,unchanged_loss=True,lifting_feature_channels=[x[0] for x in calls],passed=True))
            del m,full,out,calls
        assert not guard.denied
    freeze(BASE/'checks/cpu.json',dict(at=now(),passed=True,runs=records,audit=guard.summary()))
    print('CPU engineering PASS: six arms',flush=True)


def gpu():
    backend();assert verified(BASE/'checks/cpu.json')['passed'];records=[]
    with TrainGuard() as guard:
        for arm,seed in TASKS:
            key=f'{arm}_seed{seed}';p=verified(BASE/'training'/key/'protocol.json');cfg=p['config']
            ds=selfsup.make_dataset(cfg);ds.set_epoch(0)
            batch=selfsup.move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),'cuda:0')
            m=model_for(cfg,arm,'cuda:0').train();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            loss,parts,volumes,projections=selfsup.objective(m,batch);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            assert m.interaction.qkv.weight.grad.abs().sum()>0
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.encoder.parameters())
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            assert all(v.min()>=0 and v.max()<=1 and torch.isfinite(v).all() for v in volumes)
            path=BASE/'checks'/f'{key}_checkpoint_smoke.pt'
            saved=dict(model=m.state_dict(),optimizer=opt.state_dict(),config=cfg,engineering_only=True)
            canonical_worker.probe.atomic_checkpoint(path,saved)
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            other=model_for(cfg,arm,'cpu');other.load_state_dict(loaded['model'],strict=True)
            assert canonical_worker.probe.state_hash(m.state_dict())==canonical_worker.probe.state_hash(other.state_dict())
            records.append(dict(key=key,passed=True,loss=float(loss.detach()),gradient_norm=float(norm),batch_size=2,steps=1,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=sha(path),checkpoint_save_reload=True,formal_state_reused=False))
            print('GPU smoke PASS',key,records[-1]['peak_allocated_bytes'],flush=True)
            del m,opt,batch,loss,parts,volumes,projections,saved,loaded,other;torch.cuda.empty_cache()
        assert not guard.denied
    freeze(BASE/'checks/gpu.json',dict(at=now(),passed=True,runs=records,audit=guard.summary()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['cpu','gpu']);a=p.parse_args();{'cpu':cpu,'gpu':gpu}[a.kind]()
