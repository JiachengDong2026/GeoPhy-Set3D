"""Fail-closed real-CUDA mechanism, gradient and cross-process resume gates."""
from runtime import *
import math
from geophy_set3d.training import selfsup
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.utils import move_batch
from torch.utils.data import Subset


def compare(a,b):
    if isinstance(a,torch.Tensor):
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
        else:assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
    else:assert a==b,(a,b)


def architecture():
    backend();protocol=verified(BASE/'protocol.json');verify_hashes(protocol['sources'])
    result=[]
    with TrainGuard() as guard:
        for seed in SEEDS:
            p=verified(run_protocol(seed)/'protocol.json');config=p['config']
            model=model_for(config,ARM,'cuda:0').train()
            from module_models_v2 import UniformFusion
            assert type(model.fusion) is UniformFusion and not list(model.fusion.parameters())
            assert state_hash(model.state_dict())==p['initial_model_sha256']
            ds=selfsup.make_dataset(config);raw=dual_measurement_collate([ds[0],ds[1]])
            batch=selfsup.move_dual_batch(raw,'cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            expected={}
            for key in ('A','B'):
                b=batch[key];grid,vis=model.lifting.angular_grid(b)
                valid=vis.reshape(len(vis),b['view_mask'].shape[1],32,32,32)
                directions,relations=model.ray_geometry(b['input_station_origin_m'],model.lifting.voxel_xyz,b['input_station_azimuth_deg'],valid)
                local=valid.flatten(2).transpose(1,2).reshape(-1,valid.shape[1])
                count=b['view_mask'].sum(1)[:,None].expand(-1,32**3).reshape(-1)
                expected[b['input_projection'].data_ptr()]=dict(visibility=valid,geometry=model.geometry_descriptor(b),relations=relations,
                                    observability=model.observability(directions,local,count))
            calls=[];features=[];current=[None];interaction_calls=[0];fusion_calls=[0]
            def before_lift(module,args):current[0]=args[1]['input_projection'].data_ptr()
            def lifted(module,args,out):
                f,b=args;e=expected[b['input_projection'].data_ptr()];x,vis=out
                assert torch.equal(vis,e['visibility']) and x.dtype==f.dtype and x.device==f.device
                assert x.shape==(*f.shape[:3],32,32,32)
                plane=F.interpolate(f.flatten(0,1),size=(32,32),mode='bilinear',align_corners=True)
                independent=plane.transpose(-1,-2).reshape(*f.shape[:3],1,32,32).repeat(1,1,1,32,1,1)
                assert torch.equal(x,independent)
                assert bool((x.min()>=f.min()-1e-5)&(x.max()<=f.max()+1e-5))
                assert torch.equal(x[:,:,:,0],x[:,:,:,31])
                assert x.std()>1e-7
                f.retain_grad();features.append(f);calls.append(f.shape[2])
            def interaction(module,args):
                e=expected[current[0]]
                assert torch.equal(args[1],e['geometry']) and torch.equal(args[2],e['visibility'])
                assert torch.equal(args[4],e['relations']) and torch.equal(args[5],e['observability'])
                interaction_calls[0]+=1
            def fusion(module,args,out):
                e=expected[current[0]]
                assert torch.equal(args[1],e['geometry']) and torch.equal(args[2],e['visibility'])
                valid=args[2]&args[3][:,:,None,None,None]
                weights=valid.to(args[0].dtype)/valid.sum(1,keepdim=True).clamp_min(1).to(args[0].dtype)
                assert torch.equal(out[1],weights)
                assert torch.equal(out[0],(torch.where(valid[:,:,None],args[0],0)*weights[:,:,None]).sum(1))
                fusion_calls[0]+=1
            hooks=[model.lifting.register_forward_pre_hook(before_lift),model.lifting.register_forward_hook(lifted),
                   model.interaction.register_forward_pre_hook(interaction),model.fusion.register_forward_hook(fusion)]
            gradients=[]
            for step in range(2):
                calls.clear();features.clear();interaction_calls[0]=fusion_calls[0]=0
                opt.zero_grad(set_to_none=True)
                loss,parts,volumes,projections=selfsup.objective(model,batch)
                assert calls==[32,16,32,16] and interaction_calls[0]==2 and fusion_calls[0]==4
                assert torch.isfinite(loss) and all(torch.isfinite(v).all() and v.min()>=0 and v.max()<=1 and v.std()>1e-7 for v in volumes)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                group={n:float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None))
                       for n,module in [('encoder',model.encoder),('interaction',model.interaction),
                                        ('decoder_context',model.decoder.context),('residual_encoder',model.residual_encoder),('residual_adapter',model.residual_adapter)]}
                if step==1:
                    assert all(v>0 for v in group.values()),group
                    assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in features)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                gradients.append(dict(step=step+1,loss=float(loss.detach()),gradient_abs_sums=group))
                del loss,parts,volumes,projections
            for h in hooks:h.remove()
            assert model.lifting.angular_grid.__func__ is GeometryLifting.angular_grid
            result.append(dict(seed=seed,passed=True,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                both_lifting_calls_replaced=True,original_visibility_descriptors_relations_observability_equal=True,
                independent_resize_repeat_equal=True,feature_range_preserved=True,depth_constant_but_plane_nonconstant=True,
                zero_init_gated_branches='residual encoder and context internal zero gradients at first step are expected; all tested branches nonzero at secondstep',
                gradients=gradients))
            del features,model,opt,batch,raw,expected,ds;torch.cuda.empty_cache()
        assert not guard.denied
    freeze(BASE/'architecture_checks.json',dict(at=now(),passed=True,results=result,audit=guard.summary(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))


def resume_checks():
    resume=[]
    for seed in SEEDS:
        for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
            target=run_protocol(seed,name);log=BASE/'checks_logs'/f'seed{seed}_{name}_stop{stop}.log';log.parent.mkdir(exist_ok=True)
            with log.open('x') as f:
                subprocess.run([sys.executable,str(HERE/'worker.py'),'--target',str(target),'--stop-after',str(stop)],
                               cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)
        a=torch.load(run_protocol(seed,'uninterrupted')/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(run_protocol(seed,'resumed')/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        for k in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):compare(a[k],b[k])
        for x,y in zip(a['metrics'],b['metrics']):
            for k in ('epoch','learning_rate','loader_rng_sha256'):compare(x[k],y[k])
            for split in ('train','val'):compare({k:v for k,v in x[split].items() if k!='seconds'},{k:v for k,v in y[split].items() if k!='seconds'})
        assert a['updates']==b['updates']==4
        resume.append(dict(seed=seed,passed=True,epochs=2,updates=4,workers=4,train_subset=4,val_subset=2,
                           integer_RNG_exact=True,float_rtol=1e-4,float_atol=1e-6,all_optimizer_scheduler_sampler_digests_checked=True))
    freeze(BASE/'engineering_checks.json',dict(at=now(),passed=True,architecture_sha256=sha(BASE/'architecture_checks.json'),resume=resume,
        training_GT64_read=False,GT128_read=False,test_metrics_read=False,formal_training_started=False,
        sources=training_identity(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))


if __name__=='__main__':
    try:architecture();resume_checks();print('ENGINEERING PASSED bothseeds CUDA objective/gradients/resume',flush=True)
    except BaseException:
        import traceback
        write(BASE/'engineering_failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
