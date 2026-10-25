"""Smoke and bounded pilot; original selfsup objective/epoch/selector reused."""
import argparse,os,time,math
from context import *
from diagnostics import diagnose,gates
from train_deepsets_selfsup import InputLedger
from geophy_set3d.training.utils import permute_views,environment
from geophy_set3d.data.measurement_only import DualMeasurementDataset
from torch.utils.data import Subset

class TrackedAdamW(torch.optim.AdamW):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.updates=0;self.trace=[];self.journal=None;self.failure_snapshot=None
    @torch.no_grad()
    def step(self,closure=None):
        params=[p for g in self.param_groups for p in g['params']]
        before=[p.detach().clone() for p in params]
        grads=[p.grad for p in params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        grad=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))
        if not (grad>0 and math.isfinite(grad)):
            if self.failure_snapshot is not None:self.failure_snapshot()
            raise FloatingPointError('Nonpositive or nonfinite FP32 gradient norm; preserve current state and stop')
        result=super().step(closure)
        update=float(torch.linalg.vector_norm(torch.stack([(p-b).norm() for p,b in zip(params,before)])))
        assert update>0 and math.isfinite(update) and all(torch.isfinite(p).all() for p in params)
        self.updates+=1;self.trace.append(dict(update=self.updates,postclip_gradient_norm=grad,actual_parameter_update_l2=update))
        if self.journal is not None:
            with self.journal.open('a') as stream:
                stream.write(json.dumps(self.trace[-1],allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        return result

def optimizer(model,c):return TrackedAdamW(model.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])

def smoke(arm,c):
    target=ROOT/arm/'smoke';target.mkdir(parents=True,exist_ok=False)
    setup();guard=GTReadGuard(c['dataset']['root'],ROOT/arm)
    with guard:
        model=build_model(c);initial=state_hash(model.state_dict());opt=optimizer(model,c)
        ds=DualMeasurementDataset(c['dataset']['root'],split='train',a_range=(5,5),b_range=(5,5),hold_views=2,
            base_seed=20261011,normalization=c['normalization']['method'])
        b=move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),'cuda:0')
        calls=[];handle=model.register_forward_pre_hook(lambda m,a:calls.append(id(a[0])))
        torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        loss,parts,volumes,projections=objective(model,b)
        assert calls==[id(b['A']),id(b['B'])] and id(b['H']) not in calls;handle.remove()
        for p,key in zip(volumes,['A','B']):check_prediction(p,b[key])
        loss.backward();gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        pre=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        assert opt.updates==1 and state_hash(model.state_dict())!=initial
        model.eval()
        with torch.no_grad():
            base=model(b['A']);assert torch.equal(base,model(b['A']))
            perm=model(permute_views(b['A'],torch.tensor([4,2,0,3,1],device='cuda:0')))
            err=float((base-perm).abs().max());assert torch.allclose(base,perm,atol=2e-6,rtol=2e-5)
            padded=dict(b['A'])
            for key,val in b['A'].items():
                if isinstance(val,torch.Tensor) and (key.startswith('input_') or key=='view_mask'):
                    extra=val[:,:1].clone()
                    if key=='view_mask':extra.fill_(False)
                    elif extra.is_floating_point():extra.fill_(float('nan'))
                    padded[key]=torch.cat([val,extra],1)
            pad=model(padded);paderr=float((base-pad).abs().max());assert torch.allclose(base,pad,atol=2e-6,rtol=2e-5)
            invalid=dict(b['A']);invalid['input_valid_ray']=torch.zeros_like(invalid['input_valid_ray'])
            for key in ['input_projection','input_projection_raw','input_path_length_m','input_t_near','input_ray_direction']:
                invalid[key]=torch.full_like(invalid[key],float('nan'))
            check_prediction(model(invalid),invalid)
            allpad=dict(padded);allpad['view_mask']=torch.zeros_like(padded['view_mask']);check_prediction(model(allpad),allpad)
            z=dict(b['A'])
            for key in ['input_projection','input_projection_raw']:z[key]=torch.zeros_like(z[key])
            zero=model(z);check_prediction(zero,z)
            mismatched=dict(b['A'])
            # Same recipient geometry: donor measurements gathered at recipient stations.
            donor_row,donor=ds._projections(1)
            sample=ds[0]['A'];aligned=dict(sample);aligned.update(ds._views(donor,sample['input_station_indices'].numpy(),'input'))
            donor_batch=projection_pool_collate([aligned])
            for key in ['input_projection','input_projection_raw']:
                mismatched[key]=b['A'][key].clone();mismatched[key][0]=donor_batch[key][0].to('cuda:0')
            check_prediction(model(mismatched),mismatched)
            atomic_checkpoint(target/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict()))
            saved=torch.load(target/'last.pt',map_location='cuda:0',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            assert torch.equal(base,model(b['A']))
        access=guard.summary();no_forbidden(access)
        result=dict(passed=True,arm=arm,initial_state_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
            batch=2,max_views=dict(A=5,B=5,H=2),valid_counts={k:b[k]['input_valid_ray'].flatten(1).sum(1).tolist() for k in ['A','B','H']},
            shape=list(base.shape),loss=float(loss),parts={k:float(v) for k,v in parts.items()},gradient_norms=gradients,
            preclip_norm=pre,update=opt.trace,steps=1,H_not_passed=True,finite=True,invalid_nan_safe=True,padding_max_abs=paderr,
            permutation_max_abs=err,repeat_bitwise=True,serialization_bitwise=True,zero_relative_MA=float((base-zero).abs().sum()/base.abs().sum()),
            seconds=time.perf_counter()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),access=access,environment=environment(c))
    write(target/'result.json',result);write(ART/(arm+'_smoke.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def pilot(arm,c):
    target=ROOT/arm/'pilot';target.mkdir(parents=True,exist_ok=False)
    assert read(ROOT/arm/'smoke/result.json')['passed']
    setup();guard=GTReadGuard(c['dataset']['root'],ROOT/arm);started=time.perf_counter()
    with guard:
        model=build_model(c);initial=state_hash(model.state_dict());assert initial==read(ROOT/arm/'smoke/result.json')['initial_state_sha256']
        opt=optimizer(model,c);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(20261011)
        opt.journal=target/'actual_updates.jsonl'
        def failure_snapshot():
            atomic_checkpoint(target/'failure_state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng_state(generator),optimizer_steps=opt.updates,config=c,pilot_only=True,volumetric_gt_used=False,
                source_manifest_sha256=sha(ART/'source_manifest.json'),epoch_in_progress=len(records)+1 if 'records' in locals() else None))
            write(target/'failure_access.json',guard.summary())
        opt.failure_snapshot=failure_snapshot
        train,val=make_dataset(c),make_dataset(c,validation=True);assert (len(train),len(val))==(166,36)
        subset=Subset(train,list(range(32)))
        train_loader=make_loader(subset,c,True,generator);val_loader=make_loader(val,c)
        torch.cuda.reset_peak_memory_stats()
        initial_train=run_epoch(model,make_loader(subset,c),c);initial_val=run_epoch(model,val_loader,c)
        zero=zero_volume_baseline(val_loader,c);assert abs(zero['relative_l1']-1)<1e-5
        write(target/'epoch0.json',dict(train=initial_train,val=initial_val,zero=zero,initial_state_sha256=initial,environment=environment(c)))
        initial_state={k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
        best=float('inf');records=[];diagnostic_checks={};ledger=InputLedger(model)
        for epoch in range(31):
            if epoch:
                train.set_epoch(epoch-1);val.set_epoch(0);ledger.reset();before=opt.updates;began=time.perf_counter();lr=opt.param_groups[0]['lr']
                tr=run_epoch(model,train_loader,c,opt)
                actual=ledger.rows;order=[case for r in actual if r['subset']=='A' for case in r['case_ids']]
                expected={train.records[i]['case_id']:i for i in range(32)};assert len(order)==32 and set(order)==set(expected)
                for r in actual:
                    for case,ids in zip(r['case_ids'],r['station_ids']):
                        a,b,h=train.sampler.sample(train.station_count,epoch=epoch-1,sample_index=expected[case],case_id=case)
                        assert ids==(a if r['subset']=='A' else b).tolist()
                with (target/'actual_training_inputs.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,batches=actual))+'\n')
                assert opt.updates-before==16;ledger.reset();va=run_epoch(model,val_loader,c);scheduler.step();score=selection_metric(va)
                record=dict(epoch=epoch,optimizer_steps=opt.updates,learning_rate=lr,train=tr,val=va,seconds=time.perf_counter()-began,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),state_sha256=state_hash(model.state_dict()))
                records.append(record)
                with (target/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=epoch,
                    optimizer_steps=opt.updates,config=c,initial_state_sha256=initial,source_manifest_sha256=sha(ART/'source_manifest.json'),
                    best_val_metric=min(best,score),training_signal='TDLAS measurements only',volumetric_gt_used=False,pilot_only=True)
                atomic_checkpoint(target/'last.pt',payload)
                if score<best:atomic_checkpoint(target/'best.pt',payload);best=score
                print(f'{arm} pilot {epoch}/30 steps={opt.updates} valH={score:.9f} mean={va["output"]["mean"]:.4g} seconds={record["seconds"]:.2f}',flush=True)
            else:va=initial_val
            if epoch in [0,10,20,29,30]:
                ledger.close();directory=target/f'diagnostic_epoch{epoch}';directory.mkdir()
                rows=diagnose(model,c,train,val,epoch,directory);checks=gates(rows,va,initial_val)
                diagnostic_checks[str(epoch)]=checks;write(directory/'gates.json',checks)
                ledger=InputLedger(model)
        ledger.close();assert opt.updates==480 and len(records)==30
        access=guard.summary();no_forbidden(access)
        delta=math.sqrt(sum(float((p.detach().cpu()-initial_state[k]).double().square().sum()) for k,p in model.state_dict().items()))
        engineering=all(r['actual_parameter_update_l2']>0 and r['postclip_gradient_norm']>0 for r in opt.trace)
        training_improved=records[-1]['train']['loss']['total']<initial_train['loss']['total'] and records[-1]['train']['measurement']['heldout']['relative_l1']<initial_train['measurement']['heldout']['relative_l1']
        eligible=engineering and delta>0 and training_improved and all(all(diagnostic_checks[str(i)].values()) for i in [29,30])
        decision='GO' if eligible else 'NO_GO'
        result=dict(arm=arm,status=decision,engineering=engineering,training_improved=training_improved,epochs=30,optimizer_steps=480,
            initial_val_H=selection_metric(initial_val),last_val_H=selection_metric(records[-1]['val']),best_val_H=best,
            zero_baseline=zero,final_checks=diagnostic_checks,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,
            final_state_sha256=state_hash(model.state_dict()),parameter_net_change_l2=delta,checkpoint=str(target/'last.pt'),checkpoint_sha256=sha(target/'last.pt'),
            pilot_best_sha256=sha(target/'best.pt'),peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started,access=access,
            formal_started=False,formal_updates=0,formal_eligibility='requires independent review, both final checks, and GitHub main verification' if eligible else 'blocked by failed necessary pilot gates')
        write(target/'updates.json',opt.trace);write(target/'result.json',result);write(ART/(arm+'_pilot.json'),result)
        write(target/'access.json',access);print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','pilot','formal']);ap.add_argument('arm',choices=['deeponet','fno']);a=ap.parse_args()
    verify();assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('GPU-37424437-e208-62d1-e617-53c421c94a49','GPU-cd80a78f-7d90-c291-4680-4895db9aac8b'),'Use locked launcher'
    c=load_config(a.arm)
    if a.phase=='smoke':smoke(a.arm,c)
    elif a.phase=='pilot':pilot(a.arm,c)
    else:
        from formal import train
        train(a.arm,c)
if __name__=='__main__':main()
