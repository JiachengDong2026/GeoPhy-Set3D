"""Smoke and bounded pilot; original selfsup objective/epoch/selector reused."""
import argparse,os,time,math,traceback
from unet_context import *
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
        self.updates+=1;self.trace.append(dict(update=self.updates,postclip_gradient_norm=grad,actual_parameter_update_l2=update))
        if self.journal is not None:
            with self.journal.open('a') as stream:
                stream.write(json.dumps(self.trace[-1],allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        assert update>0 and math.isfinite(update) and all(torch.isfinite(p).all() for p in params)
        return result

def optimizer(model,c):return TrackedAdamW(model.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])

def pilot(arm,c):
    target=ROOT/arm/'pilot';target.mkdir(parents=True,exist_ok=False)
    assert read(ART/(arm+'_cpu_tests.json'))['passed'] and read(ART/'primary_regression.json')['passed'] and read(ART/'boundary_tests.json')['passed']
    setup();guard=GTReadGuard(c['dataset']['root'],ROOT/arm);started=time.perf_counter()
    with guard:
        records=[];current_epoch=0;ledger=None
        model=build_model(c);initial=state_hash(model.state_dict());assert initial==read(ART/(arm+'_cpu_tests.json'))['initial_state_sha256']
        opt=optimizer(model,c);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(20261011)
        opt.journal=target/'actual_updates.jsonl'
        def failure_snapshot():
            atomic_checkpoint(target/'failure_state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng_state(generator),optimizer_steps=opt.updates,config=c,pilot_only=True,volumetric_gt_used=False,
                source_manifest_sha256=sha(ART/'source_manifest.json'),epoch_in_progress=current_epoch, failure=traceback.format_exc(), access=guard.summary(), actual_inputs=ledger.rows if ledger is not None else []))
            write(target/'failure_access.json',guard.summary())
        opt.failure_snapshot=failure_snapshot
        try:
            train,val=make_dataset(c),make_dataset(c,validation=True);assert (len(train),len(val))==(166,36)
            subset=Subset(train,list(range(32)))
            train_loader=make_loader(subset,c,True,generator);val_loader=make_loader(val,c)
            torch.cuda.reset_peak_memory_stats()
            atomic_checkpoint(target/'epoch0.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=0,optimizer_steps=0,pilot_only=True,volumetric_gt_used=False,config=c))
            write(target/'epoch0_checkpoint.json',dict(path=str(target/'epoch0.pt'),sha256=sha(target/'epoch0.pt')))
            model.train();batch=move_dual_batch(dual_measurement_collate([train[0],train[1]]),'cuda:0')
            before_memory=state_hash(model.state_dict());loss,*_=objective(model,batch);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            model.zero_grad(set_to_none=True);assert state_hash(model.state_dict())==before_memory
            write(target/'gpu_memory_check.json',dict(passed=True,batch=2,grid=[64,64,64],optimizer_updates=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),state_hash=before_memory))
            del batch,loss,_
            initial_train=run_epoch(model,make_loader(subset,c),c);initial_val=run_epoch(model,val_loader,c)
            zero_train=zero_volume_baseline(make_loader(subset,c),c)
            zero=zero_volume_baseline(val_loader,c);assert abs(zero['relative_l1']-1)<1e-5
            write(target/'epoch0.json',dict(train=initial_train,val=initial_val,zero=zero,zero_train=zero_train,initial_state_sha256=initial,environment=environment(c)))
            initial_state={k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
            best=float('inf');records=[];diagnostic_checks={};ledger=InputLedger(model)
            for epoch in range(31):
                current_epoch=epoch
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
                    atomic_checkpoint(target/f'epoch{epoch}.pt',payload)
                    write(target/f'epoch{epoch}_checkpoint.json',dict(path=str(target/f'epoch{epoch}.pt'),sha256=sha(target/f'epoch{epoch}.pt')))
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
        except BaseException:
            failure_snapshot()
            raise

def main():
    import signal
    def stop(signum,frame):raise RuntimeError('Owned worker interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['unet']);a=ap.parse_args()
    verify();assert os.environ.get('GEOPHY_UNET_UUID_LOCKED')==os.environ.get('CUDA_VISIBLE_DEVICES'),'Use UUID-locked launcher'
    pilot(a.arm,load_config(a.arm))
if __name__=='__main__':main()
