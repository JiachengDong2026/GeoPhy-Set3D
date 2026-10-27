"""One fixed final GNOT epoch after separating background from signal saturation."""
from sparse_guard_v2_context import *
from train_multicase_debug import interventions
import shutil,traceback,signal


def verify_stage():
    verify()
    registration=read(ART/'gnot_continuation_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    for path,digest in registration['boundary'].items():assert sha(path)==digest,path


def gate(row,initial):
    vi=row['validation_interventions'];ti=row['training_interventions'];score=selection_metric(row['val'])
    return dict(val_H=score<=.99,improvement=score<=.95*initial and initial-score>=.01,
        healthy=healthy(row['health']),val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in vi)>=2,
        train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in ti)>=2)


def main():
    setup();verify_stage();arm='gnot';source=BASE_ROOT/'pilot_gnot';target=ROOT/'pilot_gnot';target.mkdir(exist_ok=False)
    original=read(BASE_ART/'gnot_pilot_summary.json');assert original['epochs']==29 and original['decision']=='STOP_PERSISTENT_COLLAPSE'
    assert read(BASE_ART/'gnot_multicase_review.json')['passed']
    for name in ['metrics.jsonl','preheat_updates.jsonl','physical_updates.jsonl','preheat_complete.json']:
        shutil.copy2(source/name,target/name)
    os.link(source/'last.pt',target/'boundary29.pt');os.link(source/'best.pt',target/'best.pt')
    resumed=torch.load(target/'boundary29.pt',map_location='cpu',weights_only=False)
    assert resumed['epoch']==29 and resumed['optimizer_steps']==2407 and resumed['scheduler']['last_epoch']==29
    assert {int(v['step']) for v in resumed['optimizer']['state'].values()}=={2407}
    generator=torch.Generator().manual_seed(20261011);c=config();model=build(arm).to('cuda:0');model.query_chunk_size=65536;model.load_state_dict(resumed['model'])
    opt=TrackedAdam(model,target/'physical_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
    opt.load_state_dict(resumed['optimizer']);opt.updates=2407;scheduler.load_state_dict(resumed['scheduler'])
    best=original['best_val_H'];epoch=29;started=time.monotonic();guard=GTReadGuard(DATA,ROOT)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('Owned GNOT v2 worker interrupted')))
    def checkpoint(path):
        save_checkpoint(path,model,opt,scheduler,generator,stage='physical',epoch=epoch,config=c,
            initial_state_sha256=resumed['initial_state_sha256'],best_val_H=best,guard_version='sparse_guard_v2',
            original_decision_preserved=True,protocol_sha256=sha(ART/'gnot_continuation_registration.json'))
    with guard:
        try:
            train,val=make_dataset(c),make_dataset(c,validation=True);tl=make_loader(train,c,True,generator);vl=make_loader(val,c)
            row29=[json.loads(s) for s in (target/'metrics.jsonl').read_text().splitlines()][-1]
            row29=copy.deepcopy(row29);row29['legacy_health']=row29['health']
            row29['health']=score_health(model,val,(0,1,2,3))+score_health(model,train)
            checks={29:gate(row29,original['initial_val_H'])};write(target/'revised_epoch29.json',row29);write(target/'gate_epoch29.json',checks[29])
            assert all(checks[29].values()),'Revised29 gate fails; no new training permitted'
            restore_rng(resumed['rng'],generator)
            epoch=30;train.set_epoch(29);val.set_epoch(0)
            tr=run_epoch(model,tl,c,opt);assert opt.updates==2490
            va=run_epoch(model,vl,c);scheduler.step();score=selection_metric(va)
            row=dict(stage='physical',epoch=30,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'],guard_version='sparse_guard_v2')
            row['health']=score_health(model,val,(0,1,2,3))+score_health(model,train)
            row['validation_interventions']=interventions(model,val,[0,1,2,3]);row['training_interventions']=interventions(model,train,[0,1,2])
            append(target/'metrics.jsonl',row);checks[30]=gate(row,original['initial_val_H']);write(target/'gate_epoch30.json',checks[30])
            if score<best:best=score;checkpoint(target/'best.pt')
            checkpoint(target/'last.pt')
            decision='MULTICASE_PILOT_GO_SPARSE_GUARD_V2' if all(all(v.values()) for v in checks.values()) else 'STOP_REVISED_MULTICASE_GATE_FAILED'
            summary=dict(decision=decision,epochs=30,physical_updates=2490,new_physical_updates=83,preheat_epochs=60,preheat_updates=4980,
                best_val_H=best,best_checkpoint=str(target/'best.pt'),best_checkpoint_sha256=sha(target/'best.pt'),
                last_checkpoint_sha256=sha(target/'last.pt'),initial_val_H=original['initial_val_H'],last_val_H=score,
                preheat_val_H=original['preheat_val_H'],access=guard.summary(),seconds=time.monotonic()-started,
                formal_complete=False,test_allowed=False,arm=arm,pilot_GO=decision=='MULTICASE_PILOT_GO_SPARSE_GUARD_V2',
                GT_used=False,original_decision=original['decision'],original_epochs=29,guard_revision_disclosed=True)
            write(target/'summary.json',summary);write(ART/'gnot_pilot_summary.json',summary)
            write(ART/'gnot_pilot_status.json',dict(state='complete',summary=summary))
            print('GNOT_SPARSE_GUARD_V2',decision,'ValH',score,flush=True)
        except BaseException as exc:
            checkpoint(target/'interrupted.pt')
            write(target/'failure.json',dict(epoch=epoch,updates=opt.updates,error=str(exc),traceback=traceback.format_exc(),access=guard.summary()))
            raise
    verify_stage()


if __name__=='__main__':main()
