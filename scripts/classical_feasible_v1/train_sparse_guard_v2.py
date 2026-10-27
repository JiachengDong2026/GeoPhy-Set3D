"""Exact continuation from original175 boundary under one separately registered guard."""
from sparse_guard_v2_context import *
from prepared_inputs import PreparedDataset,prepare_model
from train_unet_workers import interventions
import shutil,signal,traceback


def main():
    setup();verify();assert read(ART/'cpu_checks.json')['passed']
    source=BASE_ROOT/'unet';target=ROOT/'unet';target.mkdir(exist_ok=False)
    original=read(BASE_ART/'unet_summary.json');assert original['decision']=='STOP_PERSISTENT_COLLAPSE' and original['epochs']==175
    for name in ['metrics.jsonl','preheat_updates.jsonl','physical_updates.jsonl','preheat_complete.json','fresh_val.json','physical_epoch0.json']:
        shutil.copy2(source/name,target/name)
    os.link(source/'last.pt',target/'boundary175.pt');os.link(source/'best.pt',target/'best.pt')
    resumed=torch.load(target/'boundary175.pt',map_location='cpu',weights_only=False)
    assert resumed['epoch']==175 and resumed['optimizer_steps']==14525 and resumed['scheduler']['last_epoch']==175
    assert {int(v['step']) for v in resumed['optimizer']['state'].values()}=={14525}
    rows=[json.loads(line) for line in (target/'physical_updates.jsonl').read_text().splitlines()]
    assert [v['update'] for v in rows]==list(range(1,14526))
    generator=torch.Generator().manual_seed(20261011);c=config();c['loader'].update(num_workers=4,persistent_workers=False)
    model=prepare_model(build('unet')).to('cuda:0');model.load_state_dict(resumed['model'])
    opt=TrackedAdam(model,target/'physical_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
    opt.load_state_dict(resumed['optimizer']);opt.updates=14525;scheduler.load_state_dict(resumed['scheduler'])
    assert scheduler.state_dict()==resumed['scheduler']
    assert opt.param_groups[0]['lr']==resumed['optimizer']['param_groups'][0]['lr']
    best=original['best_val_H'];epoch=175;bad_health=0;decision='running';started=time.monotonic()
    initial_hash=resumed['initial_state_sha256']
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('Owned sparse guard worker interrupted')))
    def checkpoint(path):
        save_checkpoint(path,model,opt,scheduler,generator,stage='physical',epoch=epoch,config=c,
            initial_state_sha256=initial_hash,best_val_H=best,volumetric_gt_used=False,
            original_decision_preserved=True,guard_version='sparse_guard_v2',protocol_sha256=sha(ART/'registration.json'))
    guard=GTReadGuard(DATA,ROOT)
    with guard:
        try:
            train,val=PreparedDataset(make_dataset(c)),PreparedDataset(make_dataset(c,validation=True))
            tl=make_loader(train,c,True,generator);vl=make_loader(val,c)
            restore_rng(resumed['rng'],generator);del resumed
            print('RESUME_PHYSICAL175_EXACT optimizer14525 scheduler175 unchanged RNG',flush=True)
            for epoch in range(176,301):
                train.set_epoch(epoch-1);val.set_epoch(0);before=opt.updates
                tr=run_epoch(model,tl,c,opt);assert opt.updates-before==83
                va=run_epoch(model,vl,c);scheduler.step();score=selection_metric(va)
                row=dict(stage='physical',epoch=epoch,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'],guard_version='sparse_guard_v2')
                if epoch%25==0 or epoch==176:
                    row['health']=score_health(model,val,(0,1,2,3))+score_health(model,train)
                    bad_health=0 if healthy(row['health']) else bad_health+1
                    row['validation_interventions']=interventions(model,val,[0,1,2,3])
                    row['training_interventions']=interventions(model,train,[0,1,2])
                append(target/'metrics.jsonl',row)
                if score<best:best=score;checkpoint(target/'best.pt')
                checkpoint(target/'last.pt')
                write(ART/'unet_status.json',dict(state='running',latest=row,best_val_H=best))
                print('PHYSICAL',epoch,'updates',opt.updates,'trainH',selection_metric(tr),'valH',score,'bestH',best,flush=True)
                if bad_health>=2:decision='STOP_REVISED_GUARD_FAILED';break
            else:decision='FORMAL_300_COMPLETE_SPARSE_GUARD_V2'
            summary=dict(decision=decision,epochs=epoch,physical_updates=opt.updates,preheat_epochs=60,preheat_updates=4980,
                best_val_H=best,best_checkpoint=str(target/'best.pt'),best_checkpoint_sha256=sha(target/'best.pt'),
                last_checkpoint_sha256=sha(target/'last.pt'),initial_val_H=original['initial_val_H'],last_val_H=selection_metric(va),
                preheat_val_H=original['preheat_val_H'],access=guard.summary(),seconds=time.monotonic()-started,
                formal_complete=decision=='FORMAL_300_COMPLETE_SPARSE_GUARD_V2',test_allowed=decision=='FORMAL_300_COMPLETE_SPARSE_GUARD_V2',
                GT_used=False,original_decision=original['decision'],original_epochs=175,new_physical_updates=opt.updates-14525,
                guard_revision_disclosed=True,model_loss_optimizer_data_unchanged=True)
            write(target/'summary.json',summary);write(ART/'unet_summary.json',summary)
            write(ART/'unet_status.json',dict(state='complete',summary=summary))
        except BaseException as exc:
            checkpoint(target/'interrupted.pt')
            write(target/'failure.json',dict(epoch=epoch,updates=opt.updates,error=str(exc),traceback=traceback.format_exc(),access=guard.summary()))
            raise
    verify()


if __name__=='__main__':main()
