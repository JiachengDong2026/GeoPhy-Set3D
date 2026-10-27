"""Resume the exact partial checkpoint; preserve all3914 real preheat updates."""
from feasible_context import *
from train_gnot_chunked import teacher_epoch,interventions
import signal,traceback


def verify_stage():
    verify()
    for registration in ['multicase_debug_registration.json','gnot_chunk_registration.json','gnot_partial_resume_registration.json']:
        r=read(ART/registration)
        for path,digest in r['sources'].items():assert sha(PROJECT/path)==digest,path
        for path,digest in r.get('boundary',{}).items():assert sha(path)==digest,path


def main():
    setup();verify_stage();audit=read(ART/'gnot_partial_resume_checks.json');assert audit['passed']
    target=ROOT/'pilot_gnot';resumed=torch.load(target/'conflict_partial48.pt',map_location='cpu',weights_only=False)
    assert resumed['optimizer_steps']==3914 and resumed['epoch']==48
    generator=torch.Generator().manual_seed(20261011);c=config();model=build('gnot').to('cuda:0');model.query_chunk_size=65536
    model.load_state_dict(resumed['model']);opt=TrackedAdam(model,target/'preheat_updates.jsonl')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    opt.load_state_dict(resumed['optimizer']);opt.updates=3914;scheduler.load_state_dict(resumed['scheduler'])
    assert scheduler.state_dict()==resumed['scheduler'] and opt.param_groups[0]['lr']==resumed['optimizer']['param_groups'][0]['lr']
    stage='preheat';epoch=48;best=float('inf');bad_health=0;started=time.monotonic();guard=GTReadGuard(DATA,ROOT)
    initial_hash=resumed['initial_state_sha256']
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('Owned resumed GNOT interrupted')))
    def checkpoint(path):
        save_checkpoint(path,model,opt,scheduler,generator,stage=stage,epoch=epoch,config=c,
            initial_state_sha256=initial_hash,query_chunk_size=65536,best_val_H=best,volumetric_gt_used=False,
            no_optimizer_replay=True,partial_resume_protocol_sha256=sha(ART/'gnot_partial_resume_registration.json'))
    with guard:
        try:
            train,val=make_dataset(c),make_dataset(c,validation=True)
            tl=make_loader(train,c,True,generator);vl=make_loader(val,c)
            teachers={}
            for r in read(ART/'teachers.json')['cases']:
                assert sha(r['path'])==r['sha256']
                teachers[r['case_id']]=torch.load(r['path'],map_location='cpu',weights_only=False)['concentration']
            initial=read(target/'fresh_val.json');restore_rng(resumed['rng'],generator);del resumed
            # Recompute only the interrupted, UNUPDATED batch and the remaining suffix.
            # Manual collation avoids consuming another DataLoader base_seed/permutation.
            train.set_epoch(47);model.train();suffix_total=0.;suffix_cases=0;maximum=0.;began=time.monotonic()
            for indices in audit['remaining_batches']:
                raw=dual_measurement_collate([train[index] for index in indices]);batch=move_dual_batch(raw,'cuda:0')
                ids=batch['A']['case_id'];reference=torch.cat([teachers[i] for i in ids]).to('cuda:0')
                opt.zero_grad(set_to_none=True);a,b=model(batch['A']),model(batch['B'])
                scale=reference.double().square().flatten(1).mean(1).float().clamp_min(1e-20)
                error=.5*((a-reference).square().flatten(1).mean(1)+(b-reference).square().flatten(1).mean(1))
                loss=(error/scale).mean();assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);maximum=max(maximum,float(norm));opt.step()
                suffix_total+=loss.item()*len(ids);suffix_cases+=len(ids)
                append(target/'partial_resume_batches.jsonl',dict(update=opt.updates,indices=indices,case_ids=ids))
            assert opt.updates==3984 and suffix_cases==140
            suffix_seconds=time.monotonic()-began
            generator.set_state(torch.tensor(audit['loader_rng_after_epoch48'],dtype=torch.uint8))
            val.set_epoch(0);va=run_epoch(model,vl,c);scheduler.step()
            tr=dict(normalized_MSE=None,cases=166,observed_suffix_normalized_MSE=suffix_total/suffix_cases,
                observed_suffix_cases=140,unrecorded_prefix_cases=26,seconds=suffix_seconds,
                seconds_cover_full_epoch=False,gradient_norm_max=None,observed_suffix_gradient_norm_max=maximum)
            row=dict(stage=stage,epoch=48,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'],partial_resume_no_update_replay=True)
            append(target/'metrics.jsonl',row);checkpoint(target/'last_preheat.pt')
            print('RESUMED_PREHEAT48_COMPLETE updates3984 no replay; prefix loss intentionally unavailable',flush=True)
            for epoch in range(49,61):
                train.set_epoch(epoch-1);before=opt.updates;tr=teacher_epoch(model,tl,teachers,opt,c);assert opt.updates-before==83
                val.set_epoch(0);va=run_epoch(model,vl,c);scheduler.step()
                row=dict(stage=stage,epoch=epoch,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'])
                append(target/'metrics.jsonl',row);checkpoint(target/'last_preheat.pt')
                write(ART/'gnot_pilot_status.json',dict(state='running',latest=row))
                print('PREHEAT',epoch,'MSE',tr['normalized_MSE'],'valH',selection_metric(va),flush=True)
            assert opt.updates==4980
            write(target/'preheat_complete.json',dict(epochs=60,updates=4980,checkpoint_sha256=sha(target/'last_preheat.pt'),final_val=va,no_optimizer_replay=True))
            del teachers;stage='physical';epoch=0;opt=TrackedAdam(model,target/'physical_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
            physical_initial=run_epoch(model,vl,c);write(target/'physical_epoch0.json',physical_initial);checkpoint(target/'physical_epoch0.pt');checks={}
            for epoch in range(1,31):
                train.set_epoch(epoch-1);val.set_epoch(0);before=opt.updates
                tr=run_epoch(model,tl,c,opt);assert opt.updates-before==83
                va=run_epoch(model,vl,c);scheduler.step();score=selection_metric(va)
                row=dict(stage=stage,epoch=epoch,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'])
                if epoch in [1,5,10,20,29,30] or epoch%25==0:
                    row['health']=score_health(model,val,(0,1,2,3))+score_health(model,train)
                    bad_health=0 if healthy(row['health']) else bad_health+1
                    row['validation_interventions']=interventions(model,val,[0,1,2,3]);row['training_interventions']=interventions(model,train,[0,1,2])
                append(target/'metrics.jsonl',row)
                if score<best:best=score;checkpoint(target/'best.pt')
                checkpoint(target/'last.pt')
                if epoch in [29,30]:
                    vi=row['validation_interventions'];ti=row['training_interventions']
                    checks[epoch]=dict(val_H=score<=.99,improvement=score<=.95*selection_metric(initial) and selection_metric(initial)-score>=.01,
                        healthy=healthy(row['health']),val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in vi)>=2,
                        train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in ti)>=2)
                    write(target/f'gate_epoch{epoch}.json',checks[epoch])
                write(ART/'gnot_pilot_status.json',dict(state='running',latest=row,best_val_H=best))
                print('PHYSICAL',epoch,'updates',opt.updates,'trainH',selection_metric(tr),'valH',score,'bestH',best,flush=True)
                if bad_health>=2:decision='STOP_PERSISTENT_COLLAPSE';break
                if epoch==30 and not all(all(v.values()) for v in checks.values()):decision='STOP_MULTICASE_GATE_FAILED';break
            else:decision='MULTICASE_PILOT_GO'
            summary=dict(decision=decision,epochs=epoch,physical_updates=opt.updates,preheat_epochs=60,preheat_updates=4980,
                best_val_H=best,best_checkpoint=str(target/'best.pt'),best_checkpoint_sha256=sha(target/'best.pt'),
                last_checkpoint_sha256=sha(target/'last.pt'),initial_val_H=selection_metric(initial),last_val_H=selection_metric(va),
                preheat_val_H=selection_metric(physical_initial),access=guard.summary(),seconds=time.monotonic()-started,
                formal_complete=False,test_allowed=False,arm='gnot',query_chunk_size=65536,pilot_GO=decision=='MULTICASE_PILOT_GO',
                GT_used=False,foreign_conflict_preserved=True,no_optimizer_replay=True,partial_epoch48_prefix_loss_unavailable=True)
            write(target/'summary.json',summary);write(ART/'gnot_pilot_summary.json',summary);write(ART/'gnot_pilot_status.json',dict(state='complete',summary=summary))
            print('GNOT_RESUME_FINISHED',decision,flush=True)
        except BaseException as exc:
            checkpoint(target/'resume_interrupted.pt')
            write(target/'resume_failure.json',dict(stage=stage,epoch=epoch,updates=opt.updates,error=str(exc),traceback=traceback.format_exc(),access=guard.summary()))
            raise
    verify_stage()


if __name__=='__main__':main()
