"""Multi-case bounded architecture check: dense train-only preheat then original physical engine."""
from feasible_context import *
import signal,traceback


def verify_stage():
    verify()
    for registration in ['multicase_debug_registration.json','gnot_chunk_registration.json']:
        for path,digest in read(ART/registration)['sources'].items():assert sha(PROJECT/path)==digest,path


def teacher_epoch(model,loader,teachers,opt,c):
    model.train();total=0.;count=0;started=time.monotonic();max_norm=0.
    for raw in loader:
        batch=move_dual_batch(raw,'cuda:0');ids=batch['A']['case_id']
        target=torch.cat([teachers[i] for i in ids]).to('cuda:0')
        opt.zero_grad(set_to_none=True)
        a,b=model(batch['A']),model(batch['B'])
        scale=target.double().square().flatten(1).mean(1).float().clamp_min(1e-20)
        error=.5*((a-target).square().flatten(1).mean(1)+(b-target).square().flatten(1).mean(1))
        loss=(error/scale).mean();assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        max_norm=max(max_norm,float(norm));opt.step()
        total+=loss.item()*len(ids);count+=len(ids)
    return dict(normalized_MSE=total/count,cases=count,seconds=time.monotonic()-started,gradient_norm_max=max_norm)


@torch.no_grad()
def interventions(model,ds,indices):
    old=int(ds._epoch.item());ds.set_epoch(0);rows=[];model.eval()
    for index in indices:
        sample=ds[index];batch=move_dual_batch(dual_measurement_collate([sample]),'cuda:0')
        donor,raw=ds._projections(index+1)
        h=batch['H'];mask=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
        values=[]
        for branch in ['A','B']:
            normal=model(batch[branch]);zero=dict(batch[branch]);wrong=dict(batch[branch])
            aligned=dict(sample[branch]);aligned.update(ds._views(raw,sample[branch]['input_station_indices'].numpy(),'input'))
            donor_batch=move_batch(projection_pool_collate([aligned]),'cuda:0')
            for name in ['input_projection_raw','input_projection']:
                zero[name]=torch.zeros_like(zero[name]);wrong[name]=donor_batch[name]
            pz,pw=model(zero),model(wrong)
            metrics=[relative_projection_metrics(project_input(v,h),h['input_projection_raw'],mask)['relative_l1'] for v in [normal,pz,pw]]
            values.append(dict(normal_H=metrics[0],zero_benefit=metrics[1]-metrics[0],wrong_increase=metrics[2]-metrics[0],
                response_relative=(normal-pz).abs().sum().item()/max(normal.abs().sum().item(),1e-30),response_absolute=(normal-pz).abs().mean().item()))
        rows.append(dict(index=index,case_id=ds.records[index]['case_id'],**{k:float(np.mean([v[k] for v in values])) for k in values[0]}))
    ds.set_epoch(old);return rows


def main():
    setup();verify_stage()
    arm=sys.argv[1];assert arm=='gnot'
    assert read(ART/'gnot_chunk_checks.json')['passed']
    assert 'passed' in (ROOT/'gnot_chunk_tests.log').read_text()
    assert next(v for v in read(ART/'debug_independent_review.json')['models'] if v['arm']==arm)['local_fit_passed']
    assert read(ART/'cpu_checks.json')['passed']
    target=ROOT/('pilot_'+arm);target.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT);generator=torch.Generator().manual_seed(20261011)
    c=config();model=build(arm).to('cuda:0');initial_hash=state_hash(model.state_dict())
    model.query_chunk_size=65536
    assert state_hash(model.state_dict())==initial_hash
    torch.cuda.reset_peak_memory_stats()
    opt=TrackedAdam(model,target/'preheat_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    stage='preheat';epoch=0;records=[];best=float('inf');bad_health=0;decision='running';started=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('Owned worker interrupted')))
    def checkpoint(path):
        save_checkpoint(path,model,opt,scheduler,generator,stage=stage,epoch=epoch,config=c,
            initial_state_sha256=initial_hash,query_chunk_size=65536,execution_supplement_sha256=sha(ART/'gnot_chunk_registration.json'),protocol_sha256=sha(ART/'multicase_debug_registration.json'),best_val_H=best,
            volumetric_gt_used=False,teacher_preheat_epochs=60,teacher_preheat_updates=4980)
    with guard:
        try:
            train,val=make_dataset(c),make_dataset(c,validation=True);assert (len(train),len(val))==(166,36)
            tl=make_loader(train,c,True,generator);vl=make_loader(val,c)
            teachers={}
            for r in read(ART/'teachers.json')['cases']:
                assert sha(r['path'])==r['sha256'];payload=torch.load(r['path'],map_location='cpu',weights_only=False)
                teachers[r['case_id']]=payload['concentration']
            assert set(teachers)=={r['case_id'] for r in train.records}
            initial=run_epoch(model,vl,c);write(target/'fresh_val.json',initial);checkpoint(target/'fresh.pt')
            print('FRESH fullVal H',selection_metric(initial),flush=True)
            for epoch in range(1,61):
                train.set_epoch(epoch-1);before=opt.updates
                tr=teacher_epoch(model,tl,teachers,opt,c);assert opt.updates-before==83
                val.set_epoch(0);va=run_epoch(model,vl,c);scheduler.step()
                row=dict(stage=stage,epoch=epoch,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'])
                append(target/'metrics.jsonl',row);records.append(row);checkpoint(target/'last_preheat.pt')
                write(ART/(arm+'_pilot_status.json'),dict(state='running',latest=row))
                print('PREHEAT',epoch,'MSE',tr['normalized_MSE'],'valH',selection_metric(va),flush=True)
            assert opt.updates==4980
            write(target/'preheat_complete.json',dict(epochs=60,updates=opt.updates,checkpoint_sha256=sha(target/'last_preheat.pt'),final_val=va))
            del teachers
            stage='physical';epoch=0;opt=TrackedAdam(model,target/'physical_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
            physical_initial=run_epoch(model,vl,c);write(target/'physical_epoch0.json',physical_initial);checkpoint(target/'physical_epoch0.pt')
            checks={}
            for epoch in range(1,31):
                train.set_epoch(epoch-1);val.set_epoch(0);before=opt.updates
                tr=run_epoch(model,tl,c,opt);assert opt.updates-before==83
                va=run_epoch(model,vl,c);scheduler.step();score=selection_metric(va)
                row=dict(stage=stage,epoch=epoch,updates=opt.updates,train=tr,val=va,learning_rate=opt.param_groups[0]['lr'])
                if epoch in [1,5,10,20,29,30] or epoch%25==0:
                    health=score_health(model,val,(0,1,2,3))+score_health(model,train)
                    row['health']=health
                    bad_health=0 if healthy(health) else bad_health+1
                    row['validation_interventions']=interventions(model,val,[0,1,2,3])
                    row['training_interventions']=interventions(model,train,[0,1,2])
                append(target/'metrics.jsonl',row);records.append(row)
                if score<best:best=score;checkpoint(target/'best.pt')
                checkpoint(target/'last.pt')
                if epoch in [29,30]:
                    vi=row['validation_interventions'];ti=row['training_interventions']
                    checks[epoch]=dict(val_H=score<=.99,improvement=score<=.95*selection_metric(initial) and selection_metric(initial)-score>=.01,
                        healthy=healthy(row['health']),val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in vi)>=2,
                        train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in ti)>=2)
                    write(target/f'gate_epoch{epoch}.json',checks[epoch])
                write(ART/(arm+'_pilot_status.json'),dict(state='running',latest=row,best_val_H=best))
                print('PHYSICAL',epoch,'updates',opt.updates,'trainH',selection_metric(tr),'valH',score,'bestH',best,flush=True)
                if bad_health>=2:decision='STOP_PERSISTENT_COLLAPSE';break
                if epoch==30 and not all(all(v.values()) for v in checks.values()):decision='STOP_MULTICASE_GATE_FAILED';break
            else:decision='MULTICASE_PILOT_GO'
            summary=dict(decision=decision,epochs=epoch,physical_updates=opt.updates,preheat_epochs=60,preheat_updates=4980,
                best_val_H=best,best_checkpoint=str(target/'best.pt'),best_checkpoint_sha256=sha(target/'best.pt'),
                last_checkpoint_sha256=sha(target/'last.pt'),initial_val_H=selection_metric(initial),last_val_H=selection_metric(va),
                preheat_val_H=selection_metric(physical_initial),access=guard.summary(),seconds=time.monotonic()-started,
                formal_complete=False,test_allowed=False,arm=arm,query_chunk_size=65536,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),pilot_GO=decision=='MULTICASE_PILOT_GO',GT_used=False)
            write(target/'summary.json',summary);write(ART/(arm+'_pilot_summary.json'),summary)
            write(ART/(arm+'_pilot_status.json'),dict(state='complete',summary=summary))
        except BaseException as exc:
            checkpoint(target/'interrupted.pt')
            write(target/'failure.json',dict(stage=stage,epoch=epoch,updates=opt.updates,error=str(exc),traceback=traceback.format_exc(),access=guard.summary()))
            raise
    verify_stage()
    print(arm+'_PILOT_FINISHED',decision,flush=True)


if __name__=='__main__':main()
