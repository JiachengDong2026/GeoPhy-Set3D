"""One fixed teacher-then-physics diagnostic for each prior failed architecture."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/unet_continuation_v1'))
from evaluation import evaluate


def main():
    setup();verify()
    for path,digest in read(ART/'debug_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    assert read(ART/'cpu_checks.json')['passed']
    root=ROOT/'debug';root.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT.parent)
    with guard:
        payload=torch.load('/scratch/djc/GeoPhy-Set3D/voxel_probe_v1/prepared/case0.pt',map_location='cpu',weights_only=False)
        batch=move_dual_batch(payload['batch'],'cuda:0');extra=move_batch(payload['extra'],'cuda:0')
        teacher_path=ROOT/'teachers/000.pt'
        teacher=torch.load(teacher_path,map_location='cpu',weights_only=False)['concentration'].to('cuda:0')
        denominator=teacher.square().mean().clamp_min(1e-20)
        summaries=[]
        for arm in ['deeponet','fno','senseiver','gnot','deepsets']:
            directory=root/arm;directory.mkdir(exist_ok=False);phase='dense';step=0;started=time.monotonic()
            model=build(arm).to('cuda:0');initial_hash=state_hash(model.state_dict());generator=torch.Generator().manual_seed(20261011)
            opt=TrackedAdam(model,directory/'dense_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
            records=[]
            def measure(label):
                row=evaluate(model,batch,extra,directory,label,True)
                with torch.no_grad():
                    a,b=model(batch['A']),model(batch['B'])
                    row['teacher_normalized_MSE']=(.5*((a-teacher).square().mean()+(b-teacher).square().mean())/denominator).item()
                records.append(row);append(directory/'metrics.jsonl',row);return row
            try:
                initial=measure('fresh')
                for phase in ['dense','physical']:
                    if phase=='physical':
                        opt=TrackedAdam(model,directory/'physical_updates.jsonl');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
                    for step in range(1,201):
                        model.train();opt.zero_grad(set_to_none=True)
                        if phase=='dense':
                            a,b=model(batch['A']),model(batch['B']);loss=.5*((a-teacher).square().mean()+(b-teacher).square().mean())/denominator
                        else:loss,*_=objective(model,batch)
                        assert torch.isfinite(loss)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                        if step%16==0:scheduler.step()
                        if step in [50,100,200]:
                            row=measure(phase+str(step))
                            save_checkpoint(directory/'last.pt',model,opt,scheduler,generator,arm=arm,phase=phase,phase_updates=step,volumetric_gt_used=False)
                            print('DEBUG',arm,phase,step,'loss',row['loss'],'teacherMSE',row['teacher_normalized_MSE'],'H',row['measurement']['heldout'],'E',row['extra_E_H'],flush=True)
                    if phase=='dense':dense_final=row
                final=records[-1]
                result=dict(arm=arm,teacher_updates=200,physical_updates=200,initial=initial,dense_final=dense_final,final=final,metrics=records,
                    dense_MSE_half=dense_final['teacher_normalized_MSE']<=.5*initial['teacher_normalized_MSE'],
                    fitted_ABH=all(v<=.9 for v in final['measurement'].values()),E_below099=final['extra_E_H']<=.99,
                    teacher_sha256=sha(teacher_path),initial_state_sha256=initial_hash,checkpoint_sha256=sha(directory/'last.pt'),
                    seconds=time.monotonic()-started,status='bounded_complete',access=guard.summary(),formal_training=False)
            except BaseException as exc:
                save_checkpoint(directory/'interrupted.pt',model,opt,scheduler,generator,arm=arm,phase=phase,phase_updates=step)
                import traceback
                result=dict(arm=arm,status='engineering_failure',phase=phase,actual_phase_updates=opt.updates,error=str(exc),traceback=traceback.format_exc(),access=guard.summary())
                write(directory/'result.json',result);write(ART/f'debug_{arm}.json',result)
                # Numerical/engineering failure must be inspected before another architecture is launched.
                raise
            write(directory/'result.json',result);write(ART/f'debug_{arm}.json',result);summaries.append(result)
            del model,opt,scheduler
            torch.cuda.empty_cache()
        write(ART/'debug_summary.json',dict(models=summaries,access=guard.summary(),protocol='single fixed case, dense200 plus original200 per architecture'))
    verify()


if __name__=='__main__':main()
