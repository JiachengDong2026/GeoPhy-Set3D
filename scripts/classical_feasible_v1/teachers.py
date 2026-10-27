"""166 train-only bounded field solves; no GT/val targets, no test access."""
from feasible_context import *
from scipy.optimize import minimize


def main():
    setup();p=verify();directory=ROOT/'teachers';directory.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT);ledger=[];began=time.monotonic()
    with guard:
        ds=make_dataset(config());assert len(ds)==166
        c0=read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/initialization.json')['c0']
        for index in range(166):
            started=time.monotonic();cpu=dual_measurement_collate([ds[index]])
            batch=move_dual_batch(cpu,'cuda:0')
            double={k:{n:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for n,v in b.items()} for k,b in batch.items()}
            field=SharedField(torch.full((1,1,64,64,64),c0,device='cuda:0',dtype=torch.float64))
            evaluations=0;accepted=0;last=np.full(64**3,c0,dtype=np.float64)
            initial=float(objective(SharedField(field.value.float()),batch)[0].detach())
            def fun(x):
                nonlocal evaluations
                if evaluations>=120:raise RuntimeError('function_ceiling')
                field.value.data.copy_(torch.from_numpy(x.reshape(1,1,64,64,64)).to('cuda:0'))
                loss,*_=objective(field,double);gradient,=torch.autograd.grad(loss,field.value)
                assert torch.isfinite(loss) and torch.isfinite(gradient).all()
                evaluations+=1
                return loss.item(),gradient.detach().cpu().numpy().reshape(-1).copy()
            def callback(x):
                nonlocal accepted,last
                accepted+=1;last=x.copy()
            try:
                result=minimize(fun,last.copy(),method='L-BFGS-B',jac=True,bounds=[(0.,1.)]*(64**3),callback=callback,
                    options=dict(maxiter=60,maxfun=120,maxls=10,maxcor=5,ftol=1e-12,gtol=1e-9))
                value=result.x;status=str(result.message)
            except RuntimeError as exc:
                if str(exc)!='function_ceiling':raise
                value=last;status='hard_function_ceiling'
            teacher=torch.from_numpy(value.reshape(1,1,64,64,64)).float()
            with torch.no_grad():
                loss,_,_,projections=objective(SharedField(teacher.to('cuda:0')),batch)
                scores={k:float(np.mean([r['relative_l1'] for r in v])) for k,v in projection_scores(projections,batch).items()}
            assert teacher.square().mean()>0 and torch.isfinite(teacher).all()
            row=dict(index=index,case_id=ds.records[index]['case_id'],accepted_iterations=accepted,function_evaluations=evaluations,
                initial_loss=initial,final_loss=loss.item(),measurement=scores,solver_status=status,seconds=time.monotonic()-started,
                mean=teacher.mean().item(),std=teacher.std(correction=0).item(),GT=False,sampling_epoch=0)
            path=directory/f'{index:03d}.pt'
            atomic_checkpoint(path,dict(concentration=teacher,case_id=row['case_id'],summary=row))
            row.update(path=str(path),sha256=sha(path));ledger.append(row)
            append(directory/'ledger.jsonl',row)
            write(ART/'teacher_progress.json',dict(completed=index+1,total=166,latest=row))
            print('TEACHER',index+1,'/166 H',scores['heldout'],'cross',scores['cross'],'seconds',round(row['seconds'],2),flush=True)
        write(ART/'teachers.json',dict(cases=ledger,access=guard.summary(),seconds=time.monotonic()-began,GT_used=False))
    verify()


if __name__=='__main__':main()
