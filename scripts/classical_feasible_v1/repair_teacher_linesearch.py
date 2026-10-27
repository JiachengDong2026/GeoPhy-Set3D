"""One bounded engineering repair for the four abnormal train-only field solves."""
from feasible_context import *
from scipy.optimize import minimize


def main():
    setup();verify();registration=read(ART/'teacher_linesearch_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    original=read(ART/'teachers.json');failed=[r for r in original['cases'] if r['solver_status'].startswith('ABNORMAL')]
    assert [r['index'] for r in failed]==registration['indices']
    directory=ROOT/'teachers_linesearch_v2';directory.mkdir(exist_ok=False)
    c0=read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/initialization.json')['c0']
    guard=GTReadGuard(DATA,ROOT);repaired=[]
    with guard:
        ds=make_dataset(config());ds.set_epoch(0)
        for old in failed:
            batch=dual_measurement_collate([ds[old['index']]])
            double={key:{name:value.double() if isinstance(value,torch.Tensor) and value.is_floating_point() else value for name,value in branch.items()} for key,branch in batch.items()}
            field=SharedField(torch.full((1,1,64,64,64),c0,dtype=torch.float64))
            evaluations=0;accepted=0;last=np.full(64**3,c0,dtype=np.float64);began=time.monotonic()
            initial=float(objective(SharedField(field.value.float()),batch)[0].detach())
            np.testing.assert_allclose(initial,old['initial_loss'],rtol=3e-6,atol=1e-7)
            def fun(x):
                nonlocal evaluations
                if evaluations>=120:raise RuntimeError('function_ceiling')
                field.value.data.copy_(torch.from_numpy(x.reshape(1,1,64,64,64)))
                loss,*_=objective(field,double);gradient,=torch.autograd.grad(loss,field.value)
                assert torch.isfinite(loss) and torch.isfinite(gradient).all()
                evaluations+=1
                return loss.item(),gradient.detach().numpy().reshape(-1).copy()
            def callback(x):
                nonlocal accepted,last
                accepted+=1;last=x.copy()
            try:
                result=minimize(fun,last.copy(),method='L-BFGS-B',jac=True,bounds=[(0.,1.)]*(64**3),callback=callback,
                    options=dict(maxiter=60,maxfun=120,maxls=40,maxcor=5,ftol=1e-12,gtol=1e-9))
                value=result.x;status=str(result.message)
            except RuntimeError as exc:
                if str(exc)!='function_ceiling':raise
                value=last;status='hard_function_ceiling'
            teacher=torch.from_numpy(value.reshape(1,1,64,64,64)).float()
            with torch.no_grad():
                loss,_,_,projections=objective(SharedField(teacher),batch)
                scores={key:float(np.mean([r['relative_l1'] for r in values])) for key,values in projection_scores(projections,batch).items()}
            row=dict(index=old['index'],case_id=old['case_id'],accepted_iterations=accepted,function_evaluations=evaluations,
                initial_loss=initial,final_loss=loss.item(),measurement=scores,solver_status=status,seconds=time.monotonic()-began,
                mean=teacher.mean().item(),std=teacher.std(correction=0).item(),GT=False,sampling_epoch=0,
                original_path=old['path'],original_sha256=old['sha256'],original_solver_status=old['solver_status'])
            path=directory/f"{old['index']:03d}.pt"
            atomic_checkpoint(path,dict(concentration=teacher,case_id=row['case_id'],summary=row))
            row.update(path=str(path),sha256=sha(path));repaired.append(row)
            append(directory/'ledger.jsonl',row)
            print('LINESEARCH_REPAIR',row['index'],'iterations',accepted,'evaluations',evaluations,'H',scores['heldout'],'status',status,flush=True)
    promoted=all(not new['solver_status'].startswith('ABNORMAL') and new['measurement']['heldout']<=.5*old['measurement']['heldout'] and new['final_loss']<old['final_loss'] for old,new in zip(failed,repaired))
    write(ART/'teacher_linesearch_summary.json',dict(promoted=promoted,records=repaired,access=guard.summary(),GT_used=False,
        original_targets_unchanged=True,new_network_updates=0,diagnostic_only=not promoted))
    if promoted:
        replacements={r['index']:r for r in repaired}
        combined=[replacements.get(r['index'],r) for r in original['cases']]
        write(ART/'teachers_repaired_v2.json',dict(cases=combined,GT_used=False,
            unchanged_original_cases=162,repaired_abnormal_cases=4,original_manifest_sha256=sha(ART/'teachers.json'),
            repair_registration_sha256=sha(ART/'teacher_linesearch_registration.json')))
    verify()


if __name__=='__main__':main()
