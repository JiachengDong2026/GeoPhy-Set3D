"""Fixed paired16case test of amplitude normalization, no test/GT/parameter search."""
from feasible_context import *
from amplitude_adapter import amplitude,AmplitudeModel

INDICES=[0,1,2,7,10,31,77,82,83,100,110,126,140,150,160,165]


def main():
    setup();verify()
    for path,digest in read(ART/'amplitude_pair_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    root=ROOT/'amplitude_pair';root.mkdir(exist_ok=False);ds=make_dataset(config());guard=GTReadGuard(DATA,ROOT)
    with guard:
        values=[]
        for i in range(166):
            b=dual_measurement_collate([ds[i]])
            values.extend([float(amplitude(b[key])[0]) for key in ['A','B']])
        reference=float(np.median(values));assert 0<reference<1
        write(ART/'amplitude_reference.json',dict(reference_chi=reference,method='median train166 epoch0 A/B least-squares constant mole fraction',values=values,GT_used=False))
        teacher_rows=read(ART/'teachers.json')['cases']
        batches=[];targets=[]
        for i in INDICES:
            batches.append(ds[i]);targets.append(torch.load(teacher_rows[i]['path'],map_location='cpu',weights_only=False)['concentration'])
        def measure(model):
            model.eval();rows=[]
            with torch.no_grad():
                for index,sample,target in zip(INDICES,batches,targets):
                    batch=move_dual_batch(dual_measurement_collate([sample]),'cuda:0');t=target.to('cuda:0')
                    loss,_,fields,projections=objective(model,batch)
                    mse=.5*sum((v-t).square().mean() for v in fields)/t.square().mean()
                    scores=projection_scores(projections,batch)
                    rows.append(dict(index=index,normalized_MSE=mse.item(),physical_loss=loss.item(),
                        H=float(np.mean([v['relative_l1'] for v in scores['heldout']])),predicted_mean=float(np.mean([v.mean().item() for v in fields])),teacher_mean=t.mean().item()))
            return dict(mean_normalized_MSE=float(np.mean([r['normalized_MSE'] for r in rows])),mean_H=float(np.mean([r['H'] for r in rows])),cases=rows)
        results=[]
        order=np.random.default_rng(20261011).integers(0,len(INDICES),size=(200,2))
        for arm in ['control','amplitude']:
            base=build('unet');initial_state=state_hash(base.state_dict())
            model=(base if arm=='control' else AmplitudeModel(base,reference)).to('cuda:0')
            opt=TrackedAdam(model,root/(arm+'_updates.jsonl'));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
            initial=measure(model);started=time.monotonic()
            for step,chosen in enumerate(order,1):
                batch=move_dual_batch(dual_measurement_collate([batches[i] for i in chosen]),'cuda:0')
                target=torch.cat([targets[i] for i in chosen]).to('cuda:0');model.train();opt.zero_grad(set_to_none=True)
                a,b=model(batch['A']),model(batch['B'])
                denominator=target.square().flatten(1).mean(1).clamp_min(1e-20)
                loss=(.5*((a-target).square().flatten(1).mean(1)+(b-target).square().flatten(1).mean(1))/denominator).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if step%83==0:scheduler.step()
                if step in [50,100,200]:print('AMPLITUDE_PAIR',arm,step,'batch_MSE',loss.item(),flush=True)
            final=measure(model)
            atomic_checkpoint(root/(arm+'.pt'),dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),steps=opt.updates,reference_chi=reference))
            result=dict(arm=arm,initial=initial,final=final,updates=200,initial_base_state_sha256=initial_state,seconds=time.monotonic()-started)
            results.append(result);write(ART/('amplitude_pair_'+arm+'.json'),result)
            print('AMPLITUDE_PAIR_DONE',arm,final['mean_normalized_MSE'],final['mean_H'],flush=True)
            del model,base,opt,scheduler;torch.cuda.empty_cache()
        assert results[0]['initial_base_state_sha256']==results[1]['initial_base_state_sha256']
        write(ART/'amplitude_pair_summary.json',dict(results=results,indices=INDICES,identical_batch_order=order.tolist(),reference_chi=reference,access=guard.summary(),GT_used=False))


if __name__=='__main__':main()
