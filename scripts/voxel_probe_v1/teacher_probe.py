"""Fresh U-Net fit to a self-generated feasible field, never volumetric GT."""
from probe_context import *


def main():
    setup();verify()
    protocol=read(ART/'teacher_probe_registration.json')
    assert sha(PROJECT/protocol['source'])==protocol['source_sha256']
    assert read(ART/'bounded_independent_review.json')['passed']
    target=ROOT/'case0_teacher_probe';target.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT)
    metrics=[];steps=0;started=time.monotonic()
    with guard:
        prepared=read(ART/'prepared.json')['cases'][0]
        payload=torch.load(prepared['path'],map_location='cpu',weights_only=False)
        batch,extra=payload['batch'],payload['extra']
        solver=read(ART/'case0_bounded_solver.json')
        path=solver['final']['arrays']['path'];assert sha(path)==protocol['teacher_array_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            teacher=torch.from_numpy(arrays['A_volume'].copy())
            assert np.array_equal(arrays['A_volume'],arrays['B_volume'])
        denominator=teacher.double().square().mean().float()
        assert denominator>0
        model=fresh()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)

        @torch.no_grad()
        def measure(point,save):
            row=evaluate(model,batch,extra,target,point,save)
            a,b=model(batch['A']),model(batch['B'])
            row['teacher_normalized_MSE']=float(.5*((a-teacher).square().mean()+(b-teacher).square().mean())/denominator)
            row['teacher_relative_L1']=float(.5*((a-teacher).abs().sum()+(b-teacher).abs().sum())/teacher.abs().sum())
            return row

        initial=measure(0,True);metrics.append(initial)
        for step in range(1,protocol['updates']+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            a,b=model(batch['A']),model(batch['B'])
            loss=.5*((a-teacher).square().mean()+(b-teacher).square().mean())/denominator
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert norm>0
            previous=[p.detach().clone() for p in model.parameters()]
            optimizer.step();steps=step
            delta=sum(float((p.detach().double()-v.double()).square().sum()) for p,v in zip(model.parameters(),previous))**.5
            assert delta>0 and np.isfinite(delta)
            if step%16==0:scheduler.step()
            append(target/'updates.jsonl',dict(update=step,normalized_MSE=float(loss.detach()),gradient_norm=float(norm),actual_parameter_delta=delta))
            if step in [50,100,200]:
                row=measure(step,step==200);metrics.append(row)
                atomic_checkpoint(target/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),rng=rng_state(torch.Generator().manual_seed(20261011)),steps=steps,metrics=metrics))
                print('Teacher self-generated field',step,'MSE',row['teacher_normalized_MSE'],'L1',row['teacher_relative_L1'],
                      'H',row['measurement']['heldout'],'E',row['extra_E_H'],flush=True)
        result=dict(index=0,actual_updates=steps,teacher_is_GT=False,teacher_source=path,teacher_sha256=sha(path),
            initial=initial,final=metrics[-1],metrics=metrics,access=guard.summary(),seconds=time.monotonic()-started,
            normalized_MSE_below_half_initial=metrics[-1]['teacher_normalized_MSE']<=.5*initial['teacher_normalized_MSE'],
            teacher_relative_L1_below090=metrics[-1]['teacher_relative_L1']<=.9,
            fitted_ABH=all(v<=.9 for v in metrics[-1]['measurement'].values()),E_below099=metrics[-1]['extra_E_H']<=.99,
            checkpoint_sha256=sha(target/'last.pt'),formal_training=False)
        assert sha(path)==protocol['teacher_array_sha256']
        write(target/'result.json',result);write(ART/'case0_teacher_probe.json',result)


if __name__=='__main__':main()
