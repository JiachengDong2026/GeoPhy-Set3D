"""Bounded original-objective continuation after self-generated-field fitting."""
from probe_context import *


def main():
    setup();verify()
    protocol=read(ART/'refinement_registration.json')
    assert sha(PROJECT/protocol['source'])==protocol['source_sha256']
    assert read(ART/'teacher_independent_review.json')['passed']
    directory=ROOT/'case0_original_refinement';directory.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT);steps=0;metrics=[];started=time.monotonic()
    with guard:
        prepared=read(ART/'prepared.json')['cases'][0]
        payload=torch.load(prepared['path'],map_location='cpu',weights_only=False)
        batch,extra=payload['batch'],payload['extra']
        model=fresh()
        source=ROOT/'case0_teacher_probe/last.pt'
        assert sha(source)==protocol['initial_checkpoint_sha256']
        ck=torch.load(source,map_location='cpu',weights_only=False)
        assert ck['steps']==200
        model.load_state_dict(ck['model']);del ck
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        initial=evaluate(model,batch,extra,directory,0,True);metrics.append(initial)
        teacher=read(ART/'case0_teacher_probe.json')['final']
        np.testing.assert_allclose(initial['loss'],teacher['loss'],rtol=1e-6,atol=1e-7)
        for step in range(1,101):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,_,_,_=objective(model,batch)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert norm>0
            previous=[p.detach().clone() for p in model.parameters()]
            optimizer.step();steps=step
            delta=sum(float((p.detach().double()-v.double()).square().sum()) for p,v in zip(model.parameters(),previous))**.5
            assert np.isfinite(delta) and delta>0
            if step%16==0:scheduler.step()
            append(directory/'updates.jsonl',dict(update=step,original_loss=float(loss.detach()),gradient_norm=float(norm),actual_parameter_delta=delta))
            if step in [25,50,100]:
                row=evaluate(model,batch,extra,directory,step,step==100);metrics.append(row)
                atomic_checkpoint(directory/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),rng=rng_state(torch.Generator().manual_seed(20261011)),steps=steps,metrics=metrics))
                print('Original objective refinement',step,'loss',row['loss'],'H',row['measurement']['heldout'],'E',row['extra_E_H'],flush=True)
        assert sha(source)==protocol['initial_checkpoint_sha256']
        final=metrics[-1]
        result=dict(index=0,actual_updates=steps,source_checkpoint=str(source),source_sha256=sha(source),initial=initial,final=final,
            metrics=metrics,access=guard.summary(),seconds=time.monotonic()-started,formal_training=False,
            fitted_ABH=all(v<=.9 for v in final['measurement'].values()) and final['loss']<initial['loss'],E_below099=final['extra_E_H']<=.99,
            checkpoint_sha256=sha(directory/'last.pt'),optimizer_reset=True,teacher_reads_in_this_stage=0)
        write(directory/'result.json',result);write(ART/'case0_original_refinement.json',result)


if __name__=='__main__':main()
