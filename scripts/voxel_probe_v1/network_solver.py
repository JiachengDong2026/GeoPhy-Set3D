"""One bounded fresh U-Net optimizer-only diagnostic, no E selection."""
from probe_context import *


class EvaluationLimit(RuntimeError):
    pass


def main():
    setup()
    verify()
    protocol=read(ART/'network_solver_registration.json')
    assert sha(PROJECT/protocol['source'])==protocol['source_sha256']
    assert read(ART/'bounded_independent_review.json')['passed']
    target=ROOT/'case0_unet_lbfgs'
    target.mkdir(exist_ok=False)
    prepared=read(ART/'prepared.json')['cases'][0]
    assert prepared['index']==0 and sha(prepared['path'])==prepared['sha256']
    guard=GTReadGuard(DATA,ROOT)
    accepted=0;closures=0;started=time.monotonic();metrics=[]
    with guard:
        payload=torch.load(prepared['path'],map_location='cpu',weights_only=False)
        batch,extra=payload['batch'],payload['extra']
        model=fresh().eval()
        initial=evaluate(model,batch,extra,target,0,True)
        metrics.append(initial)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=1,max_eval=10,history_size=5,
            tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        parameters=list(model.parameters())
        status='ITERATION_LIMIT'
        for iteration in range(1,protocol['max_outer_iterations']+1):
            previous=[p.detach().clone() for p in parameters]

            def closure():
                nonlocal closures
                if closures>=protocol['max_closures']:
                    raise EvaluationLimit('Registered original-objective closure ceiling reached')
                optimizer.zero_grad(set_to_none=True)
                loss,_,_,_=objective(model,batch)
                loss.backward()
                norm=sum(float(p.grad.detach().double().square().sum()) for p in parameters)**.5
                assert np.isfinite(norm) and norm>0
                closures+=1
                append(target/'closures.jsonl',dict(closure=closures,loss=float(loss.detach()),gradient_norm=norm))
                return loss

            try:
                optimizer.step(closure)
            except EvaluationLimit:
                # Strong-Wolfe may leave a trial point in-place on an exception.
                # Restore the last accepted state; never publish the trial.
                with torch.no_grad():
                    for p,before in zip(parameters,previous):p.copy_(before)
                status='CLOSURE_LIMIT_RESTORED_LAST_ACCEPTED'
                break
            delta=sum(float((p.detach().double()-v.double()).square().sum()) for p,v in zip(parameters,previous))**.5
            if delta==0:
                status='NO_PARAMETER_CHANGE_STOP'
                break
            assert np.isfinite(delta) and all(torch.isfinite(p).all() for p in parameters)
            accepted+=1
            append(target/'accepted_iterations.jsonl',dict(iteration=accepted,closures=closures,actual_parameter_delta=delta))
            atomic_checkpoint(target/'last_accepted.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                              accepted_iterations=accepted,closures=closures,rng=rng_state(torch.Generator().manual_seed(20261011))))
            if accepted in [5,10,20]:
                metric=evaluate(model,batch,extra,target,accepted,False)
                metrics.append(metric)
                print('U-Net LBFGS',accepted,'closures',closures,'H',metric['measurement']['heldout'],'E',metric['extra_E_H'],flush=True)
        final=evaluate(model,batch,extra,target,'final',True)
        metrics.append(final)
        output=dict(index=0,arm='unet_lbfgs',accepted_iterations=accepted,closure_evaluations=closures,status=status,
            initial=initial,final=final,metrics=metrics,seconds=time.monotonic()-started,access=guard.summary(),
            fitted_ABH=all(v<=.9 for v in final['measurement'].values()) and final['loss']<initial['loss'],
            E_below099=final['extra_E_H']<=.99,formal_training=False,
            limitation='Fresh single-case optimizer-only diagnostic; no cross-case or 3D truth result; line search accepted iterates are not Adam update equivalents')
        atomic_checkpoint(target/'final.pt',dict(model=model.state_dict(),summary=output))
        output['checkpoint_sha256']=sha(target/'final.pt')
        write(target/'result.json',output);write(ART/'case0_unet_lbfgs.json',output)
        print('U-Net bounded LBFGS finished',accepted,closures,final['measurement'],'E',final['extra_E_H'],status,flush=True)


if __name__=='__main__':main()
