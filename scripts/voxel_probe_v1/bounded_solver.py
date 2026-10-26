"""One separately registered bounded L-BFGS-B existence probe, no network fit."""
from probe_context import *
from scipy.optimize import minimize


class SharedConcentration(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(value.clone())

    def forward(self, batch):
        return self.value


def main():
    setup()
    registration = verify()
    protocol = read(ART / 'bounded_solver_registration.json')
    assert sha(PROJECT / protocol['source']) == protocol['source_sha256']
    index = int(sys.argv[1])
    assert index in protocol['indices']
    target = ROOT / f'case{index}_bounded_solver'
    target.mkdir(exist_ok=False)
    prepared = next(r for r in read(ART / 'prepared.json')['cases'] if r['index'] == index)
    evaluations = 0
    accepted = 0
    started = time.monotonic()
    guard = GTReadGuard(DATA, ROOT)
    with guard:
        assert sha(prepared['path']) == prepared['sha256']
        payload = torch.load(prepared['path'], map_location='cpu', weights_only=False)
        batch32, extra32 = payload['batch'], payload['extra']
        batch = {k: {name: v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v
                     for name,v in b.items()} for k,b in batch32.items()}
        initial = payload['initial_shared'].double()
        original = SharedConcentration(initial)
        # Evaluate FP32 initial through the same unchanged physical objective.
        initial_metrics = evaluate(SharedConcentration(initial.float()), batch32, extra32, target, 0, True)

        def fun(x):
            nonlocal evaluations
            if evaluations >= protocol['max_function_evaluations']:
                raise RuntimeError('Registered hard function-evaluation ceiling reached')
            original.value.data.copy_(torch.from_numpy(x.reshape(1,1,64,64,64)))
            loss, _, _, _ = objective(original,batch)
            gradient, = torch.autograd.grad(loss,original.value)
            assert torch.isfinite(loss) and torch.isfinite(gradient).all()
            evaluations += 1
            append(target/'function_evaluations.jsonl',dict(evaluation=evaluations,loss=float(loss.detach()),
                gradient_norm=float(gradient.norm()),minimum=float(x.min()),maximum=float(x.max())))
            return float(loss.detach()), gradient.detach().numpy().reshape(-1).copy()

        last_accepted = initial.numpy().reshape(-1).copy()

        def callback(x):
            nonlocal accepted,last_accepted
            accepted += 1
            delta = float(np.linalg.norm(x-last_accepted))
            append(target/'accepted_iterations.jsonl',dict(iteration=accepted,function_evaluations=evaluations,delta_l2=delta))
            last_accepted = x.copy()
            atomic_checkpoint(target/'last_accepted.pt',dict(concentration=torch.from_numpy(x.reshape(1,1,64,64,64)).clone(),
                              accepted_iterations=accepted,function_evaluations=evaluations))

        initial_loss = None
        try:
            result = minimize(fun,initial.numpy().reshape(-1).copy(),method='L-BFGS-B',jac=True,
                bounds=[(0.,1.)]*(64**3),callback=callback,
                options=dict(maxiter=protocol['max_iterations'],maxfun=protocol['max_function_evaluations'],
                             maxls=10,maxcor=5,ftol=1e-12,gtol=1e-9))
            final_x = result.x
            solver_status = dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                                 nfev=int(result.nfev),reported_loss=float(result.fun))
        except RuntimeError as error:
            if 'Registered hard function-evaluation ceiling' not in str(error):
                raise
            final_x = last_accepted
            solver_status = dict(success=False,message=str(error),iterations=accepted,nfev=evaluations,
                                 hard_ceiling=True)
        # Final judgement uses original FP32 projector and unchanged objective,
        # avoiding dependence on the FP64 solver's geometry rounding.
        concentration = torch.from_numpy(final_x.reshape(1,1,64,64,64)).float()
        model32 = SharedConcentration(concentration)
        final_metrics = evaluate(model32,batch32,extra32,target,"final",True)
        output = dict(index=index,arm='bounded_solver',optimizer='SciPy bounded L-BFGS-B, original objective in FP64',
            accepted_iterations=accepted,function_evaluations=evaluations,solver_status=solver_status,
            initial=initial_metrics,final=final_metrics,seconds=time.monotonic()-started,
            fitted_ABH=all(v<=.9 for v in final_metrics['measurement'].values()) and final_metrics['loss']<initial_metrics['loss'],
            E_below099=final_metrics['extra_E_H']<=.99,access=guard.summary(),
            formal_training=False,network_updates=0,FP32_final_judgement=True)
        atomic_checkpoint(target/'final_field.pt',dict(concentration=concentration,summary=output))
        output['final_checkpoint_sha256']=sha(target/'final_field.pt')
        write(target/'result.json',output)
        write(ART/f'case{index}_bounded_solver.json',output)
        print('Bounded solver',index,'iterations',accepted,'evaluations',evaluations,'final',final_metrics['measurement'],
              'E',final_metrics['extra_E_H'],'status',solver_status,flush=True)


if __name__=='__main__':
    main()
