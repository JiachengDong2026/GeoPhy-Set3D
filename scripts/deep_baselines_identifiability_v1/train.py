"""Exactly bounded fresh A/B training. No H or truth parameter exists in step."""
from diagnostics import *
from evaluate import evaluate,decide
import argparse
import time

def step(net,ab,opt,version):
    with TrainGuard() as access:
        opt.zero_grad(set_to_none=True)
        total,parts,_,_=objective(net,ab,version);total.backward()
        groups=parameter_groups(net)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
        opt.step()
    return dict(loss=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
                groups=groups,gradient_norm_before_clip=norm,H_in_training=False,truth_in_training=False,
                forbidden_accesses=access.denied)

def run(experiment,arm,version,n):
    registration=verified('registration.json');c.verify_hashes(registration['sources'])
    name=f'{experiment}_{arm}_{version}_n{n}';target=ROOT/'runs'/name
    target.mkdir(parents=True,exist_ok=False)
    c.backend();torch.set_num_threads(2);began=time.monotonic()
    with TrainGuard(synthetic_truth=True) as access:
        inputs=ROOT/('inputs/synthetic_AB.pt' if experiment=='B' else 'inputs/real_AB.pt')
        samples=torch.load(inputs,map_location='cpu',weights_only=False)[:n]
        assert all(set(s)=={'A','B'} for s in samples)
        net=model(arm,'cuda:0');initial_hash=state_hash(net.state_dict())
        assert initial_hash==registration['initial_states'][arm]
        opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
        protocol=dict(experiment=experiment,arm=arm,version=version,n=n,seed=design()['seed'],
            initial_state_sha256=initial_hash,parameters=sum(p.numel() for p in net.parameters()),
            budget=200,config=design(),registration_sha256=c.sha(ROOT/'registration.json'),
            inputs_sha256=c.sha(inputs),H_in_loss=False,GT_in_loss=False)
        c.freeze(target/'protocol.json',protocol)
        history=[evaluate(net,samples,list(range(n)),experiment=='B',target,0)]
        c.freeze(target/'initial.json',history[0]);updates=[]
        size=min(n,2);torch.cuda.reset_peak_memory_stats()
        for update in range(1,201):
            start=((update-1)*size)%n;chosen=[samples[(start+i)%n] for i in range(size)]
            ab={k:batch(chosen,k) for k in ('A','B')};net.train()
            row=step(net,ab,opt,version)
            row.update(update=update,case_indices=[(start+i)%n for i in range(size)],learning_rate=opt.param_groups[0]['lr'],
                elapsed_seconds=time.monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            assert all(int(s['step'])==update for s in opt.state.values())
            updates.append(row);c.append(target/'updates.jsonl',row)
            if update in design()['evaluation_updates']:
                evaluation=evaluate(net,samples,list(range(n)),experiment=='B',target,update)
                evaluation['elapsed_seconds']=time.monotonic()-began;history.append(evaluation)
                c.append(target/'evaluations.jsonl',evaluation)
                atomic(target/f'checkpoint_update{update:03d}.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=update,
                    initial_state_sha256=initial_hash,protocol_sha256=c.sha(target/'protocol.json'),H_in_training=False,
                    RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()))
                c.write(target/'status.json',dict(at=c.now(),state='diagnostic_training',update=update,H=evaluation['H_relative_l1'],train=evaluation['train_relative_l1']))
                print(name,update,'AB',evaluation['train_relative_l1'],'H',evaluation['H_relative_l1'],flush=True)
        result=decide(history)
        result.update(name=name,at=c.now(),updates=200,initial_train=history[0]['train_relative_l1'],
            final_train=history[-1]['train_relative_l1'],initial_H=history[0]['H_relative_l1'],final_H=history[-1]['H_relative_l1'],
            checkpoint=str(target/'checkpoint_update200.pt'),checkpoint_sha256=c.sha(target/'checkpoint_update200.pt'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-began,
            initial_state_sha256=initial_hash,final_state_sha256=state_hash(net.state_dict()),audit=access.summary())
        assert result['final_state_sha256']!=initial_hash and not access.denied
        c.verify_hashes(registration['sources']);c.freeze(target/'result.json',result)
    c.write(target/'status.json',dict(at=c.now(),**result));print('DONE',name,result['status'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',choices=['A','B'],required=True)
    ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--version',choices=['v2','v3'],required=True)
    ap.add_argument('--n',type=int,choices=[1,2,4],required=True);a=ap.parse_args()
    assert (a.experiment=='A' and a.version=='v2') or (a.experiment=='B' and a.n==4)
    run(a.experiment,a.arm,a.version,a.n)
