"""One training-only process; stored nodes are evaluated by a separate process."""
from numerics_v2 import *
import argparse
import time

def main(arm,smoke=False):
    registration=verified(ROOT/'registration.json') if not smoke else None
    if registration:
        verify(registration['sources'])
        verify({str(ROOT/'inputs/train_AB.pt'):registration['inputs'][str(ROOT/'inputs/train_AB.pt')]})
    backend();began=time.monotonic();target=ROOT/('smoke' if smoke else 'runs')/arm;target.mkdir(parents=True,exist_ok=False)
    with Boundary('smoke' if smoke else 'train') as access:
        samples=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)
        assert len(samples)==6 and all(set(s)=={'A','B'} for s in samples)
        net=model(arm,'cuda:0');initial_hash=state_hash(net.state_dict())
        expected=verified(ROOT/'input_registration.json')['initial_states'][arm];assert initial_hash==expected
        if smoke:
            ab={k:batch(samples[:2],k,'cuda:0') for k in ('A','B')}
            loss,parts=objective(net,ab);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            save_pt(target/'roundtrip.pt',dict(model=net.state_dict()))
            # Smoke cannot read any checkpoint by design; serialization checked
            # in the separate CPU engineering test, no warm start here.
            assert state_hash(net.state_dict())==initial_hash
            freeze(target/'result.json',dict(at=now(),passed=True,optimizer_updates=0,loss=float(loss.detach()),initial_hash=initial_hash,
                checkpoint_saved=True,access=access.summary()))
            print('SMOKE',arm,'forward/backward/checkpoint saved;0 updates',flush=True);return
        opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
        freeze(target/'protocol.json',dict(at=now(),arm=arm,config=design(),parameters=sum(p.numel() for p in net.parameters()),
            initial_state_sha256=initial_hash,registration_sha256=sha(ROOT/'registration.json')))
        checkpoint_hashes={}
        def checkpoint(update):
            path=target/f'checkpoint_{update:03d}.pt'
            save_pt(path,dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=update,initial_state_sha256=initial_hash,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            # Training cannot reopen its checkpoints; hashes are computed later.
            checkpoint_hashes[str(update)]=str(path)
        checkpoint(0);torch.cuda.reset_peak_memory_stats()
        for update in range(1,201):
            ids=[((update-1)*2+i)%6 for i in range(2)];ab={k:batch([samples[i] for i in ids],k,'cuda:0') for k in ('A','B')}
            net.train();opt.zero_grad(set_to_none=True);loss,parts=objective(net,ab);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            head=net.head[-1] if arm=='gnot' else net.decoder[-1]
            gradients={k:stats(v.grad) for k,v in [('readout_weight',head.weight),('readout_bias',head.bias)]}
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
            assert all(int(v['step'])==update for v in opt.state.values())
            append(target/'updates.jsonl',dict(update=update,case_indices=ids,loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
                gradients=gradients,gradient_norm_before_clip=norm,learning_rate=2e-4,elapsed_seconds=time.monotonic()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),H_in_training=False,truth_in_training=False))
            if update in design()['evaluation_updates']:
                checkpoint(update);write(target/'status.json',dict(at=now(),state='training',update=update));print(arm,update,'loss',float(loss.detach()),flush=True)
        assert not access.denied
        result=dict(at=now(),completed=True,arm=arm,updates=200,initial_state_sha256=initial_hash,final_state_sha256=state_hash(net.state_dict()),
            checkpoints=checkpoint_hashes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-began,
            access=access.summary(),evaluation_in_this_process=False)
        freeze(target/'training_result.json',result)
    verify(registration['sources']);print('TRAIN COMPLETE',arm,'200 updates,no H/truth opens',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gnot','deepsets'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.arm,a.smoke)
