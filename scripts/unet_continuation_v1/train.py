"""Resume successful case0 weights AND optimizer/scheduler/RNG; bounded monitoring."""
from context import *
from evaluation import evaluate
import signal
import traceback


def stop_reason(rows):
    row=rows[-1]
    for field in row['fields']:
        if field['mean']<1e-8 or field['nearzero']>=.95 or field['std']/max(field['mean'],1e-30)<.05:
            return 'output_collapse'
    if len(rows)>=4:
        recent=rows[-3:];first=rows[0]
        if all(v['extra_E_H']>1.25*first['extra_E_H'] for v in recent):return 'unseen_E_persistently_worse_25percent'
        if all(v['loss']>2*first['loss'] and v['measurement']['heldout']>1.2*first['measurement']['heldout'] for v in recent):return 'loss_and_H_divergence'
    if row['step']>=200:
        previous=rows[-8:-4];recent=rows[-4:]
        loss_improve=1-np.mean([v['loss'] for v in recent])/np.mean([v['loss'] for v in previous])
        H_improve=1-np.mean([v['measurement']['heldout'] for v in recent])/np.mean([v['measurement']['heldout'] for v in previous])
        if loss_improve<.01 and H_improve<.01:return 'loss_and_H_plateau_two_100step_windows'
    return None


def main():
    setup();protocol=verify_all()
    assert os.environ['GEOPHY_CONTINUATION_UUID_LOCKED']==os.environ['CUDA_VISIBLE_DEVICES']
    device='cuda:0';assert torch.cuda.is_available()
    directory=ROOT/'case0';directory.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT.parent)
    steps=0;rows=[];reason='budget_complete';started=time.monotonic()
    generator=torch.Generator().manual_seed(20261011)
    def interrupted(signum,frame):raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    with guard:
        payload=torch.load(read(OLD_ART/'prepared.json')['cases'][0]['path'],map_location='cpu',weights_only=False)
        batch={k:move_batch(v,device) for k,v in payload['batch'].items()}
        extra=move_batch(payload['extra'],device)
        model=fresh().to(device)
        ck=torch.load(SOURCE,map_location='cpu',weights_only=False)
        assert ck['steps']==100
        model.load_state_dict(ck['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck['rng'],generator);del ck
        def checkpoint(name):
            atomic_checkpoint(directory/name,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng_state(generator),steps=100+steps,added_updates=steps,teacher_preheat_updates=200,metrics=rows,source_sha256=protocol['source_checkpoint_sha256']))
        try:
            initial=evaluate(model,batch,extra,directory,0,True);rows.append(initial)
            reference=read(OLD_ART/'case0_original_refinement.json')['final']
            np.testing.assert_allclose([initial['loss'],initial['extra_E_H']], [reference['loss'],reference['extra_E_H']],rtol=.002,atol=2e-6)
            append(directory/'metrics.jsonl',initial);checkpoint('initial.pt')
            print('BASELINE',json.dumps(initial),flush=True)
            for step in range(1,protocol['max_additional_updates']+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                loss,_,_,_=objective(model,batch)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                assert norm>0
                previous=[v.detach().clone() for v in model.parameters()]
                optimizer.step();steps=step
                delta=torch.stack([(v.detach()-old).double().square().sum() for v,old in zip(model.parameters(),previous)]).sum().sqrt().item()
                assert np.isfinite(delta) and delta>0
                if (100+step)%16==0:scheduler.step()
                append(directory/'updates.jsonl',dict(update=step,original_objective_update=100+step,loss_before_update=float(loss.detach()),
                    gradient_norm=float(norm),actual_parameter_delta=delta,lr=optimizer.param_groups[0]['lr']))
                if step%protocol['evaluation_interval']==0:
                    row=evaluate(model,batch,extra,directory,step,True)
                    assert np.isfinite(row['loss']) and all(np.isfinite(v) for v in row['measurement'].values()) and np.isfinite(row['extra_E_H'])
                    rows.append(row);append(directory/'metrics.jsonl',row);checkpoint('last.pt')
                    reason=stop_reason(rows) or 'running'
                    write(ART/'status.json',dict(state=reason,additional_updates=steps,total_original_objective_updates=100+steps,latest=row))
                    print('MONITOR',step,'loss',row['loss'],'H',row['measurement']['heldout'],'cross',row['measurement']['cross'],'E',row['extra_E_H'],'state',reason,flush=True)
                    if reason!='running':break
            else:reason='budget_complete'
        except BaseException as exc:
            reason='exception_'+type(exc).__name__
            checkpoint('interrupted.pt')
            write(ART/'failure.json',dict(reason=reason,message=str(exc),actual_additional_updates=steps,traceback=traceback.format_exc()))
            raise
        finally:
            checkpoint('last.pt')
            write(ART/'access.json',guard.summary())
        result=dict(index=0,actual_additional_updates=steps,total_original_objective_updates=100+steps,teacher_preheat_updates=200,
            stop_reason=reason,initial=rows[0],final=rows[-1],metrics=rows,source_checkpoint=str(SOURCE),source_sha256=sha(SOURCE),
            checkpoint=str(directory/'last.pt'),checkpoint_sha256=sha(directory/'last.pt'),optimizer_reset=False,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,access=guard.summary(),formal_training=False,
            E_used_for_early_stopping=True,E_is_final_test=False)
        write(ART/'result.json',result);write(directory/'result.json',result)
    verify_all()
    print('FINISHED',reason,'additional_updates',steps,flush=True)


if __name__=='__main__':main()
