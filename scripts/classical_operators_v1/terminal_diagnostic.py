"""Separate, zero-update audit of an interrupted pilot; never resume training."""
import argparse,time
from context import *
from diagnostics import diagnose,gates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['deeponet','fno']);a=ap.parse_args();arm=a.arm
    verify();c=load_config(arm);target=ROOT/arm/'terminal_audit';target.mkdir(exist_ok=False)
    failure=ROOT/arm/'pilot/failure_state.pt'
    exact_failure_updates=int(torch.load(failure,map_location='cpu',weights_only=False)['optimizer_steps']) if failure.exists() else None
    reg=dict(script_sha256=sha(__file__),checkpoint=str(ROOT/arm/'pilot/last.pt'),checkpoint_sha256=sha(ROOT/arm/'pilot/last.pt'),
        source_manifest_sha256=sha(ART/'source_manifest.json'),new_updates=0,mode='saved last complete epoch only, not interrupted in-memory state')
    write(target/'registration.json',reg);setup();began=time.perf_counter()
    guard=GTReadGuard(c['dataset']['root'],ROOT/arm)
    with guard:
        ck=torch.load(reg['checkpoint'],map_location='cuda:0',weights_only=False);assert ck['pilot_only'] and not ck['volumetric_gt_used']
        model=build_model(c);model.load_state_dict(ck['model']);before=state_hash(model.state_dict())
        train,val=make_dataset(c),make_dataset(c,validation=True)
        torch.cuda.reset_peak_memory_stats();validation=run_epoch(model,make_loader(val,c),c)
        history=[json.loads(s) for s in (ROOT/arm/'pilot/metrics.jsonl').read_text().splitlines()]
        assert len(history)==ck['epoch'];logged=history[-1]['val']
        assert validation['measurement']==logged['measurement'] and validation['case_station_digest']==logged['case_station_digest']
        directory=target/'interventions';directory.mkdir();rows=diagnose(model,c,train,val,ck['epoch'],directory)
        initial=read(ROOT/arm/'pilot/epoch0.json')['val'];checks=gates(rows,validation,initial)
        # No optimizer: inspect genuine gradient values with a FP64 norm to
        # distinguish tiny FP32 norm underflow from exactly zero elements.
        batch=move_dual_batch(dual_measurement_collate([train[0],train[1]]),'cuda:0')
        model.eval();model.zero_grad(set_to_none=True);loss,*_=objective(model,batch);loss.backward()
        gradients={k:dict(norm_fp32=float(p.grad.norm()),norm_fp64=float(p.grad.double().norm()),nonzero=int(p.grad.count_nonzero()),finite=bool(torch.isfinite(p.grad).all())) for k,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True);after=state_hash(model.state_dict());assert before==after
        access=guard.summary();no_forbidden(access)
        checkpoint=dict(epoch=ck['epoch'],updates=ck['optimizer_steps'],optimizer_steps=sorted({int(v['step']) for v in ck['optimizer']['state'].values()}),
            scheduler_epoch=ck['scheduler']['last_epoch'],RNG_saved=bool(ck['rng']))
        result=dict(arm=arm,status='NO_GO' if not all(checks.values()) else 'INCONCLUSIVE',checks=checks,checkpoint=checkpoint,
            checkpoint_sha256=reg['checkpoint_sha256'],initial_val_H=selection_metric(initial),last_val_H=selection_metric(validation),
            validation_replay=validation,logged_validation_exact=True,gradient_diagnostic=gradients,model_hash_before=before,model_hash_after=after,
            access=access,new_updates=0,optimizer_created=False,seconds=time.perf_counter()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            actual_training_updates=dict(authoritative_checkpoint=ck['optimizer_steps'],process_lower_bound=ck['optimizer_steps'],
                process_upper_bound=ck['optimizer_steps']+15,exact_interrupted_epoch_count_available=exact_failure_updates is not None,exact_process_updates=exact_failure_updates,
                reason='Per-step telemetry was only persisted at pilot completion; interrupted epoch in-memory counter lost. No replay or fabricated exact count.'),
            formal_started=False,formal_updates=0)
        write(target/'result.json',result);write(ART/(arm+'_terminal_audit.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
