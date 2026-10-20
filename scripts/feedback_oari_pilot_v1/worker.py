"""Fixed 20-epoch prefix of the canonical300 schedule, with full recovery state."""
from runtime import *
import argparse
import copy
import torch
import train as engine
import probe_mrgr_multiscale_v1 as probe
from pilot_models import model_for

engine.original_identity=engine.source_identity
engine.OUT=BASE;engine.model_for=model_for;engine.source_identity=identity

def register(arm,smoke=None):
    target=BASE/('training' if smoke is None else 'resume_smoke')/f'{arm}_seed{SEED}'
    if smoke:target=target/smoke
    if (target/'protocol.json').exists():return target
    config=copy.deepcopy(verified(HISTORY/'model_manifest.json')['models'][0]['config'])
    config['experiment']=dict(name=target.name,output_root=str(target));config['training']['seed']=SEED
    m=model_for(config,arm,'cpu');reg=verified(HISTORY/'registration.json')
    freeze(target/'protocol.json',dict(registered_at=now(),arm=arm,seed=SEED,config=config,epochs=EPOCHS,parameters=sum(p.numel() for p in m.parameters()),
        sources=identity(),train_cases=reg['cases']['train'],validation_cases=reg['cases']['val'],
        selection='strict lowest mean validation H4 relative L1 within the predeclared20 epoch prefix; no volume GT',
        fresh_initialization=True,data_files=verified(HISTORY/'model_manifest.json')['training_data_sha256'],
        subset_smoke=smoke is not None,planned_updates=1660 if smoke is None else 4,phase='exploratory pilot, not full300 training',
        initial_model_sha256=probe.state_hash(m.state_dict()),
        schedule='canonical cosine T_max300 retained; run exactly first20 epochs; all3 arms equal budget',
        initialization='all baseline retained tensors exact; added OARI is an independent deepcopy of same-seed first-stage initial OARI; no RNG consumed',
        resume='canonical engine last.pt epoch commit; full model/best/optimizer/scheduler/RNG/loader/counters and deterministic epoch sampling'))
    return target

def run(target,stop=EPOCHS):
    target=Path(target);p=verified(target/'protocol.json')
    if not p['subset_smoke']:
        assert stop==EPOCHS and verified(BASE/'gpu_checks.json')['passed']
        assert verified(BASE/'cpu_checks.json')['passed']
    with DevGuard(splits=['train','val']) as guard:
        engine.train(target,stop)
        assert not guard.denied
        write(target/'boundary_audit.json',guard.summary())
        if not p['subset_smoke']:
            log=rows(target/'metrics.jsonl');assert len(log)==EPOCHS
            saved=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
            assert saved['updates']==1660 and saved['epoch']==EPOCHS
            assert read(target/'environment.json')['initial_model_sha256']==p['initial_model_sha256']
            summary=dict(state='pilot_complete',completed_pilot=True,full300_completed=False,epochs=EPOCHS,updates=1660,
                arm=p['arm'],seed=SEED,parameters=p['parameters'],best_epoch=saved['best_epoch'],best_val_H=saved['best_score'],
                seconds=saved['elapsed_seconds'],peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in log),
                checkpoint_sha256=sha(target/'checkpoints/best.pt'),at=now())
            freeze(target/'pilot_summary.json',summary);write(target/'status.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--stop-after',type=int,default=EPOCHS);a=p.parse_args();run(a.target,a.stop_after)
