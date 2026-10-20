"""Canonical FP32 300-epoch engine with an isolated model dispatcher."""
from runtime import *
import argparse
import copy
import torch
import train as engine
import probe_mrgr_multiscale_v1 as probe
from uniform_models import model_for

engine.OUT = BASE
engine.model_for = model_for

def register(arm, seed, smoke=None):
    assert arm in ARMS
    target = BASE/('training' if smoke is None else 'resume_smoke')/f'{arm}_seed{seed}'
    if smoke:
        target /= smoke
    if (target/'protocol.json').exists():
        verified(target/'protocol.json')
        return target
    config = copy.deepcopy(verified(HISTORY/'model_manifest.json')['models'][0]['config'])
    config['training']['seed'] = seed
    config['training']['device'] = 'cuda:0'
    config['experiment'] = dict(name=target.name, output_root=str(target))
    if arm == 'uniform_no_consistency':
        config['loss']['consistency'] = 0.
    model = model_for(config, arm, 'cpu')
    old = verified(HISTORY/'registration.json')
    freeze(target/'protocol.json', dict(
        registered_at=now(), arm=arm, seed=seed, config=config, epochs=300,
        parameters=sum(p.numel() for p in model.parameters()),
        sources=training_identity(), train_cases=old['cases']['train'],
        validation_cases=old['cases']['val'], subset_smoke=smoke is not None,
        planned_updates=24900 if smoke is None else 4,
        data_files=verified(HISTORY/'model_manifest.json')['training_data_sha256'],
        fresh_initialization=True, initial_model_sha256=probe.state_hash(model.state_dict()),
        selection='strict minimum mean validation projection H4 relative L1 over all300 epochs; no volume GT',
        precision='FP32, TF32 disabled, original numerical engine unchanged',
        resume='canonical last.pt epoch commit: full model/best/AdamW/cosine/RNG/loader/metrics; interrupted epoch replay',
        phase='formal300, with predeclared20-epoch engineering prefix for OARI/control only',
        excluded_from_main_model=arm!='uniform'))
    return target

def run(target, stop=300):
    target = Path(target)
    p = verified(target/'protocol.json')
    assert 0 < stop <= 300
    if not p['subset_smoke']:
        assert verified(BASE/'checks/cpu.json')['passed']
        assert verified(BASE/'checks/gpu.json')['passed']
        assert str(target) in verified(BASE/'design.json')['runs']
    with TrainGuard() as guard:
        engine.train(target, stop)
        assert not guard.denied
        write(target/'boundary_audit.json', guard.summary())
    status = read(target/'status.json')
    if not p['subset_smoke'] and stop==20:
        log = rows(target/'metrics.jsonl')
        saved = torch.load(target/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert len(log)==20 and saved['updates']==1660
        write(target/'prefix_complete.json', dict(at=now(), epoch=20, updates=1660,
            best_H4=saved['best_score'], seconds=saved['elapsed_seconds'],
            peak_bytes=max(r['peak_allocated_bytes'] for r in log),
            checkpoint_sha256=sha(target/'checkpoints/last.pt')))
    if stop==300 and not p['subset_smoke']:
        assert status['completed'] and status['epochs']==300 and status['updates']==24900

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--stop-after', type=int, default=300)
    options=parser.parse_args()
    run(options.target, options.stop_after)
