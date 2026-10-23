"""Pre-test module-extension identities and fail-closed integration gate."""
from common import *
import time
from module_models_v2 import ARMS
BASE=OUT/'module_ablations_v2'


def register():
    assert not (OUT/'final_method_choices_closed.json').exists() and not (OUT/'final_lock.json').exists()
    from module_train_v2 import register as register_run
    runs=[]
    # Prioritize one seed for all four questions before repetitions.
    for seed in (20261011,20261019):
        for arm in ARMS:runs.append(str(register_run(arm,seed)))
    evidence={str(Path(r)/'protocol.json'):sha(Path(r)/'protocol.json') for r in runs}
    freeze(BASE/'registration.json',dict(at=now(),reason='User requested OARI/GCA and other module ablations and completion of final35 tests before test access',
        parent_registration_sha256=sha(OUT/'registration.json'),arms=list(ARMS),seeds=[20261011,20261019],runs=runs,protocol_hashes=evidence,
        epochs=300,updates=24900,initialization='fresh; identical full-model initial tensors for all retained parameters per seed',
        designs=dict(no_ray_relations='zero first3 pairwise bias inputs (cosine, sine, azimuth difference); retain local descriptor and all validity masks',
            no_observability='zero all5 descriptor inputs to both embedding and pairwise bias; retain pairwise ray relations and masks',
            uniform_fusion='replace shared GCA at both C0/C1 by valid-view arithmetic mean; learned GCA parameters removed; OARI geometry retained',
            zero_residual='same full-model parameter tensors; residual channel always zero in training and evaluation; path/mask channels remain; skip internal P(C0); retain external measurement losses'),
        limitations=['no_ray_relations/no_observability preserve nominal parameters but zeroed-input weights cannot receive task gradients; report active capacity difference',
            'uniform_fusion removes learned fusion parameters and changes computation, not a strict capacity-matched control',
            'zero_residual retains contextual adapter and acquisition geometry, tests measurement residual information beyond extra capacity',
            'historical seed11 full-model execution reused; CUDA backward nondeterminism remains',
            'two seeds cannot establish seed-population uncertainty'],
        selection='all300 epochs, lowest mean validation H4 relativeL1 only; no volume GT for training or selection',
        final_methods=21,neural_methods=17,classical_methods=4,final_primary_comparators=20,final_reconstructions=12495,final_fields=22610,
        primary=verified(OUT/'registration.json')['primary'],statistics=verified(OUT/'registration.json')['statistics'],
        comparison_inclusion='all eight trained ablations retained regardless of development results; no dropping based on test scores',
        test_GT_opened=False,GT128_opened=False))
    freeze(BASE/'queue.json',[dict(key='module_checks_v2',args=['scripts/paper_final_v1/module_checks_v2.py'])]+
        [dict(key=Path(r).name,args=['scripts/paper_final_v1/module_train_v2.py','train','--target',r]) for r in runs])


def methods(wait=True):
    reg=verified(BASE/'registration.json')
    if wait:
        while not (BASE/'training_complete.json').exists():
            s=read(BASE/'status.json')
            if s['state']=='failed':raise RuntimeError('Module ablation training failed; preserve logs and inspect')
            try:os.kill(s['pid'],0)
            except ProcessLookupError:raise RuntimeError('Module controller absent before completion')
            print('Waiting for registered eight module ablations before final evaluation',flush=True);time.sleep(15)
        verify_hashes(verified(BASE/'training_complete.json')['evidence'])
    result=[]
    for r in reg['runs']:
        d=Path(r);p=verified(d/'protocol.json');s=read(d/'summary.json');verify_hashes(p['sources'])
        assert s['completed'] and s['epochs']==300 and s['updates']==24900 and not p['subset_smoke']
        assert all(s['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        log=rows(d/'metrics.jsonl');assert len(log)==300 and [v['epoch'] for v in log]==list(range(1,301))
        best=min(log,key=lambda x:x['val']['measurement']['heldout']['relative_l1'])
        assert best['epoch']==s['best_epoch'] and best['val']['measurement']['heldout']['relative_l1']==s['best_val_H']
        ck=d/'checkpoints/best.pt';assert sha(ck)==s['best_checkpoint_sha256']
        result.append(dict(key=d.name,kind='neural',run=str(d),checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=s['best_epoch'],
            arm=p['arm'],config=p['config'],parameters=p['parameters'],seed=p['seed']))
    return result


def ablation_arms():
    return ('single_scale','no_feedback','no_consistency',*ARMS) if (BASE/'registration.json').exists() else ('single_scale','no_feedback','no_consistency')


if __name__=='__main__':register()
