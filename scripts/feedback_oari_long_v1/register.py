"""Immutable continuation registration; preserve the completed pilot artifacts."""
from runtime import *
import copy
import shutil
import torch
import numpy as np
import train as engine

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'design.json').exists():return verified(BASE/'design.json')
    acceptance=verified(PILOT/'acceptance.json');verify_hashes(acceptance['evidence'])
    old_design=verified(PILOT/'design.json');verify_hashes(old_design['sources'])
    evidence=[];runs=[]
    for name in ('registration.json','model_manifest.json','layouts.json','cpu_checks.json','gpu_checks.json'):
        verified(PILOT/name);shutil.copy2(PILOT/name,BASE/name);shutil.copy2((PILOT/name).with_suffix('.json.sha256'),(BASE/name).with_suffix('.json.sha256'))
    # User requested a longer follow-up after observing exploratory results.
    # This is explicitly an adaptive extension, not an independent confirmation.
    for arm in ('feedback_oari','baseline','feedback_oari_zero_residual'):
        origin=PILOT/'training'/f'{arm}_seed{SEED}';target=BASE/'training'/origin.name
        source=origin/'checkpoints/last.pt';old=verified(origin/'protocol.json');summary=verified(origin/'pilot_summary.json')
        assert summary['epochs']==20 and summary['updates']==1660
        verify_hashes(old['sources']);saved=torch.load(source,map_location='cpu',weights_only=False)
        assert saved['epoch']==20 and saved['updates']==1660 and len(saved['metrics'])==20
        assert saved['protocol_sha256']==sha(origin/'protocol.json') and saved['config']==old['config']
        assert saved['scheduler']['T_max']==300 and not saved['volumetric_gt_used']
        protocol=copy.deepcopy(old);protocol.update(registered_at=now(),epochs=EPOCHS,planned_updates=24900,
            sources=identity(),fresh_initialization=False,phase='adaptive single-seed development continuation to300',
            selection='strict minimum validation H4 over all300 epochs including retained20-epoch prefix; no volume GT',
            continuation=dict(origin=str(origin),checkpoint=str(source),checkpoint_sha256=sha(source),
                protocol_sha256=sha(origin/'protocol.json'),start_epoch=20,additional_epochs=280,
                config_note='Original config, including experiment metadata, preserved exactly for true resume; artifacts use new target directory.',
                only_checkpoint_field_changed='protocol_sha256; explicit migration binds identical states to extended-budget protocol'))
        freeze(target/'protocol.json',protocol)
        migrated=dict(saved,protocol_sha256=sha(target/'protocol.json'))
        target.joinpath('checkpoints').mkdir(parents=True,exist_ok=True)
        torch.save(migrated,target/'checkpoints/last.pt')
        loaded=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert loaded.keys()==saved.keys()
        for k in saved:
            if k!='protocol_sha256':exact(saved[k],loaded[k])
        engine.publish_best(target,loaded);jsonl(target/'metrics.jsonl',loaded['metrics'])
        write(target/'status.json',dict(state='queued',epoch=20,max_epochs=300,best_epoch=saved['best_epoch'],best_val_H=saved['best_score']))
        evidence.append(dict(arm=arm,origin_checkpoint=str(source),origin_sha256=sha(source),
            migrated_checkpoint_sha256=sha(target/'checkpoints/last.pt'),
            unchanged_fields=[k for k in saved if k!='protocol_sha256'],exact=True))
        runs.append(str(target))
    freeze(BASE/'migration_verification.json',dict(passed=True,at=now(),runs=evidence,
        states='model/best/optimizer/scheduler/RNG/loader RNG/metrics/config/counters exact; only protocol binding explicitly migrated',
        pilot_acceptance_sha256=sha(PILOT/'acceptance.json')))
    design=dict(at=now(),git_base=old_design['git_base'],sources=identity(),runs=runs,arms=list(ARMS),seed=SEED,
        epochs=300,additional_epochs_per_arm=280,total_updates_per_arm=24900,additional_updates_per_arm=23240,
        pilot=str(PILOT),pilot_acceptance_sha256=sha(PILOT/'acceptance.json'),
        architecture='Identical to completed pilot; second OARI inside residual adapter, unchanged projection and losses',
        continuation='All three arms resume their own last.pt at epoch20, original cosineT300 and complete recovery state. No LR reset or optimizer reset.',
        selection='minimum validation H4 over full300; no early stopping',
        data='166train/36previously-used development; all17layouts; predictions sealed before GTval scoring; no test projection/GT64/GT128',
        scheduling='at most one independent fully idle GPU, shared UUID locks; no processes, used<100MiB, free>=20000MiB, util<=5% twice15s and immediate recheck; no interruption of running tasks',
        analysis='same full volume/projection/peak/centroid/Dice metrics and paired case bootstrap20000; clustered5draw averaged within case; independent recomputation',
        limitations=['one training seed at300','adaptive extension chosen after observing20epoch development results; not independent confirmation','new OARI increases parameters versus baseline','no capacity-matched non-attention block','no declaration of final-test superiority'],
        original_checks_reused=True,no_early_stopping=True,no_test_or_GT128=True)
    freeze(BASE/'design.json',design);return design

if __name__=='__main__':main()
