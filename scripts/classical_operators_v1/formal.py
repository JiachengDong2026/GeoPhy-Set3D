"""Fresh full-budget training, gated by independently reviewed pilot."""
import math,time
from context import *
from train_deepsets_selfsup import InputLedger

def train(arm,c):
    from worker import optimizer
    verify();decision=read(ART/(arm+'_pilot.json'));review=read(ART/(arm+'_independent_review.json'))
    assert decision['status']=='GO' and review['all_passed'] and review['scientific_GO']
    remote=read(ART/'github_main.json');assert remote['verified'] and remote['commit']==read(ART/'history_audit.json')['base_commit']
    target=ROOT/arm/'formal';target.mkdir(parents=True,exist_ok=True)
    assert not (target/'training_complete.json').exists(),'Completed formal run is immutable'
    guard=GTReadGuard(c['dataset']['root'],target);setup()
    with guard:
        model=build_model(c);initial=state_hash(model.state_dict())
        assert initial==decision['initial_state_sha256'],'Fresh initialization must reproduce registered state'
        opt=optimizer(model,c);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(20261011)
        train_ds,val=make_dataset(c),make_dataset(c,validation=True);assert (len(train_ds),len(val))==(166,36)
        train_loader=make_loader(train_ds,c,True,generator);val_loader=make_loader(val,c)
        start=0;best=float('inf');last=target/'last.pt'
        if last.exists():
            # Only atomically completed epochs are authoritative; no pilot checkpoint is loaded.
            ck=torch.load(last,map_location='cuda:0',weights_only=False)
            assert ck['config']==c and not ck['pilot_only'] and ck['source_manifest_sha256']==sha(ART/'source_manifest.json')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore_rng(ck['rng'],generator)
            start=ck['epoch'];best=ck['best_val_metric'];opt.updates=ck['optimizer_steps']
            for name in ['metrics.jsonl','actual_training_inputs.jsonl']:
                p=target/name
                if p.exists():
                    rows=[json.loads(s) for s in p.read_text().splitlines() if json.loads(s)['epoch']<=start]
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        ledger=InputLedger(model);by_id={r['case_id']:i for i,r in enumerate(train_ds.records)}
        for epoch in range(start+1,301):
            began=time.perf_counter();train_ds.set_epoch(epoch-1);val.set_epoch(0);ledger.reset();before=opt.updates;lr=opt.param_groups[0]['lr'];torch.cuda.reset_peak_memory_stats()
            tr=run_epoch(model,train_loader,c,opt);order=[case for r in ledger.rows if r['subset']=='A' for case in r['case_ids']]
            assert len(order)==166 and set(order)==set(by_id)
            for r in ledger.rows:
                for case,ids in zip(r['case_ids'],r['station_ids']):
                    a,b,h=train_ds.sampler.sample(train_ds.station_count,epoch=epoch-1,sample_index=by_id[case],case_id=case)
                    assert ids==(a if r['subset']=='A' else b).tolist()
            assert opt.updates-before==83
            with (target/'actual_training_inputs.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,batches=ledger.rows))+'\n')
            ledger.reset();va=run_epoch(model,val_loader,c);scheduler.step();score=selection_metric(va);improved=score<best;best=min(score,best)
            row=dict(epoch=epoch,optimizer_steps=opt.updates,learning_rate=lr,train=tr,val=va,seconds=time.perf_counter()-began,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),updates=opt.trace)
            with (target/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            opt.trace=[]
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=epoch,
                config=c,optimizer_steps=opt.updates,best_val_metric=best,pilot_only=False,initial_state_sha256=initial,
                source_manifest_sha256=sha(ART/'source_manifest.json'),volumetric_gt_used=False,selection_metric='mean held-out measurement relative L1')
            atomic_checkpoint(last,ck)
            if improved:atomic_checkpoint(target/'best.pt',ck)
            no_forbidden(guard.summary());write(target/'training_access.json',guard.summary())
            print(f'{arm} formal {epoch}/300 steps={opt.updates} valH={score:.8f}',flush=True)
        ledger.close();assert opt.updates==24900
        rows=[json.loads(s) for s in (target/'metrics.jsonl').read_text().splitlines()];assert len(rows)==300
        bestrow=min(rows,key=lambda r:selection_metric(r['val']))
        write(target/'training_complete.json',dict(completed=True,epochs=300,optimizer_steps=24900,best_epoch=bestrow['epoch'],
            best_val_heldout_relative_l1=best,total_seconds=sum(r['seconds'] for r in rows),peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rows),access=guard.summary()))
        write(target/'frozen_best.json',dict(checkpoint=str(target/'best.pt'),sha256=sha(target/'best.pt'),completed_epochs=300,optimizer_steps=24900,
            best_epoch=bestrow['epoch'],volumetric_gt_used=False,pilot_only=False,config=c,source_manifest_sha256=sha(ART/'source_manifest.json')))
