"""Freeze every checkpoint and analysis choice before supplementary test access."""
from runtime import *
import torch
import evaluation as historical_engine

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'final_lock.json').exists():return verified(BASE/'final_lock.json')
    old=verified(HISTORY/'final_lock.json');verify_hashes(old['data_identity'])
    for root in (PRUNING,FEEDBACK):
        a=verified(root/'acceptance.json');verify_hashes(a['evidence'])
    cases=verified(HISTORY/'registration.json')['cases']['test'];assert len(cases)==35
    layouts=verified(HISTORY/'layouts.json')['test'];assert len(layouts)==17
    freeze(BASE/'registration.json',dict(at=now(),cases={'test':cases},statistics=dict(seed=20261020,resamples=20000)))
    freeze(BASE/'layouts.json',dict(test=layouts))
    new=verified(PRUNING/'methods.json')+verified(FEEDBACK/'methods.json');assert len(new)==7
    audit=[]
    for m in new:
        run=Path(m['run']);p=verified(run/'protocol.json');verify_hashes(p['sources'])
        rr=rows(run/'metrics.jsonl');assert len(rr)==300 and [r['epoch'] for r in rr]==list(range(1,301))
        chosen=min(rr,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert chosen['epoch']==m['best_epoch']
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==chosen['epoch'] and ck['selection_metric']==chosen['val']['measurement']['heldout']['relative_l1']
        assert ck['config']==m['config'] and ck['protocol_sha256']==sha(run/'protocol.json') and not ck['volumetric_gt_used']
        last=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==300 and last['updates']==24900
        model=model_for(m['config'],m['arm'],'cpu');model.load_state_dict(ck['model'],strict=True)
        assert sum(x.numel() for x in model.parameters())==m['parameters']
        audit.append(dict(key=m['key'],epochs=300,updates=24900,best_epoch=chosen['epoch'],parameters=m['parameters'],selection='minimum validation H4 only',strict_state_load=True))
    reused=[m for m in old['methods'] if m['key'] in ('MRGR','multiscale_seed20261019','uniform_fusion_seed20261011','uniform_fusion_seed20261019')]
    assert len(reused)==4
    previous_seal=verified(HISTORY/'evaluation/test/all_predictions_sealed.json')
    previous_metrics=HISTORY/'evaluation/test/metrics.jsonl'
    references=[];evidence={str(previous_metrics):sha(previous_metrics),str(HISTORY/'final_lock.json'):sha(HISTORY/'final_lock.json'),str(HISTORY/'evaluation/test/all_predictions_sealed.json'):sha(HISTORY/'evaluation/test/all_predictions_sealed.json')}
    for m in reused:
        assert m in previous_seal['methods'];assert sha(m['checkpoint'])==m['checkpoint_sha256']
        target=HISTORY/'evaluation/test'/m['key'];manifest=target/'manifest.jsonl';complete=read(target/'complete.json');rr=rows(manifest)
        assert sha(manifest)==complete['manifest_sha256']==previous_seal['manifests'][str(manifest)]
        assert len(rr)==595 and complete['fields']==1190 and not complete['audit']['denied']
        historical_engine.validate_rows(cases,layouts,rr)
        for row in rr:assert sha(target/row['prediction'])==row['sha256']
        evidence[str(manifest)]=sha(manifest);evidence[str(target/'complete.json')]=sha(target/'complete.json')
        references.append(dict(m,prediction_root=str(target),manifest=str(manifest)))
    freeze(BASE/'reuse_methods.json',references)
    freeze(BASE/'reuse_audit.json',dict(at=now(),methods=[m['key'] for m in references],predictions_per_method=595,all_arrays_sha256_verified=True,evidence=evidence))
    freeze(BASE/'training_audit.json',dict(at=now(),passed=True,models=audit))
    freeze(BASE/'final_lock.json',dict(locked_at=now(),methods=new,sources=identity(),data_identity=old['data_identity'],gt64_stat_only=old['gt64_stat_only'],
        evidence={str(p):sha(p) for p in (BASE/'registration.json',BASE/'layouts.json',BASE/'training_audit.json',BASE/'reuse_audit.json',BASE/'reuse_methods.json')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),cases=35,layouts=17,new_models=7,new_reconstructions=4165,new_fields=8330,
        test_scope='Explicit user-requested supplementary evaluation of previously frozen development-selected models; historical35case test already used, not a new independent holdout',
        no_training=True,no_checkpoint_selection_on_test=True,no_test_tuning=True,GT128_forbidden=True,
        primary='C1 ordinary fixedV4 relativeL1; also report all17 layouts, per-case5draw aggregates, peaks, centroid,Dice and projection',
        comparisons='feedback_vs_matchedbaseline,feedback_vs_zeroresidual; C-A,D-B,D-C,B-A and (D-C)-(B-A) within each seed and descriptive two-seed case means',
        bootstrap='case paired20000 seed20261020; exploratory unadjusted95% intervals; training seed population uncertainty not covered',
        figures='fixed case indices0,17,34; GT peak y slice, shared asinh scale; clustered draw0 for image only, five draws for numerical aggregate'))
    # Boundary checks intentionally attempt forbidden reads before opening bytes.
    probes=[];source=next(r for r in rows(DATA/'source_index.jsonl') if r['case_id']==cases[0])
    for path in [DATA/source['gt_volume_64_path'],Path(source['source_target_path'])]:
        try:
            with TestGuard():
                with path.open('rb'):pass
        except PermissionError:probes.append(str(path))
        else:raise AssertionError('GT boundary did not block')
    freeze(BASE/'boundary_checks.json',dict(passed=True,blocked_before_bytes=probes,preseal_test_GT64_blocked=True,GT128_blocked=True))
    print('Locked7new models and4historical references, before new test predictions',flush=True)

if __name__=='__main__':main()
