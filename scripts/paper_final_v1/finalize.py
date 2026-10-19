"""Fail-closed final method lock and post-score acceptance audit."""
from common import *
import argparse
from evaluation import training_methods,target_for
from classical import METHODS


def lock():
    assert read(OUT/'numerical_checks.json')['passed'] and read(OUT/'training_resume_verification.json')['passed'] and read(OUT/'boundary_checks.json')['passed']
    assert read(OUT/'loss_checks.json')['passed']
    assert read(OUT/'evaluation/classical_dev/independent_verification.json')['passed']
    assert read(OUT/'evaluation/val/independent_verification.json')['passed']
    assert not read(OUT/'evaluation/val/summary.json')['failures'],'Resolve development numerical failures before final method lock'
    reg=verified(OUT/'registration.json');selected=verified(OUT/'classical/selected.json');methods=training_methods()
    methods += [dict(key=m,kind='classical',config=selected['selected'][m]['config']) for m in METHODS]
    from sart import wait_development
    assert verified(OUT/'sart/integration_checks.json')['passed']
    methods.append(wait_development())
    extension=verified(OUT/'module_ablations_v2/registration.json')
    assert verified(OUT/'module_ablations_v2/checks.json')['passed']
    assert len(methods)==extension['final_methods']==21 and len({m['key'] for m in methods})==21
    neural_count=sum(m['kind']=='neural' for m in methods)
    fields_total=sum(1190 if m['kind']=='neural' else 595 for m in methods)
    assert neural_count==17 and fields_total==extension['final_fields']==22610
    for m in methods:
        if m['kind']=='neural':assert sha(m['checkpoint'])==m['checkpoint_sha256']
    execution_sources=sources()
    for name in ('scripts/analyze_frozen_ab_morphology_v1.py','scripts/analyze_historical_reconstruction_v1.py','tests/test_classical_reconstruction.py','tests/test_classical_workflow_boundary.py'):
        execution_sources[str(ROOT/name)]=sha(ROOT/name)
    choices=OUT/'final_method_choices_closed.json'
    closure=dict(methods=methods,sources=execution_sources,cases=reg['cases']['test'],settings_sha256=sha(OUT/'layouts.json'),primary=reg['primary'],statistics=reg['statistics'],no_pending_hyperparameters=True)
    if choices.exists():
        previous=verified(choices)
        for k,v in closure.items():assert previous[k]==v,k
    else:freeze(choices,dict(closed_at=now(),**closure))
    # Identity of test measurements is now allowed after all development choices are final;
    # never open/hash GT64 or GT128 here. GT identity uses manifest and filesystem stat only.
    source=rows(DATA/'source_index.jsonl');test=[r for r in source if r['split']=='test'];measurements={};gt_stat={}
    with AccessGuard(splits=['test']) as preflight_guard:
        for r in test:
            measurements[str(DATA/r['train_projection_path'])]=sha(DATA/r['train_projection_path'])
            path=DATA/r['gt_volume_64_path'];s=path.stat();gt_stat[r['case_id']]=dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns,source_GT128_hash_from_existing_manifest=r['source_gt_sha256'])
    identity={str(DATA/n):sha(DATA/n) for n in ('source_index.jsonl','train_statistics.json','geometry/train_geometry.npz')}
    evidence={str(OUT/n):sha(OUT/n) for n in ('registration.json','model_manifest.json','layouts.json','classical/search.json','classical/selected.json','classical/implementation_amendment_v2.json','evaluation/val/summary.json','evaluation/val/independent_verification.json')}
    evidence[str(choices)]=sha(choices)
    for name in ('sart/registration.json','sart/selected.json','sart/calibration_seal.json','sart/numerical_checks.json','sart/integration_checks.json','sart/development_complete.json','evaluation/sart_dev/independent_verification.json'):
        evidence[str(OUT/name)]=sha(OUT/name)
    for name in ('module_ablations_v2/registration.json','module_ablations_v2/queue.json','module_ablations_v2/checks.json','module_ablations_v2/training_complete.json','module_ablations_v2/integration_checks.json'):
        record=verified(OUT/name)
        if name.endswith('checks.json'):assert record['passed']
        evidence[str(OUT/name)]=sha(OUT/name)
    scheduling_amendment=OUT/'module_ablations_v2/scheduling_overlap_v3/amendment.json'
    if scheduling_amendment.exists():
        verified(scheduling_amendment)
        evidence[str(scheduling_amendment)]=sha(scheduling_amendment)
    for m in methods:
        if m['kind']=='neural':
            for n in ('protocol.json','summary.json'):evidence[str(Path(m['run'])/n)]=sha(Path(m['run'])/n)
    freeze(OUT/'final_lock.json',dict(locked_at=now(),methods=methods,case_ids=reg['cases']['test'],settings_sha256=sha(OUT/'layouts.json'),
        source_commit=verified(OUT/'model_manifest.json')['git_commit'],sources=execution_sources,data_identity=dict(identity,**measurements),gt64_stat_only=gt_stat,
        evidence=evidence,preflight_access_audit=preflight_guard.summary(),primary=reg['primary'],statistics=reg['statistics'],reconstructions_per_method=595,
        fields_total=fields_total,expected_reconstructions_total=len(methods)*595,no_pending_hyperparameters=True,
        runtime_scope=dict(neural='synchronized C0+C1 forward; excludes batch loading, final array writes and external projections',classical='solver wall time including periodic recovery-checkpoint I/O; geometry setup and final array writes separate',SART='CPU sparse CSR sweeps and conversion including periodic recovery I/O; no GPU allocation; CPU peak RSS recorded separately',comparison='report scopes explicitly; scheduler wall time and complete worker wall time are also retained; no unqualified pure-compute speed ratio'),
        supplementary_operator=dict(SART='A interpolates nodal mole fraction; discrete approximation to frozen P(C); all evaluation projections use exact frozen P(C)'),
        failure_disposition='fixed-budget exhaustion retained; no methods omitted; numerical failure blocks scoring pending explicit documented software diagnosis before labels',
        GT64_opened=False,GT128_opened=False,recovery='only verifiable software bug; preserve original logs and predictions; versioned amendment then regenerate affected outputs without parameter/metric/endpoint change',
        stopping_rule='after final score, no further training, checkpoint selection or tuning'))
    print(f'Final {len(methods)}-method lock created before test prediction and GT access',flush=True)


def accept():
    r=verified(OUT/'final_lock.json');verify_hashes(r['sources']);verify_hashes(r['data_identity']);verify_hashes(r['evidence'])
    seal=verified(OUT/'evaluation/test/all_predictions_sealed.json');assert seal['methods']==r['methods']
    score=read(OUT/'evaluation/test/gt_access.json');independent=read(OUT/'evaluation/test/independent_verification.json')
    assert independent['passed'] and score['first_GT64_at']>seal['sealed_at'] and score['unique_GT64_files']==35
    assert not score['gt128_read'];assert not score['audit']['denied'] and not independent['audit']['denied']
    assert sum(seal['counts'].values())==r['expected_reconstructions_total']==12495
    metrics=rows(OUT/'evaluation/test/metrics.jsonl');assert len(metrics)==r['fields_total']==22610
    assert len({(x['method'],x['stage'],x['condition'],x['draw'],x['case_id']) for x in metrics})==22610
    for m in r['methods']:
        if m['kind']=='neural':assert sha(m['checkpoint'])==m['checkpoint_sha256']
    old=verified(OUT/'model_manifest.json')['models'][0];verify_hashes({str(ROOT/k):v for k,v in old['source_sha256'].items()})
    summary=read(OUT/'evaluation/test/summary.json');assert not summary['failures'],'Explicit numerical failure disposition required'
    primary=[p for p in summary['paired'] if p['primary']]
    assert len(primary)==20 and all(p['comparisons']==20 for p in primary)
    from module_registry_v2 import ablation_arms
    ablations=read(OUT/'evaluation/test/ablation_paired.json')['results']
    assert {p['arm'] for p in ablations}==set(ablation_arms())
    assert len(ablations)==7*5*2*3==independent['within_seed_ablation_comparisons_verified']
    # Require every launched process to have an end time and exit code; early test failures remain visible.
    tasks=[]
    for p in (OUT/'scheduling').glob('*/*.json'):
        x=read(p)
        if 'pid' in x:assert 'exit_code' in x and 'ended_at' in x,p;tasks.append(x)
    from interpret import main as interpretation
    conclusions=interpretation()
    write(OUT/'final_acceptance.json',dict(at=now(),completed=True,methods=len(r['methods']),test_cases=35,layouts=17,reconstructions=12495,fields=22610,
        GT64_after_complete_seal=True,GT128_read=False,test_tuning=False,post_score_training=False,original_model_unchanged=True,
        independent_verification=independent,interpretation=conclusions,task_records=len(tasks),historical_negative_test_exit_codes_preserved=[x['key'] for x in tasks if x['exit_code']!=0],
        limitations=['remote HEAD unavailable; proxy refused','two seeds, not robust population seed variance','development reuse','fixed classical compute budget','no unseen geometric domain/noise/calibration study']))
    print('Final acceptance PASSED',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['lock','accept']);args=a.parse_args();globals()[args.action]()
