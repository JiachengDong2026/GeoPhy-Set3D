"""Reuse-only observability/error pairing. Audit never opens a target array.

The score command requires a separately recorded, explicit phase authorization;
historical GT access is deliberately insufficient. No training or inference API.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/paper_final_v1'))
from common import *
import argparse
import csv
import platform
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ART = ROOT / 'docs/reports/artifacts/mrgr_observability_error_pairing_v1'
REPORT = ROOT / 'docs/reports/mrgr_observability_error_pairing_v1_zh.md'
OLDART = ROOT / 'docs/reports/artifacts/mrgr_oari_mechanism_v2'
PRIOR = Path(verified(OLDART / 'acceptance.json')['scratch'])
UNIFORM = PRIOR.parent/'mrgr_uniform_final_v1'
CONDITIONS = ('Val-A-V4', 'clustered-V4')
GROUPS = ('A0', 'obs_token', 'obs_bias', 'obs_both')
KEYS = tuple(f'{a}_seed{s}' for a in ('uniform_fusion', 'uniform_pruned_observability') for s in (20261011, 20261019))
SEED, RESAMPLES, EPS, TIE = 20261025, 20000, 1e-6, 1e-8
METRICS = ('relative_l1', 'relative_l2', 'peak_ratio', 'absolute_peak_ratio_error',
           'peak_absolute_error', 'peak_location_error_mm', 'centroid_error_mm',
           'centroid_distance_mm', 'Dice50')


def save_csv(path, rr):
    assert rr and not path.exists(), path
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rr for k in r)), lineterminator='\n')
        w.writeheader(); w.writerows(rr)


def fields(path):
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'C0', 'C1'}
        out = {s: z[s].astype(np.float64) for s in ('C0', 'C1')}
    assert all(a.shape == (64,)*3 and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in out.values())
    return out


def stats(values, indices):
    x = np.asarray(values, dtype=float)
    assert len(x) == 35 and np.isfinite(x).all()
    ci = np.quantile(x[indices].mean(1), [.025, .975])
    return dict(n_cases=35, mean=float(x.mean()), median=float(np.median(x)), sd=float(x.std(ddof=1)),
                ci95_low=float(ci[0]), ci95_high=float(ci[1]), minimum=float(x.min()),
                q25=float(np.quantile(x,.25)), q75=float(np.quantile(x,.75)),
                p90=float(np.quantile(x,.9)), maximum=float(x.max()))


def flatten(r):
    v, m = r['volume'], r['morphology']
    return dict(relative_l1=v['relative_l1'], relative_l2=v['relative_l2'],
        peak_ratio=m['peak_ratio'], absolute_peak_ratio_error=m['absolute_peak_ratio_error'],
        peak_absolute_error=v['peak_absolute_error'], peak_location_error_mm=v['peak_location_error_mm'],
        centroid_error_mm=v['centroid_error_mm'], centroid_distance_mm=m['centroid_distance_mm'], Dice50=m['dice_0.5'])


def guard_for(manifest, gt=False):
    g = AccessGuard(splits=[] if gt else ['test'], gt_splits=['test'] if gt else [], output=ART)
    for r in manifest:
        g.allowed[str(Path(r['prediction']).resolve())] = 'sealed_reused_prediction'
    return g


def audit():
    ART.mkdir(parents=True, exist_ok=False)
    models = [m for m in verified(UNIFORM/'reuse_methods.json') if m['key'] in KEYS]
    assert {m['key'] for m in models} == set(KEYS)
    freeze(ART/'models.json', models)
    checks = []
    for i in range(4):
        result = subprocess.run([sys.executable, str(HERE/'checkpoint_check.py'), str(ART/'models.json'), str(i)],
                                text=True, capture_output=True)
        (ART/f'checkpoint_{i}.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        checks.append(json.loads(result.stdout.strip().splitlines()[-1]))
    freeze(ART/'checkpoint_audit.json', checks)
    cases = verified(PRIOR/'cases.json'); assert len(cases) == len(set(cases)) == 35
    plans = [p for p in verified(PRIOR/'plans.json') if p['condition'] in CONDITIONS]
    assert len(plans) == 70 and all(p['draw'] == 0 for p in plans)
    op = verified(PRIOR/'protocol.json'); assert op['GT64_read'] is False and op['GT128_read'] is False
    assert op['model']['checkpoint_sha256'] == models[0]['checkpoint_sha256']
    prior_manifest = verified(PRIOR/'prediction_manifest.json')
    lookup = {(r['plan_id'],r['group']):r for r in prior_manifest}
    manifests, records, chains = {}, [], []
    for m in models:
        root = Path(m['prediction_root']); mp = root/'manifest.jsonl'
        complete = read(root/'complete.json')
        sealpath = root.parent/'all_predictions_sealed.json'; seal = verified(sealpath)
        assert seal['all_predictions_saved_before_GT']
        registered = next(x for x in seal['methods'] if x['key'] == m['key'])
        assert registered['checkpoint_sha256'] == m['checkpoint_sha256'] == complete['checkpoint_sha256']
        assert sha(mp) == complete['manifest_sha256'] == seal['manifests'][str(mp)]
        rr = rows(mp); assert len(rr) == len({(x['case_id'],x['condition'],x['draw']) for x in rr}) == 595
        manifests[m['key']] = {(r['case_id'],r['condition'],r['draw']):r for r in rr}
        chains.append(dict(model=m['key'], manifest=str(mp), manifest_sha256=sha(mp), seal=str(sealpath),
            seal_sha256=sha(sealpath), checkpoint_sha256=m['checkpoint_sha256'], config=m['config']))
    m0 = next(m for m in models if m['key']=='uniform_fusion_seed20261011')
    for p in plans:
        case, condition = p['case_id'], p['condition']; assert case in cases
        base = lookup[p['plan_id'],'A0']
        for group in GROUPS:
            r = lookup[p['plan_id'],group]
            unit = verified(PRIOR/'units'/f'p{p["plan_id"]:03d}'/f'{group}.json')
            assert r['checkpoint_sha256'] == m0['checkpoint_sha256'] == unit['checkpoint_sha256']
            assert r['input_sha256'] == base['input_sha256'] == unit['input_sha256']
            assert unit['case_id'] == case and unit['condition'] == condition and unit['group'] == group
            assert unit['identity'] == verified(PRIOR/'units'/f'p{p["plan_id"]:03d}'/'A0.json')['identity']
            assert unit['eval'] and unit['no_grad'] and unit['model_state_sha256'] == op['state_sha256']
            assert unit['prediction_sha256'] == r['prediction_sha256'] and unit['prediction'] == r['prediction']
            records.append(dict(case_id=case, condition=condition, draw=0, model=group, seed=20261011,
                input_ids=p['input_ids'], input_sha256=r['input_sha256'], checkpoint_sha256=r['checkpoint_sha256'],
                prediction=r['prediction'], prediction_sha256=r['prediction_sha256'], reused=True,
                source_record=str(PRIOR/'units'/f'p{p["plan_id"]:03d}'/f'{group}.json'), plan_id=p['plan_id']))
        for m in models:
            r = manifests[m['key']][case,condition,0]
            a = manifests[m0['key']][case,condition,0]
            assert r['input_ids'] == p['input_ids'] == a['input_ids'] and r['heldout_ids'] == a['heldout_ids']
            if m['key'] == m0['key']:
                assert str(Path(m['prediction_root'])/r['prediction']) == base['prediction'] and r['sha256'] == base['prediction_sha256']
                continue
            records.append(dict(case_id=case, condition=condition, draw=0, model=m['key'], seed=m['seed'],
                input_ids=r['input_ids'], input_sha256=base['input_sha256'], checkpoint_sha256=m['checkpoint_sha256'],
                prediction=str(Path(m['prediction_root'])/r['prediction']), prediction_sha256=r['sha256'], reused=True,
                source_record=str(Path(m['prediction_root'])/'manifest.jsonl'), plan_id=p['plan_id']))
    assert len(records) == 490
    freeze(ART/'cases.json',cases); freeze(ART/'plans.json',plans); freeze(ART/'provenance_chain.json',chains)
    freeze(ART/'prediction_manifest.json',records)
    sources = {str(p):sha(p) for p in HERE.glob('*.py')}
    for p in (ROOT/'scripts/paper_final_v1/evaluation.py', ROOT/'src/geophy_set3d/evaluation/research.py',
              ROOT/'scripts/report_historical_reconstruction_v1.py', ROOT/'scripts/analyze_frozen_ab_morphology_v1.py',
              ROOT/'scripts/oari_mechanism_v2/interventions.py', ROOT/'scripts/paper_final_v1/common.py'):
        sources[str(p)] = sha(p)
    # Historical access is stage-specific. Never reinterpret the old no-GT protocol.
    evidence = {str(p):sha(p) for p in (
        ROOT/'docs/reports/artifacts/mrgr_uniform_ablations_v1/stage_registration.json',
        ROOT/'docs/reports/artifacts/mrgr_uniform_final_v1/evaluation_stage_amendment_v1.json',
        ROOT/'docs/reports/artifacts/mrgr_uniform_ablations_v1/gt_access.json', PRIOR/'protocol.json')}
    freeze(ART/'authorization_audit.json',dict(at=now(), status='current_phase_authorization_not_found',
        historical_stage_scoring_authorized=True, historical_access_does_not_authorize_this_phase=True,
        prior_OARI_GT64_read=False, GT128_forbidden=True, evidence=evidence,
        action='Pause new GT scoring until explicit phase authorization; continue no-GT audit and historical-metric reuse'))
    freeze(ART/'protocol.json',dict(at=now(), experiment='mrgr_observability_error_pairing_v1',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=sources, cases=cases, layouts=list(CONDITIONS), draws=[0], n_cases=35,
        primary='obs_token-minus-A0 C1 relative_l1 separately by layout', auxiliary=['C0','obs_bias','obs_both','two_seed_structural_deletion'],
        bootstrap=dict(seed=SEED,resamples=RESAMPLES,unit='paired case',interval='percentile95% interval for mean paired difference; exploratory unadjusted'),
        near_unchanged_tolerance=TIE, D='sum(abs(B-A))/(sum(abs(A))+1e-6)',
        D_legacy='sum(abs(B-A))/max(sum(abs(A)),1e-6), retained separately for exact old-v2 comparison',
        metrics=list(METRICS), error_delta='intervention minus baseline; Dice50 positive means better, peak_ratio has no monotonic quality direction',
        units='relative errors/peak ratio/Dice are dimensionless, pp=100*ratio difference; peak absolute error is mass fraction; distances mm',
        metric_implementations='official volume_metrics and safe_shape, float64 raw[0,1] concentration on XYZ 64 endpoint grid over0.6m; no new normalization',
        Dice50='morphology.dice_0.5: same threshold0.5*GTmax for both arrays; NOT fixed absolute0.001/0.0001',
        centroid='both trapezoidal volume.centroid_error_mm and ordinary-node morphology.centroid_distance_mm retained',
        exploratory_patterns=dict(large_D=0.05,small_abs_delta_L1=0.005,small_D=0.01,
            large_peak_ratio_error_delta=0.1,large_location_delta_mm=600/63,
            interpretation='descriptive fixed markers, not significance or model usefulness thresholds'),
        correlation='Pearson and Spearman across35cases, exploratory only; no causal independent-contribution claim',
        training=False,backward=False,inference_planned=0,GT128_read=False,
        GT64_requires='explicit current-phase authorization linked to protocol and sealed prediction manifest',
        test_previously_used=True, independent_confirmation=False,
        dependencies=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform())))
    drows=[]
    with guard_for(records) as guard:
        for r in records: assert sha(r['prediction']) == r['prediction_sha256']
        from geophy_set3d.data.collate import projection_pool_collate
        settings={s['condition']:s for s in verified(PRIOR/'layouts.json') if s['condition'] in CONDITIONS}
        datasets={condition:dataset('test',s) for condition,s in settings.items()}
        for p in plans:
            rr = [r for r in records if r['plan_id'] == p['plan_id']]
            sample=datasets[p['condition']][p['case_index']]
            assert sample['case_id']==p['case_id']
            batch={k:v for k,v in projection_pool_collate([sample]).items() if not k.startswith('hold_')}
            h=hashlib.sha256()
            for k,v in sorted(batch.items()):
                h.update(k.encode())
                if isinstance(v,torch.Tensor):
                    a0=v.detach().cpu().contiguous().numpy()
                    h.update(hashlib.sha256(str((a0.shape,a0.dtype)).encode()+a0.tobytes()).hexdigest().encode())
                else:h.update(repr(v).encode())
            assert h.hexdigest()==rr[0]['input_sha256']
            a = fields(next(r['prediction'] for r in rr if r['model']=='A0'))
            for r in rr:
                b = fields(r['prediction'])
                reference_model='A0'
                if 'pruned' in r['model'] and r['seed']==20261019:
                    reference_model='uniform_fusion_seed20261019'
                ref= a if reference_model=='A0' else fields(next(x['prediction'] for x in rr if x['model']==reference_model))
                for stage in ('C0','C1'):
                    numerator=float(np.abs(b[stage]-ref[stage]).sum()); denominator=float(np.abs(ref[stage]).sum())
                    d = numerator/(denominator+EPS); legacy=numerator/max(denominator,EPS)
                    if r['model'] in GROUPS:
                        unit=verified(r['source_record']);old=next(x for x in unit['output_rows'] if x['stage']==stage)
                        assert abs(legacy-old['relative_l1']) < 1e-11
                    drows.append(dict(case_id=r['case_id'],condition=r['condition'],draw=0,model=r['model'],seed=r['seed'],stage=stage,D_reference_model=reference_model,
                        D=d,D_legacy=legacy,prediction_mean_abs_change=numerator/(64**3)))
        assert not guard.denied
    save_csv(ART/'prediction_changes.csv',drows)
    freeze(ART/'predictions_sealed.json',dict(at=now(),protocol_sha256=sha(ART/'protocol.json'),
        manifest_sha256=sha(ART/'prediction_manifest.json'),predictions=490,fields=980,
        all_prediction_hashes_shapes_ranges_verified=True,new_forwards=0,GT64_read=False,GT128_read=False,audit=guard.summary()))
    freeze(ART/'audit_complete.json',dict(at=now(),passed=True,missing_predictions=0,new_forwards_required=0,
        complete_inputs=70, reused_unique_predictions=490,checkpoint_count=4,
        GT64_authorization='unresolved_for_current_phase',GT64_read=False,GT128_read=False))
    print('AUDIT PASS: 4 checkpoints; 490 unique predictions; 0 missing; 0 new forwards; GT64 not opened.',flush=True)


def historical():
    """Derive draw0 structural pairs from already published metrics, without GT."""
    models=verified(ART/'models.json'); cases=verified(ART/'cases.json')
    reuse=verified(UNIFORM/'reuse_prediction_seal.json')
    rr=[]; identity=[]
    for m in models:
        ref=next(x for x in reuse['methods'] if x['key']==m['key'])
        assert sha(m['source_metrics'])==ref['metrics_sha256']
        source=[r for r in rows(m['source_metrics']) if r['method']==m['key'] and r['condition'] in CONDITIONS and r['draw']==0]
        assert len(source)==140
        for r in source: rr.append(dict(case_id=r['case_id'],condition=r['condition'],draw=0,model=m['key'],seed=m['seed'],stage=r['stage'],**flatten(r)))
        identity.append(dict(model=m['key'],source=m['source_metrics'],sha256=sha(m['source_metrics'])))
    save_csv(ART/'historical_case_metrics_draw0.csv',rr); freeze(ART/'historical_metrics_sources.json',identity)
    lookup={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in rr}
    indices=np.random.default_rng(SEED).integers(0,35,(RESAMPLES,35)); summary=[]; pairs=[]
    for condition in CONDITIONS:
        for seed in (20261011,20261019):
            for stage in ('C0','C1'):
                for metric in METRICS:
                    diffs=[]
                    for case in cases:
                        a=lookup[case,condition,stage,f'uniform_fusion_seed{seed}'][metric]
                        b=lookup[case,condition,stage,f'uniform_pruned_observability_seed{seed}'][metric]
                        assert a is not None and b is not None
                        d=b-a; diffs.append(d)
                        pairs.append(dict(case_id=case,condition=condition,draw=0,seed=seed,stage=stage,metric=metric,baseline=a,structural_deletion=b,delta=d))
                    summary.append(dict(condition=condition,draw=0,seed=seed,stage=stage,metric=metric,
                        **stats(diffs,indices),increased=sum(x>TIE for x in diffs),decreased=sum(x < -TIE for x in diffs),unchanged=sum(abs(x)<=TIE for x in diffs)))
    save_csv(ART/'historical_structural_pairs_draw0.csv',pairs)
    save_csv(ART/'historical_structural_summary_draw0.csv',summary)
    freeze(ART/'historical_analysis_complete.json',dict(at=now(),GT64_read=False,GT128_read=False,
        source='reused already scored scalar metrics, exact draw0; no claim of new GT verification',rows=len(rr),paired_rows=len(pairs)))
    for r in summary:
        if r['stage']=='C1' and r['metric']=='relative_l1': print(r)


if __name__ == '__main__':
    backend()
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','historical'])
    args=parser.parse_args()
    try:
        {'audit':audit,'historical':historical}[args.action]()
    except Exception:
        import traceback
        ART.mkdir(parents=True,exist_ok=True)
        failure=ART/f'failure_{args.action}_{datetime.now().strftime("%Y%m%dT%H%M%S%f")}.log'
        failure.write_text(traceback.format_exc());raise
