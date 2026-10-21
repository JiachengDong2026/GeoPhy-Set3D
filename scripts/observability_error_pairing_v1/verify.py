"""Independent CSV pairing and multinomial-weighted bootstrap verification, no GT."""
from analyze import *


def percentile_sorted(values,q):
    x=sorted(values);position=(len(x)-1)*q;left=int(position);fraction=position-left
    return x[left]*(1-fraction)+x[min(left+1,len(x)-1)]*fraction


def run():
    protocol=verified(ART/'protocol.json');verify_hashes(protocol['sources'])
    execution=verified(ART/'scoring_lock.json')
    amendment=verified(ART/'statistics_export_amendment.json')
    for path,digest in execution['sources'].items():
        if path==amendment['changed_source']:
            assert digest==sha(amendment['original_snapshot'])==amendment['before_sha256']
            assert sha(path)==amendment['after_sha256']
        else:assert sha(path)==digest
    seal=verified(ART/'predictions_sealed.json');access=verified(ART/'GT64_access.json')
    assert seal['at']<=access['first_GT64_at'] and len(access['GT64_sha256'])==35 and not access['GT128_read']
    assert execution['authorization_sha256']==sha(ART/'GT64_authorization.json')
    cases=verified(ART/'cases.json');manifest=verified(ART/'prediction_manifest.json')
    assert len(manifest)==490 and len({(r['case_id'],r['condition'],r['draw'],r['model']) for r in manifest})==490
    raw=list(csv.DictReader((ART/'case_metrics.csv').open()));assert len(raw)==980
    assert all(np.isfinite(float(r[k])) for r in raw for k in METRICS)
    for m in verified(ART/'models.json'):assert sha(m['checkpoint'])==m['checkpoint_sha256']
    lookup={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in raw};assert len(lookup)==980
    pairs=list(csv.DictReader((ART/'paired_case_metrics.csv').open()));assert len(pairs)==700
    changes={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in csv.DictReader((ART/'prediction_changes.csv').open())}
    idx=np.random.default_rng(SEED).integers(0,35,(RESAMPLES,35))
    stored=list(csv.DictReader((ART/'bootstrap_indices.csv').open()))
    assert np.array_equal(idx,np.array([[int(r[f'i{i}']) for i in range(35)] for r in stored]))
    # Repeated-index resampling implemented independently as case-count weights.
    counts=np.array([np.bincount(row,minlength=35) for row in idx],dtype=np.float64)
    summaries=list(csv.DictReader((ART/'summary.csv').open()));maxdiff=0.
    statistics=verified(ART/'statistics.json')
    assert len(statistics['summary'])==len(summaries)==200
    for r in statistics['correlations']:
        assert (r['pearson'] is None and r['spearman'] is None and r['undefined_reason']) or (np.isfinite(r['pearson']) and np.isfinite(r['spearman']))
    for s in summaries:
        selected=[r for r in pairs if r['model']==s['model'] and r['condition']==s['condition'] and r['stage']==s['stage']]
        assert len(selected)==35
        bycase={r['case_id']:r for r in selected};assert set(bycase)==set(cases)
        v=[]
        for case in cases:
            pair=bycase[case];a=lookup[case,s['condition'],s['stage'],s['reference']];b=lookup[case,s['condition'],s['stage'],s['model']]
            if s['metric']=='D':value=float(changes[case,s['condition'],s['stage'],s['model']]['D'])
            else:
                value=float(b[s['metric']])-float(a[s['metric']])
                assert abs(value-float(pair['delta_'+s['metric']]))<1e-13
            v.append(value)
        v=np.array(v);bs=counts@v/35
        expected=dict(mean=float(sum(v)/35),median=percentile_sorted(v,.5),sd=float(np.sqrt(sum((v-v.mean())**2)/34)),
            ci95_low=percentile_sorted(bs,.025),ci95_high=percentile_sorted(bs,.975))
        for name,value in expected.items():
            error=abs(value-float(s[name]));maxdiff=max(maxdiff,error);assert error<1e-11,(s,name,error)
        if s['metric']!='D':
            assert int(s['increased'])==sum(x>TIE for x in v)
            assert int(s['decreased'])==sum(x < -TIE for x in v)
    dmax=0.
    with guard_for(manifest) as guard:
        for r in manifest:assert sha(r['prediction'])==r['prediction_sha256']
        for p in verified(ART/'plans.json'):
            selected=[r for r in manifest if r['plan_id']==p['plan_id']]
            aa={r['model']:fields(r['prediction']) for r in selected}
            for model,ff in aa.items():
                for stage,b in ff.items():
                    row=changes[p['case_id'],p['condition'],stage,model]
                    a=aa[row['D_reference_model']][stage]
                    value=float(np.linalg.norm((b-a).ravel(),ord=1)/(np.linalg.norm(a.ravel(),ord=1)+EPS))
                    error=abs(value-float(row['D']));dmax=max(dmax,error);assert error<1e-11
        assert not guard.denied
    freeze(ART/'independent_verification.json',dict(at=now(),passed=True,
        case_pair_identity_verified=True,all980_scored_fields_finite=True,all490_prediction_hashes_rechecked=True,
        bootstrap_method='independent multinomial counts dot case differences, independently interpolated sorted percentiles',
        seed=SEED,resamples=RESAMPLES,summary_rows_verified=len(summaries),maximum_statistical_difference=maxdiff,
        D_method='independent flattened vector L1 norms',maximum_D_difference=dmax,
        archived_official_metrics_maximum_difference=access['maximum_history_metric_difference'],
        GT64_read_in_verification=False,GT128_read=False,new_forwards=0,audit=guard.summary()))
    print('INDEPENDENT VERIFICATION PASS',maxdiff,dmax)


if __name__=='__main__':
    backend();run()
