"""Frozen formal case bootstrap; independently recount resamples from raw rows."""
from context import *

KEYS=('volume.relative_l1','volume.relative_l2','morphology.absolute_peak_ratio_error','volume.peak_absolute_error',
      'volume.peak_location_error_mm','volume.centroid_error_mm','morphology.centroid_distance_mm','morphology.dice_0.5',
      'measurement.observed.relative_l1','measurement.heldout.relative_l1')


def main():
    backend();verify_hashes(verified(BASE/'protocol.json')['sources']);gt=verified(BASE/'GT64_access.json');assert sha(BASE/'metrics.jsonl')==gt['metrics_sha256']
    rr=rows(BASE/'metrics.jsonl');cases=verified(BASE/'cases.json');lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    assert len(lookup)==3570
    methods=sorted({(r['method'],r['stage']) for r in rr});idx=np.random.default_rng(20261019).integers(0,35,(20000,35))
    panels=[];paired=[];case_rows=[];vectors={}
    for condition in CONDITIONS:
        draws=[0] if condition in ('Val-A-V4','spread-V4') else list(range(5))
        for method,stage in methods:
            vals=[]
            for case in cases:
                values=[official.scalar_metrics(lookup[method,stage,condition,d,case]) for d in draws]
                x={k:None if any(v[k] is None for v in values) else float(np.mean([v[k] for v in values])) for k in values[0]}
                vals.append(x);case_rows.append(dict(method=method,stage=stage,condition=condition,draws=draws,case_id=case,metrics=x))
            stats={}
            for k in vals[0]:
                arr=np.array([v[k] if v[k] is not None else np.nan for v in vals]);assert not np.isinf(arr).any()
                vectors[method,stage,condition,k]=arr
                finite=arr[np.isfinite(arr)];sample=arr[idx];valid=np.isfinite(sample).all(1)
                bs=sample[valid].mean(1)
                stats[k]=dict(valid_cases=len(finite),mean=float(finite.mean()) if len(finite) else None,median=float(np.median(finite)) if len(finite) else None,
                    p90=float(np.quantile(finite,.9)) if len(finite) else None,max=float(finite.max()) if len(finite) else None,
                    ci95_mean=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,valid_bootstrap_resamples=int(valid.sum()),
                    incomplete_case_rule='available-case summary; CI only complete sampled cases; see valid counts')
            panels.append(dict(method=method,stage=stage,condition=condition,draws=draws,metrics=stats))
        for method in CLASSICS:
            for metric in panels[-1]['metrics']:
                if not metric.startswith(('volume.','morphology.','measurement.')):continue
                a=vectors[MAIN,'C1',condition,metric];b=vectors[method,'C1',condition,metric];delta=a-b
                finite=delta[np.isfinite(delta)];valid=np.isfinite(delta[idx]).all(1);bs=delta[idx][valid].mean(1)
                paired.append(dict(condition=condition,draws=draws,method=MAIN,reference=method,stage='C1',metric=metric,
                    primary=condition=='Val-A-V4' and metric=='volume.relative_l1',n_cases=len(finite),
                    mean_main=float(np.nanmean(a)),mean_reference=float(np.nanmean(b)),mean=float(finite.mean()) if len(finite) else None,
                    median=float(np.median(finite)) if len(finite) else None,ci95=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,
                    bonferroni4_ci=np.quantile(bs,[.025/4,1-.025/4]).tolist() if len(bs) else None,
                    historical_bonferroni20_ci=np.quantile(bs,[.025/20,1-.025/20]).tolist() if len(bs) else None,
                    increased=int((delta>1e-8).sum()),decreased=int((delta < -1e-8).sum()),unchanged=int((np.abs(delta)<=1e-8).sum()),
                    case_differences={c:float(v) if np.isfinite(v) else None for c,v in zip(cases,delta)}))
    rates=[]
    for method in (MAIN,*CLASSICS):
        r=[x for x in rr if x['method']==method and x['stage']=='C1'];failed=[x for x in r if x['solver_status'] in ('line_search_failed','nonfinite')]
        rates.append(dict(method=method,expected=595,count=len(r),missing=595-len(r),numerical_failures=len(failed),failure_rate=len(failed)/595,
            budget_exhausted=sum(x['solver_status']=='budget_exhausted' for x in r),objective_stable=sum(x['solver_status']=='objective_stable' for x in r)))
    worst=[]
    for condition in CONDITIONS:
        sel=[x for x in case_rows if x['condition']==condition and x['stage']=='C1' and x['method']==MAIN]
        worst.extend(dict(condition=condition,**{k:r[k] for k in ('case_id','draws','metrics')}) for r in sorted(sel,key=lambda r:r['metrics']['volume.relative_l1'],reverse=True)[:5])
    freeze(BASE/'summary.json',dict(at=now(),exploratory=True,independent_confirmation=False,statistics=verified(BASE/'protocol.json')['statistics'],
        delta='main minus classic; negative favors main for error metrics, positive favors main for Dice',panels=panels,paired=paired,failure_rates=rates,worst_main_cases=worst))
    jsonl(BASE/'case_panels.jsonl',case_rows)
    csv_write(BASE/'paired_summary.csv',[dict({k:v for k,v in r.items() if k not in ('case_differences','draws','ci95','bonferroni4_ci','historical_bonferroni20_ci')},
        draws=','.join(map(str,r['draws'])),ci95_low=r['ci95'][0] if r['ci95'] else None,ci95_high=r['ci95'][1] if r['ci95'] else None,
        bonf4_low=r['bonferroni4_ci'][0] if r['bonferroni4_ci'] else None,bonf4_high=r['bonferroni4_ci'][1] if r['bonferroni4_ci'] else None,
        bonf20_low=r['historical_bonferroni20_ci'][0] if r['historical_bonferroni20_ci'] else None,bonf20_high=r['historical_bonferroni20_ci'][1] if r['historical_bonferroni20_ci'] else None) for r in paired])
    # Independent algorithm: rebuild draw means from raw nested records with Python
    # arithmetic, multinomial count weights, sorted linear percentile interpolation.
    weights=np.array([np.bincount(i,minlength=35) for i in idx],dtype=float)
    def percentile(values,q):
        v=sorted(values);pos=(len(v)-1)*q;i=int(pos);f=pos-i;return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
    maximum=0.;checked=0
    for r in paired:
        if r['metric'] not in KEYS:continue
        group,key=r['metric'].split('.',1);v=[]
        for case in cases:
            def value(method,d):
                obj=lookup[method,'C1',r['condition'],d,case][group]
                if group=='measurement':label,k=key.split('.');return obj[label][k]
                return obj[key]
            vals=[value(MAIN,d)-value(r['reference'],d) for d in r['draws']]
            v.append(sum(vals)/len(vals))
        bs=weights@np.array(v)/35
        comparisons=[(sum(v)/35,r['mean']),(percentile(v,.5),r['median'])]
        for name,qs in [('ci95',[.025,.975]),('bonferroni4_ci',[.025/4,1-.025/4]),('historical_bonferroni20_ci',[.025/20,1-.025/20])]:
            comparisons.extend((percentile(bs,q),r[name][j]) for j,q in enumerate(qs))
        for a,b in comparisons:maximum=max(maximum,abs(a-b));assert abs(a-b)<1e-10,(r['metric'],a,b)
        assert sum(x>1e-8 for x in v)==r['increased'] and sum(x < -1e-8 for x in v)==r['decreased'];checked+=1
    # Paper ablation/complexity tables are unchanged prior evidence, no rerun.
    assert verified(CORE/'acceptance.json')['completed'] and verified(CORE/'independent_verification.json')['passed']
    verify_hashes(verified(CORE/'evaluation_lock.json')['sources'])
    for m in verified(CORE/'model_manifest.json'):assert sha(m['checkpoint'])==m['checkpoint_sha256']
    st=verified(CORE/'statistics.json');abl=[r for r in st['summaries'] if r['stage']=='C1' and r['metric']=='relative_l1' and (r['condition']=='Val-A-V4' or len(r['draws'])==5)]
    assert len(abl)==12
    csv_write(BASE/'paper_core_ablation.csv',[dict(r,draws=','.join(map(str,r['draws'])),ci95_low=r['ci95'][0],ci95_high=r['ci95'][1]) for r in abl])
    bench=verified(CORE/'benchmark.json');full={m['seed']:m for m in verified(CORE/'references.json')}
    costs=[]
    for r in bench['records']:
        if not r['key'].startswith(PLAIN):continue
        m=next(m for m in verified(CORE/'model_manifest.json') if m['key']==r['key']);train=read(Path(m['run'])/'summary.json')
        baseline=next(x for x in bench['records'] if x['key']==full[m['seed']]['key'])
        env=read(Path(m['run'])/'environment.json');logs=rows(Path(m['run'])/'metrics.jsonl')
        costs.append(dict(key=r['key'],seed=m['seed'],parameters=r['parameters'],delta_parameters=r['parameters']-baseline['parameters'],
            mean_forward_ms=r['mean_seconds']*1000,median_forward_ms=r['median_seconds']*1000,sd_forward_ms=r['sd_seconds']*1000,
            delta_forward_ms=(r['mean_seconds']-baseline['mean_seconds'])*1000,peak_allocated_MiB=r['peak_allocated_bytes']/1024**2,
            train_seconds=train['elapsed_seconds'],train_epochs=300,train_updates=24900,training_log=str(Path(m['run'])/'metrics.jsonl'),
            gpu_uuid=r['gpu_uuid'],FLOPs='not measured',MACs='not measured',energy='not measured'))
    csv_write(BASE/'paper_complexity.csv',costs)
    freeze(BASE/'independent_statistics_verification.json',dict(at=now(),passed=True,paired_vectors_checked=checked,maximum_difference=maximum,
        algorithm='raw draw differences + multinomial count weights + independently sorted interpolated quantiles',
        core_previous_independent_verification_sha256=sha(CORE/'independent_verification.json'),
        core_statistics_sha256=sha(CORE/'statistics.json'),core_benchmark_sha256=sha(CORE/'benchmark.json'),core_acceptance_sha256=sha(CORE/'acceptance.json')))
    print('ANALYSIS',len(paired),'paired vectors;',checked,'independently verified',flush=True)


if __name__=='__main__':main()
