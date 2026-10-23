"""Official paired case bootstrap and an independent count-weighted verification."""
from common_stage import *
import csv
METRICS=('relative_l1','relative_l2','peak_ratio','absolute_peak_ratio_error','peak_absolute_error',
    'peak_location_error_mm','centroid_error_mm','centroid_distance_mm','Dice50')

def csv_write(path,records):
    assert records and not path.exists()
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');w.writeheader();w.writerows(records)

def statistics():
    if (BASE/'statistics.json').exists():return verified(BASE/'statistics.json')
    records=list(csv.DictReader((BASE/'metrics.csv').open()));cases=verified(BASE/'cases.json')
    lookup={(r['method'],r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in records};assert len(lookup)==3360
    idx=np.random.default_rng(SEED).integers(0,35,(20000,35));summaries=[];case_rows=[]
    comparisons=[(f'{arm}_seed{s}',s) for arm,s in TASKS]
    for model,seed in comparisons:
        basekey=f'{PLAIN}_seed{seed}'
        settings=[('Val-A-V4',[0]),('clustered-V4',[0])]
        settings.append(('clustered-V4',list(range(5))))
        for condition,draws in settings:
            for stage in ('C0','C1'):
                values={k:[] for k in (*METRICS,'D')};basevals={k:[] for k in METRICS};treatvals={k:[] for k in METRICS}
                for case in cases:
                    aa=[lookup[basekey,case,condition,d,stage] for d in draws];bb=[lookup[model,case,condition,d,stage] for d in draws]
                    row=dict(case_id=case,model=model,reference=basekey,seed=seed,condition=condition,draws=','.join(map(str,draws)),stage=stage)
                    for metric in METRICS:
                        a=float(np.mean([float(x[metric]) for x in aa]));b=float(np.mean([float(x[metric]) for x in bb]));delta=b-a
                        row[metric]=b;row['baseline_'+metric]=a;row['delta_'+metric]=delta
                        values[metric].append(delta);basevals[metric].append(a);treatvals[metric].append(b)
                    row['D']=float(np.mean([float(x['D']) for x in bb]));values['D'].append(row['D']);case_rows.append(row)
                for metric,v in values.items():
                    x=np.array(v);ci=np.quantile(x[idx].mean(1),[.025,.975])
                    summaries.append(dict(model=model,seed=seed,reference=basekey,condition=condition,draws=draws,stage=stage,metric=metric,n_cases=35,
                        mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)),ci95=ci.tolist(),
                        minimum=float(x.min()),maximum=float(x.max()),
                        baseline_mean=float(np.mean(basevals[metric])) if metric!='D' else None,
                        treatment_mean=float(np.mean(treatvals[metric])) if metric!='D' else None,
                        increased=int((x>1e-8).sum()),decreased=int((x < -1e-8).sum()),unchanged=int((np.abs(x)<=1e-8).sum())))
    csv_write(BASE/'paired_case_metrics.csv',case_rows)
    csv_write(BASE/'summary.csv',[dict(r,draws=','.join(map(str,r['draws'])),ci95_low=r['ci95'][0],ci95_high=r['ci95'][1]) for r in summaries])
    freeze(BASE/'statistics.json',dict(at=now(),seed=SEED,resamples=20000,n_cases=35,summaries=summaries,
        intervals='unadjusted exploratory pairedcase mean percentile; not trainingseed uncertainty'))
    return verified(BASE/'statistics.json')


def independently_verify():
    if (BASE/'independent_verification.json').exists():return
    cases=verified(BASE/'cases.json');raw=list(csv.DictReader((BASE/'metrics.csv').open()))
    lookup={(r['method'],r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in raw}
    idx=np.random.default_rng(SEED).integers(0,35,(20000,35));weights=np.array([np.bincount(x,minlength=35) for x in idx],dtype=float)
    maximum=0.;stat=verified(BASE/'statistics.json')
    def percentile(v,q):
        v=sorted(v);position=(len(v)-1)*q;i=int(position);f=position-i;return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
    for r in stat['summaries']:
        vector=[]
        for case in cases:
            b=[float(lookup[r['model'],case,r['condition'],d,r['stage']][r['metric']]) for d in r['draws']]
            if r['metric']=='D':v=sum(b)/len(b)
            else:
                a=[float(lookup[r['reference'],case,r['condition'],d,r['stage']][r['metric']]) for d in r['draws']]
                v=sum(bb-aa for aa,bb in zip(a,b))/len(a)
            vector.append(v)
        x=np.array(vector);bs=weights@x/35
        for a,b in [(sum(vector)/35,r['mean']),(percentile(vector,.5),r['median']),(percentile(bs,.025),r['ci95'][0]),(percentile(bs,.975),r['ci95'][1])]:
            maximum=max(maximum,abs(a-b));assert abs(a-b)<1e-10,(r,a,b)
        assert sum(v>1e-8 for v in vector)==r['increased'] and sum(v < -1e-8 for v in vector)==r['decreased']
    for r in verified(BASE/'prediction_manifest.json'):assert sha(r['prediction'])==r['prediction_sha256']
    for m in verified(BASE/'model_manifest.json'):assert sha(m['checkpoint'])==m['checkpoint_sha256']
    freeze(BASE/'independent_verification.json',dict(at=now(),passed=True,summary_vectors=len(stat['summaries']),
        bootstrap='independent multinomial count weights with independently sorted interpolated percentiles',
        maximum_difference=maximum,prediction_hashes_verified=1680,GT64_read_in_verification=False,GT128_read=False))


