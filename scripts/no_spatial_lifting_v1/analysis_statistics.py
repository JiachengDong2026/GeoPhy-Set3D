"""Predeclared paired case bootstrap; draws and seeds never independent samples."""
from runtime import *
import csv
METRICS=('relative_l1','relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm')


def csv_write(path,records):
    assert not path.exists()
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');w.writeheader();w.writerows(records)


def summarize(raw):
    cases=verified(BASE/'cases.json');paired=[];aggregates=[];panels=[]
    lookup={(r['seed'],r['condition'],r['draw'],r['case_id'],r['stage'],r['variant']):r for r in raw};assert len(lookup)==len(raw)==4760
    for r in raw:
        if r['variant']!='ablation':continue
        b=lookup[r['seed'],r['condition'],r['draw'],r['case_id'],r['stage'],'full']
        paired.append(dict(seed=r['seed'],condition=r['condition'],draw=r['draw'],case_id=r['case_id'],stage=r['stage'],
            **{f'full_{k}':b[k] for k in METRICS},**{f'ablation_{k}':r[k] for k in METRICS},
            **{f'delta_{k}':None if b[k] is None or r[k] is None else r[k]-b[k] for k in METRICS},
            delta_relative_l1_percentage_points=(r['relative_l1']-b['relative_l1'])*100))
    idx=np.random.default_rng(20261019).integers(0,35,size=(20000,35))
    for seed in (*SEEDS,'two_fixed_seed_mean_descriptive'):
        seeds=SEEDS if isinstance(seed,str) else (seed,)
        for condition in CONDITIONS:
            for stage in ('C0','C1'):
                for metric in METRICS:
                    values=[];full=[];ablation=[];nd=1 if condition in ('Val-A-V4','spread-V4') else 5
                    for case in cases:
                        rr=[r for r in paired if r['case_id']==case and r['seed'] in seeds and r['condition']==condition and r['stage']==stage]
                        assert len(rr)==nd*len(seeds)
                        cols=[]
                        for label in ('full','ablation','delta'):
                            vv=[r[f'{label}_{metric}'] for r in rr]
                            cols.append(None if any(v is None for v in vv) else float(np.mean(vv)))
                        b,a,d=cols;values.append(d);full.append(b);ablation.append(a)
                        aggregates.append(dict(seed=seed,condition=condition,stage=stage,metric=metric,case_id=case,full=b,ablation=a,delta=d,
                            delta_percentage_points=d*100 if d is not None and metric=='relative_l1' else None,draws_per_seed=nd))
                    defined=all(v is not None for v in values);a=np.array(values,dtype=float) if defined else None
                    boot=a[idx].mean(1) if defined else None
                    primary=condition=='Val-A-V4' and stage=='C1' and metric=='relative_l1'
                    ci=np.quantile(boot,[.025,.975]).tolist() if defined else None
                    corrected=np.quantile(boot,[.05/(2*14),1-.05/(2*14)]).tolist() if defined and primary else None
                    panels.append(dict(seed=seed,condition=condition,stage=stage,metric=metric,primary_endpoint=primary,
                        important_secondary=condition=='clustered-V4' and stage=='C1' and metric=='relative_l1',exploratory=True,
                        n_cases=35,defined_cases=sum(v is not None for v in values),draws_per_seed=nd,case_deltas=values,
                        full_mean=float(np.mean(full)) if defined else None,ablation_mean=float(np.mean(ablation)) if defined else None,
                        mean_delta=float(a.mean()) if defined else None,median_delta=float(np.median(a)) if defined else None,ci95=ci,
                        conservative_family14_interval=corrected,increased_cases=int((a>0).sum()) if defined else None,decreased_cases=int((a<0).sum()) if defined else None,
                        mean_delta_percentage_points=float(a.mean()*100) if defined and metric=='relative_l1' else None,
                        ci95_percentage_points=[v*100 for v in ci] if defined and metric=='relative_l1' else None))
    csv_write(BASE/'paired_case_metrics.csv',paired);csv_write(BASE/'case_aggregates.csv',aggregates)
    csv_write(BASE/'paired_statistics.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in panels])
    freeze(BASE/'statistics.json',dict(at=now(),panels=panels,primary=[p for p in panels if p['primary_endpoint']],n_cases=35,resamples=20000,bootstrap_seed=20261019,
        difference='ablation-full; positive error delta means full model lower error; Dice negative delta means deterioration',
        aggregation='draw mean within case/seed/layout; report each fixed seed; descriptive2fixedseed mean within case; bootstrap35cases only',
        independent_confirmation=False,seed_population_variance_estimated=False,historical_family14_unchanged=True))
