"""Paired case statistics; development exploration, conditional on two seeds."""
from runtime import *
import csv
import numpy as np

METRICS=('volume.relative_l1','volume.relative_l2','morphology.peak_ratio','morphology.absolute_peak_ratio_error',
    'morphology.gt_peak_location_ratio','morphology.centroid_distance_mm','morphology.dice_0.1','morphology.dice_0.5',
    'volume.peak_location_error_mm','volume.centroid_error_mm','morphology.background_predicted_fraction',
    'measurement.observed.relative_l1','measurement.observed.relative_l2','measurement.heldout.relative_l1','measurement.heldout.relative_l2')

def scalar(row,key):
    # Metric names such as dice_0.1 contain a dot in the leaf.
    group,leaf=key.split('.',1)
    if group=='measurement':
        label,leaf=leaf.split('.',1);return row.get(group,{}).get(label,{}).get(leaf)
    return row.get(group,{}).get(leaf)

def main():
    target=BASE/'analysis';target.mkdir(exist_ok=True)
    if (target/'summary.json').exists():return
    verify_hashes(verified(BASE/'reuse_audit.json')['evidence'])
    rr=rows(BASE/'reuse_metrics.jsonl')+rows(BASE/'evaluation/val/metrics.jsonl')
    cases=verified(BASE/'registration.json')['cases']['val'];layouts=verified(BASE/'layouts.json')['val']
    methods=verified(BASE/'reuse_methods.json')+verified(BASE/'methods.json')
    lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    assert len(lookup)==len(rr)==len(methods)*36*17*2
    panels=[(s['condition'],[s['draw']]) for s in layouts]
    panels.extend((condition,list(range(5))) for condition in ('V3','V6','clustered-V4'))
    aggregate=[];case_rows=[];vectors={}
    for info in methods:
        for stage in ('C0','C1'):
            for condition,draws in panels:
                values=[]
                for case in cases:
                    records=[lookup[info['key'],stage,condition,d,case] for d in draws];row={}
                    for metric in METRICS:
                        vv=[scalar(r,metric) for r in records]
                        row[metric]=None if any(v is None for v in vv) else float(np.mean(vv))
                    case_rows.append(dict(method=info['key'],seed=info['seed'],stage=stage,condition=condition,draws=draws,case_id=case,metrics=row));values.append(row)
                stats={}
                for key in METRICS:
                    a=np.array([v[key] for v in values if v[key] is not None])
                    stats[key]=dict(valid_cases=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                        p90=float(np.quantile(a,.9)) if len(a) else None,max=float(a.max()) if len(a) else None)
                    if stage=='C1':vectors[info['arm'],info['seed'],condition,tuple(draws),key]=np.array([v[key] if v[key] is not None else np.nan for v in values])
                aggregate.append(dict(method=info['key'],seed=info['seed'],stage=stage,condition=condition,draws=draws,metrics=stats))
    rng=np.random.default_rng(20261019);indices=rng.integers(0,36,size=(20000,36));weights=np.stack([(indices==i).mean(1) for i in range(36)],1)
    expressions={'B-A':{'uniform_fusion':1,'multiscale':-1},'C-A':{'pruned_observability':1,'multiscale':-1},
        'D-B':{'uniform_pruned_observability':1,'uniform_fusion':-1},'D-C':{'uniform_pruned_observability':1,'pruned_observability':-1},
        '(D-C)-(B-A)':{'uniform_pruned_observability':1,'pruned_observability':-1,'uniform_fusion':-1,'multiscale':1},
        'C-zero_input':{'pruned_observability':1,'no_observability':-1}}
    paired=[]
    for condition,draws in panels:
        for metric in METRICS:
            for label,expression in expressions.items():
                deltas=[sum(coef*vectors[arm,seed,condition,tuple(draws),metric] for arm,coef in expression.items()) for seed in SEEDS]
                for seed,delta in [*zip(SEEDS,deltas),('two_seed_case_mean',np.mean(deltas,axis=0))]:
                    assert np.isfinite(delta).all(),(condition,label,metric)
                    bs=weights@delta
                    paired.append(dict(comparison=label,seed=seed,condition=condition,draws=draws,metric=metric,delta=float(delta.mean()),
                        ci95=np.quantile(bs,[.025,.975]).tolist(),case_differences=dict(zip(cases,delta.tolist())),
                        negative=int((delta<-1e-8).sum()),tie=int((np.abs(delta)<=1e-8).sum()),positive=int((delta>1e-8).sum()),
                        interpretation='negative is lower; higher is better for Dice and GT-peak ratio near1; raw peak ratio target1; interaction sign is exploratory',
                        seed_population_uncertainty_included=False))
    failures=[];worst=[]
    for info in methods:
        selected=[r for r in rr if r['method']==info['key'] and r['stage']=='C1']
        assert len(selected)==612
        failures.append(dict(method=info['key'],expected=612,reconstructions=len(selected),missing=0,nonfinite_volume_metrics=sum(not np.isfinite(r['volume']['relative_l1']) for r in selected),failure_rate=0.))
        for condition,draws in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
            vals=vectors[info['arm'],info['seed'],condition,tuple(draws),'volume.relative_l1']
            worst.append(dict(method=info['key'],condition=condition,cases=[dict(case_id=cases[i],l1=float(vals[i])) for i in np.argsort(vals)[-5:][::-1]]))
    jsonl(target/'case_panels.jsonl',case_rows);jsonl(target/'paired_statistics.jsonl',paired);jsonl(target/'all_metrics.jsonl',rr)
    write(target/'summary.json',dict(at=now(),development_only=True,cases=36,seeds=list(SEEDS),layouts=17,panels=aggregate,failures=failures,worst_cases=worst,
        paired_file=str(target/'paired_statistics.jsonl'),statistics=verified(BASE/'design.json')['statistics'],
        caveat='All intervals are exploratory and conditional on the two observed seeds. Multiple layouts/metrics are not independent confirmatory tests; zero-input is not pruning.'))
    with (target/'per_case_metrics.csv').open('w') as f:
        fields=['method','seed','stage','condition','draws','case_id',*METRICS];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in case_rows:w.writerow(dict(method=r['method'],seed=r['seed'],stage=r['stage'],condition=r['condition'],draws=','.join(map(str,r['draws'])),case_id=r['case_id'],**r['metrics']))
    with (target/'paired_statistics.csv').open('w') as f:
        fields=['comparison','seed','condition','draws','metric','delta','ci_low','ci_high','negative','tie','positive'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in paired:w.writerow({**{k:r[k] for k in fields if k not in ('draws','ci_low','ci_high')},'draws':','.join(map(str,r['draws'])),'ci_low':r['ci95'][0],'ci_high':r['ci95'][1]})

if __name__=='__main__':main()
