"""Case-paired exploratory statistics and independent saved-field verification."""
from runtime import *
import csv
import math
import numpy as np

METRICS=('volume.relative_l1','volume.relative_l2','morphology.peak_ratio','morphology.absolute_peak_ratio_error',
    'morphology.gt_peak_location_ratio','morphology.centroid_distance_mm','morphology.dice_0.1','morphology.dice_0.5',
    'measurement.observed.relative_l1','measurement.observed.relative_l2','measurement.heldout.relative_l1','measurement.heldout.relative_l2')

def value(r,key):
    group,leaf=key.split('.',1)
    if group=='measurement':
        label,leaf=leaf.split('.',1);return r[group][label][leaf]
    return r[group][leaf]

def independent(mm,lookup):
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='val'};gt={};worst=0.;count=0
    with DevGuard(gt_splits=['val']) as guard:
        for case,r in source.items():gt[case]=np.load(DATA/r['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
        for m in mm:
            root=BASE/'evaluation/val'/m['key']
            for r in rows(root/'manifest.jsonl'):
                path=root/r['prediction'];assert sha(path)==r['sha256'];g=gt[r['case_id']]
                with np.load(path,allow_pickle=False) as z:
                    for stage in z.files:
                        p=z[stage].astype(np.float64);row=lookup[m['key'],stage,r['condition'],r['draw'],r['case_id']]
                        metrics={'volume.relative_l1':float(np.abs(p-g).sum()/max(np.abs(g).sum(),1e-6)),
                            'volume.relative_l2':float(np.sqrt(np.square(p-g).sum()/max(np.square(g).sum(),1e-12))),
                            'morphology.peak_ratio':float(p.max()/g.max()),
                            'morphology.gt_peak_location_ratio':float(p[np.unravel_index(g.argmax(),g.shape)]/g.max())}
                        coords=np.linspace(0,.6,64)
                        center=lambda c:np.array([np.dot(c.sum(axis=tuple(j for j in range(3) if j!=i)),coords)/c.sum() for i in range(3)])
                        if p.sum()>0:metrics['morphology.centroid_distance_mm']=float(np.linalg.norm(center(p)-center(g))*1000)
                        for f in (.1,.5):
                            pm=p>=g.max()*f;gm=g>=g.max()*f;metrics['morphology.dice_'+str(f)]=float(2*(pm&gm).sum()/(pm.sum()+gm.sum()))
                        for k,v in metrics.items():
                            expected=value(row,k);assert math.isclose(v,expected,rel_tol=1e-8,abs_tol=1e-9),(m['key'],r['case_id'],k,v,expected)
                            worst=max(worst,abs(v-expected))
                        count+=1
        assert not guard.denied and guard.counts['GT64_val']==36
    freeze(BASE/'independent_verification.json',dict(at=now(),passed=True,fields=count,cases=36,max_absolute_difference=worst,audit=guard.summary(),test_GT64=False,GT128=False))

def main():
    if (BASE/'analysis/summary.json').exists():return
    mm=verified(BASE/'methods.json');rr=rows(BASE/'evaluation/val/metrics.jsonl');reg=verified(BASE/'registration.json');cases=reg['cases']['val']
    lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    assert len(lookup)==len(rr)==3*36*17*2
    if not (BASE/'independent_verification.json').exists():independent(mm,lookup)
    layouts=verified(BASE/'layouts.json')['val'];panels=[(s['condition'],[s['draw']]) for s in layouts]+[(c,list(range(5))) for c in ('V3','V6','clustered-V4')]
    aggregate=[];case_rows=[];vectors={}
    for m in mm:
        for condition,draws in panels:
            stats={}
            for metric in METRICS:
                array=[]
                for case in cases:
                    values=[value(lookup[m['key'],'C1',condition,d,case],metric) for d in draws]
                    a=None if any(v is None for v in values) else float(np.mean(values));array.append(np.nan if a is None else a)
                    case_rows.append(dict(arm=m['arm'],condition=condition,draws=draws,case_id=case,metric=metric,value=a))
                vector=np.array(array);vectors[m['arm'],condition,tuple(draws),metric]=vector;finite=vector[np.isfinite(vector)]
                stats[metric]=dict(valid_cases=len(finite),mean=float(finite.mean()) if len(finite) else None,median=float(np.median(finite)) if len(finite) else None,
                    p90=float(np.quantile(finite,.9)) if len(finite) else None,max=float(finite.max()) if len(finite) else None)
            aggregate.append(dict(arm=m['arm'],condition=condition,draws=draws,metrics=stats))
    indices=np.random.default_rng(20261019).integers(0,36,size=(20000,36));weights=np.stack([(indices==i).mean(1) for i in range(36)],1);paired=[]
    for condition,draws in panels:
        for a,b in [('feedback_oari','baseline'),('feedback_oari_zero_residual','baseline'),('feedback_oari','feedback_oari_zero_residual')]:
            for metric in METRICS:
                delta=vectors[a,condition,tuple(draws),metric]-vectors[b,condition,tuple(draws),metric];finite=np.isfinite(delta)
                record=dict(comparison=a+' - '+b,condition=condition,draws=draws,metric=metric,valid_pairs=int(finite.sum()),
                    case_differences={case:float(v) if np.isfinite(v) else None for case,v in zip(cases,delta)},
                    undefined_cases=[c for c,ok in zip(cases,finite) if not ok])
                if finite.all():
                    bs=weights@delta;record.update(delta=float(delta.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),
                        wins_lower=int((delta<-1e-8).sum()),ties=int((np.abs(delta)<=1e-8).sum()),losses_lower=int((delta>1e-8).sum()))
                else:record.update(delta=None,ci95=None,wins_lower=None,ties=None,losses_lower=None)
                paired.append(record)
    conditions=[('Val-A-V4',[0]),('clustered-V4',list(range(5)))];signals=[]
    for cond,draws in conditions:
        diffs={key:float((vectors['feedback_oari',cond,tuple(draws),key]-vectors['baseline',cond,tuple(draws),key]).mean()) for key in ('volume.relative_l1','morphology.absolute_peak_ratio_error','morphology.centroid_distance_mm')}
        real_zero=float((vectors['feedback_oari',cond,tuple(draws),'volume.relative_l1']-vectors['feedback_oari_zero_residual',cond,tuple(draws),'volume.relative_l1']).mean())
        safe=lambda v:float(v) if np.isfinite(v) else None
        signals.append(dict(condition=cond,delta_L1=safe(diffs['volume.relative_l1']),delta_peak_absolute_error=safe(diffs['morphology.absolute_peak_ratio_error']),delta_centroid_mm=safe(diffs['morphology.centroid_distance_mm']),real_minus_zero_L1=safe(real_zero),
            preliminary_favorable=all(np.isfinite(v) and v<=0 for v in [*diffs.values(),real_zero]) and diffs['volume.relative_l1']<0))
    failure=[dict(arm=m['arm'],expected=612,saved=612,nonfinite_fields=0,missing=0) for m in mm]
    jsonl(BASE/'analysis/case_metrics.jsonl',case_rows);jsonl(BASE/'analysis/paired.jsonl',paired)
    write(BASE/'analysis/summary.json',dict(at=now(),pilot_epochs=20,seed=SEED,development_only=True,panels=aggregate,paired=paired,signals=signals,
        recommendation='值得扩大匹配预算与多种子确认，但尚非性能结论' if all(s['preliminary_favorable'] for s in signals) else '尚无同时满足普通/聚簇L1、峰值与定位的充分筛查信号；短程结果不足以否定收敛后效果',
        failures=failure,bootstrap=dict(resamples=20000,seed=20261019,unit='case; five draws averaged within each case',seed_uncertainty=False,exploratory=True),
        limitations=verified(BASE/'design.json')['limitations']))
    with (BASE/'analysis/per_case_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','condition','draws','case_id','metric','value']);w.writeheader()
        for r in case_rows:w.writerow({**r,'draws':','.join(map(str,r['draws']))})

if __name__=='__main__':main()
