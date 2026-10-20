"""Independent NumPy recomputation from every reused/new saved development field."""
from runtime import *
import math
import numpy as np

def main():
    target=BASE/'analysis/independent_verification.json'
    if target.exists():return verified(target)
    verify_hashes(verified(BASE/'reuse_audit.json')['evidence'])
    methods=verified(BASE/'reuse_methods.json')+verified(BASE/'methods.json')
    rr=rows(BASE/'analysis/all_metrics.jsonl');lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='val'}
    manifests={};allowed={}
    for m in methods:
        root=Path(m.get('prediction_root',BASE/'evaluation/val'/m['key']))
        manifest=Path(m.get('manifest',root/'manifest.jsonl'));items=rows(manifest)
        manifests[m['key']]=(root,items)
        allowed.update({str((root/r['prediction']).resolve()):'sealed_development_prediction' for r in items})
    cache={};errors={};count=0;recomputed={}
    with DevGuard(gt_splits=['val']) as guard:
        guard.allowed.update(allowed)
        for case,r in source.items():cache[case]=np.load(DATA/r['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
        for m in methods:
            root,items=manifests[m['key']]
            for item in items:
                path=root/item['prediction'];assert sha(path)==item['sha256']
                g=cache[item['case_id']];peak=float(g.max());idx=np.unravel_index(g.argmax(),g.shape)
                coords=np.linspace(0,.6,64)
                center=lambda c:np.array([(c.sum(axis=tuple(j for j in range(3) if j!=i))*coords).sum()/c.sum() for i in range(3)])
                gc=center(g)
                with np.load(path,allow_pickle=False) as fields:
                    assert set(fields.files)=={'C0','C1'}
                    for stage in fields.files:
                        p=fields[stage].astype(np.float64);assert p.shape==(64,)*3 and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                        row=lookup[m['key'],stage,item['condition'],item['draw'],item['case_id']]
                        values={'volume.relative_l1':float(np.abs(p-g).sum()/max(np.abs(g).sum(),1e-6)),
                            'volume.relative_l2':float(np.sqrt(np.square(p-g).sum()/max(np.square(g).sum(),1e-12))),
                            'morphology.peak_ratio':float(p.max()/peak),'morphology.gt_peak_location_ratio':float(p[idx]/peak)}
                        if p.sum()>0:values['morphology.centroid_distance_mm']=float(np.linalg.norm(center(p)-gc)*1000)
                        for f in (.1,.5):
                            a=p>=peak*f;b=g>=peak*f;values['morphology.dice_'+str(f)]=float(2*(a&b).sum()/(a.sum()+b.sum()))
                        for key,value in values.items():
                            group,leaf=key.split('.',1);difference=abs(value-row[group][leaf]);errors[key]=max(errors.get(key,0.),difference)
                            assert math.isclose(value,row[group][leaf],rel_tol=1e-8,abs_tol=1e-9),(m['key'],key,difference)
                        recomputed[m['arm'],m['seed'],stage,item['condition'],item['draw'],item['case_id']]=values['volume.relative_l1'];count+=1
            print('Independent fields checked',m['key'],count,flush=True)
        assert not guard.denied and guard.counts['GT64_val']==36
    cases=verified(BASE/'registration.json')['cases']['val'];pairs=rows(BASE/'analysis/paired_statistics.jsonl')
    expressions={'B-A':{'uniform_fusion':1,'multiscale':-1},'C-A':{'pruned_observability':1,'multiscale':-1},
        'D-B':{'uniform_pruned_observability':1,'uniform_fusion':-1},'D-C':{'uniform_pruned_observability':1,'pruned_observability':-1},
        '(D-C)-(B-A)':{'uniform_pruned_observability':1,'pruned_observability':-1,'uniform_fusion':-1,'multiscale':1},
        'C-zero_input':{'pruned_observability':1,'no_observability':-1}}
    indices=np.random.default_rng(20261019).integers(0,36,size=(20000,36));checks=0
    for row in pairs:
        if row['metric']!='volume.relative_l1':continue
        seeds=SEEDS if row['seed']=='two_seed_case_mean' else [row['seed']]
        delta=np.array([np.mean([sum(coef*np.mean([recomputed[arm,seed,'C1',row['condition'],d,case] for d in row['draws']]) for arm,coef in expressions[row['comparison']].items()) for seed in seeds]) for case in cases])
        np.testing.assert_allclose(delta,[row['case_differences'][c] for c in cases],rtol=1e-10,atol=1e-10)
        if row['condition']=='Val-A-V4' or row['condition']=='clustered-V4' and len(row['draws'])==5:
            ci=np.quantile(delta[indices].mean(1),[.025,.975]);np.testing.assert_allclose(ci,row['ci95'],atol=1e-10,rtol=1e-10);checks+=1
        assert abs(delta.mean()-row['delta'])<1e-10
    freeze(target,dict(at=now(),passed=True,fields=count,methods=len(methods),cases=36,layouts=17,independent_metrics_max_absolute_error=errors,
        bootstrap_intervals_recomputed=checks,all_L1_case_pairings_recomputed=True,audit=guard.summary(),test_GT64_read=False,GT128_read=False))

if __name__=='__main__':main()
