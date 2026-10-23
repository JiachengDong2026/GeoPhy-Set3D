"""Independent formulas for frozen arrays, case aggregation and paired intervals."""
from common import *
import argparse
import numpy as np


def main(split,scope=None):
    base=OUT/'evaluation'/(scope or split);registration=verified(OUT/'registration.json');seal=verified(base/'all_predictions_sealed.json')
    verify_hashes(seal['manifests']);scored=rows(base/'metrics.jsonl');summary=read(base/'summary.json');lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in scored}
    gt_source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};cache={};counts=0;maxdiff=0.
    with AccessGuard(gt_splits=[split]) as guard:
        verify_hashes(read(base/'gt_access.json')['GT64_sha256_after_seal'])
        for manifest in seal['manifests']:
            target=Path(manifest).parent;m=next(m for m in seal['methods'] if (('classical' in str(target) and m['key'] in str(target)) or target.name==m['key']))
            rr=rows(manifest);assert len(rr)==len(registration['cases'][split])*17
            assert len({(r['condition'],r['draw'],r['case_id']) for r in rr})==len(rr)
            for r in rr:
                assert sha(target/r['prediction'])==r['sha256'];case=r['case_id']
                if case not in cache:cache[case]=np.load(DATA/gt_source[case]['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
                gt=cache[case]
                with np.load(target/r['prediction'],allow_pickle=False) as z:
                    for stage in z.files:
                        p=z[stage].astype(np.float64);delta=p-gt
                        l1=float(np.sum(np.abs(delta))/max(np.sum(np.abs(gt)),1e-6));l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(gt.ravel()),1e-6))
                        reference=lookup[m['key'],stage,r['condition'],r['draw'],case]
                        d=max(abs(l1-reference['volume']['relative_l1']),abs(l2-reference['volume']['relative_l2']));maxdiff=max(maxdiff,d);assert d<1e-10
                        # Also independently check spatial/Dice/peak semantics on fixed first case.
                        if case==registration['cases'][split][0]:
                            for f in (.1,.5):
                                a=p>=f*gt.max();b=gt>=f*gt.max();dice=2*np.count_nonzero(a&b)/(np.count_nonzero(a)+np.count_nonzero(b));assert abs(dice-reference['morphology'][f'dice_{f}'])<1e-12
                            assert abs(p.max()/gt.max()-reference['morphology']['peak_ratio'])<1e-12
                            if p.sum()>0:
                                pc=np.array([np.sum(p*np.arange(64).reshape(tuple(64 if j==i else 1 for j in range(3))))/p.sum() for i in range(3)])*.6/63
                                gc=np.array([np.sum(gt*np.arange(64).reshape(tuple(64 if j==i else 1 for j in range(3))))/gt.sum() for i in range(3)])*.6/63
                                assert abs(np.linalg.norm(pc-gc)*1000-reference['morphology']['centroid_distance_mm'])<1e-9
                        counts+=1
    cases=registration['cases'][split];metrics={}
    # Development baseline rows can come from verified historical metrics, never final test.
    if split=='val':
        for key,path in [('MRGR',OLD_EVAL/'per_case_metrics.jsonl'),('LR4E4',ROOT/'outputs/research/mrgr_autonomous_optimization_v2/frozen_formal_evaluation'/LR.name/'per_case_metrics.jsonl')]:
            for r in rows(path):lookup[key,r['stage'],r['condition'],r['draw'],r['case_id']]=r
    for panel in summary['panels']:
        vals=[np.mean([lookup[panel['method'],panel['stage'],panel['condition'],d,c]['volume']['relative_l1'] for d in panel['draws']]) for c in cases]
        actual=panel['metrics']['volume.relative_l1'];assert abs(np.mean(vals)-actual['mean'])<1e-12 and abs(np.quantile(vals,.9)-actual['p90'])<1e-12
    # Separate resampling implementation streams replicate the predeclared RNG exactly.
    rng=np.random.default_rng(registration['statistics']['seed']);indices=np.stack([rng.integers(0,len(cases),size=len(cases)) for _ in range(registration['statistics']['resamples'])])
    for pair in summary['paired']:
        diffs=[]
        for c in cases:
            a=np.mean([lookup['MRGR','C1',pair['condition'],d,c]['volume'][pair['metric']] for d in pair['draws']]);b=np.mean([lookup[pair['reference'],'C1',pair['condition'],d,c]['volume'][pair['metric']] for d in pair['draws']]);diffs.append(a-b)
        resampled=np.array(diffs)[indices].mean(axis=1);ci=np.percentile(resampled,[2.5,97.5]);adj=np.percentile(resampled,[2.5/pair['comparisons'],100-2.5/pair['comparisons']])
        assert np.max(np.abs(ci-pair['ci95']))<1e-12;assert np.max(np.abs(adj-pair['simultaneous_ci']))<1e-12
    ablation_count=0
    ablation_path=base/'ablation_paired.json'
    if ablation_path.exists():
        for row in read(ablation_path)['results']:
            deltas=[]
            for seed in ((20261011,20261019) if isinstance(row['seed'],str) else (row['seed'],)):
                reference='MRGR' if seed==20261011 else 'multiscale_seed20261019'
                treatment=f'{row["arm"]}_seed{seed}'
                deltas.append(np.array([np.mean([lookup[treatment,'C1',row['condition'],d,c]['volume'][row['metric']]-lookup[reference,'C1',row['condition'],d,c]['volume'][row['metric']] for d in row['draws']]) for c in cases]))
            delta=np.mean(deltas,axis=0);ci=np.percentile(delta[indices].mean(1),[2.5,97.5])
            assert abs(delta.mean()-row['delta_ablation_minus_full'])<1e-12
            assert np.max(np.abs(ci-row['ci95_case']))<1e-12;ablation_count+=1
    write(base/'independent_verification.json',dict(at=now(),passed=True,fields_recomputed=counts,max_L1_L2_absolute_difference=maxdiff,panels_verified=len(summary['panels']),paired_intervals_verified=len(summary['paired']),within_seed_ablation_comparisons_verified=ablation_count,audit=guard.summary(),source_sha256=sha(__file__),no_training_seed_uncertainty_claim=True))
    print(f'Independent verification PASSED: {counts} arrays, {len(summary["paired"])} paired intervals',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--scope');a=p.parse_args();main(a.split,a.scope)
