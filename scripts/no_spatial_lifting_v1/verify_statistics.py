"""Independent absolute-CSV regrouping and weighted bootstrap; no GT access."""
from runtime import *
import csv


def main():
    verified(BASE/'all_predictions_sealed.json');s=verified(BASE/'statistics.json')
    with (BASE/'per_case_metrics.csv').open() as f:rr=list(csv.DictReader(f))
    table={(int(r['seed']),r['condition'],int(r['draw']),r['case_id'],r['stage'],r['variant']):r for r in rr};assert len(table)==len(rr)==4760
    cases=verified(BASE/'cases.json');idx=np.random.default_rng(20261019).choice(35,(20000,35),replace=True)
    weights=np.stack([(idx==i).sum(1)/35 for i in range(35)],1);maximum=0.;undefined=0
    for p in s['panels']:
        seeds=SEEDS if isinstance(p['seed'],str) else (p['seed'],)
        draws=range(1 if p['condition'] in ('Val-A-V4','spread-V4') else 5)
        base=[];changed=[];delta=[];missing=False
        for case in cases:
            groups=[]
            for variant in ('full','ablation'):
                vals=[table[seed,p['condition'],draw,case,p['stage'],variant][p['metric']] for seed in seeds for draw in draws]
                if any(v=='' for v in vals):missing=True;break
                groups.append(sum(float(v) for v in vals)/len(vals))
            if missing:break
            base.append(groups[0]);changed.append(groups[1]);delta.append(groups[1]-groups[0])
        if missing:
            assert p['ci95'] is None;undefined+=1;continue
        a=np.array(delta);np.testing.assert_allclose(a,p['case_deltas'],atol=1e-12,rtol=1e-10)
        boot=weights@a;np.testing.assert_allclose(np.percentile(boot,[2.5,97.5]),p['ci95'],atol=1e-12,rtol=1e-10)
        if p['conservative_family14_interval'] is not None:np.testing.assert_allclose(np.quantile(boot,[.05/28,1-.05/28]),p['conservative_family14_interval'],atol=1e-12,rtol=1e-10)
        for actual,expected in [(a.mean(),p['mean_delta']),(np.median(a),p['median_delta']),(np.mean(base),p['full_mean']),(np.mean(changed),p['ablation_mean'])]:
            assert abs(actual-expected)<1e-12;maximum=max(maximum,abs(actual-expected))
        assert int((a>0).sum())==p['increased_cases'] and int((a<0).sum())==p['decreased_cases']
    freeze(BASE/'independent_statistics_verification.json',dict(at=now(),passed=True,panels_checked=len(s['panels']),undefined_panels_preserved=undefined,max_scalar_disagreement=maximum,
        n_cases=35,resamples=20000,seed=20261019,GT_read=False,method='independent absolute CSV grouping then subtraction, weighted bootstrap vs index bootstrap',
        evidence={str(BASE/n):sha(BASE/n) for n in ('statistics.json','per_case_metrics.csv')}))
    print('INDEPENDENT STATISTICS VERIFIED',flush=True)


if __name__=='__main__':main()
