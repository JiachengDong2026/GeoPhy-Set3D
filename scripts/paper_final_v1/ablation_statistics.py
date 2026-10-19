"""Within-training-seed paired ablations; case bootstrap never substitutes seed variance."""
from common import *
import argparse
import numpy as np


def main(split):
    base=OUT/'evaluation'/split;rr=rows(base/'metrics.jsonl')
    if split=='val':
        for r in rows(OLD_EVAL/'per_case_metrics.jsonl'):rr.append(dict(r,method='MRGR'))
    lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr};reg=verified(OUT/'registration.json');cases=reg['cases'][split]
    rng=np.random.default_rng(reg['statistics']['seed']);indices=rng.integers(0,len(cases),(20000,len(cases)))
    result=[]
    from module_registry_v2 import ablation_arms
    for arm in ablation_arms():
        for condition,draws in [('Val-A-V4',[0]),('spread-V4',[0]),('V3',list(range(5))),('V6',list(range(5))),('clustered-V4',list(range(5)))]:
            for metric in ('relative_l1','relative_l2'):
                seed_differences=[]
                for seed in (20261011,20261019):
                    reference='MRGR' if seed==20261011 else 'multiscale_seed20261019';treatment=f'{arm}_seed{seed}'
                    a=np.array([np.mean([lookup[reference,'C1',condition,d,c]['volume'][metric] for d in draws]) for c in cases]);b=np.array([np.mean([lookup[treatment,'C1',condition,d,c]['volume'][metric] for d in draws]) for c in cases]);delta=b-a;seed_differences.append(delta)
                    result.append(dict(arm=arm,seed=seed,reference=reference,treatment=treatment,condition=condition,draws=draws,metric=metric,mean_full=float(a.mean()),mean_ablation=float(b.mean()),delta_ablation_minus_full=float(delta.mean()),ci95_case=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),case_differences=dict(zip(cases,delta.tolist()))))
                mean_delta=np.mean(seed_differences,axis=0)
                result.append(dict(arm=arm,seed='two_seed_average_descriptive',condition=condition,draws=draws,metric=metric,delta_ablation_minus_full=float(mean_delta.mean()),ci95_case=np.quantile(mean_delta[indices].mean(1),[.025,.975]).tolist(),case_differences=dict(zip(cases,mean_delta.tolist())),seed_uncertainty_estimated=False))
    write(base/'ablation_paired.json',dict(at=now(),split=split,results=result,unit='paired training seed and case; draws averaged within case',inference='exploratory; two seeds do not establish population seed variance',source_sha256=sha(__file__)))
    print('Within-seed paired ablation statistics complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();main(a.split)
