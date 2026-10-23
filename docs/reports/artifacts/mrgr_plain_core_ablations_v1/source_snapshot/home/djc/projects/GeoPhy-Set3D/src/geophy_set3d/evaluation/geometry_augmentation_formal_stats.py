"""Preregistered case-unit paired statistics and exact single-intervention decision."""
import numpy as np
from scipy.stats import wilcoxon

METRICS=('relative_l1','relative_l2','mae','rmse','peak_value_absolute_error','integrated_concentration_relative_error',
         'observed_relative_l1','observed_relative_l2','heldout_relative_l1','heldout_relative_l2')

def paired(c,t,seed=2026101603,n=10000):
    c=np.asarray(c,float);t=np.asarray(t,float)
    if c.shape!=t.shape or c.ndim!=1 or len(c)<2:raise ValueError('One matched observation per case required')
    d=t-c;idx=np.random.default_rng(seed).integers(0,len(c),(n,len(c)))
    ci=np.quantile(d[idx].mean(1),[.025,.975]);ratio=float(t.mean()/c.mean())
    rci=np.quantile(t[idx].mean(1)/c[idx].mean(1),[.025,.975])
    return dict(n_cases=len(c),Control_mean=float(c.mean()),Treatment_mean=float(t.mean()),mean_delta=float(d.mean()),median_delta=float(np.median(d)),
        delta_ci95=ci.tolist(),ratio=ratio,ratio_ci95=rci.tolist(),improvement=1-ratio,Treatment_wins=int((d<0).sum()),Control_wins=int((d>0).sum()),ties=int((d==0).sum()),
        wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d) else 1.)

def decide(g5,natural,ni=1.03,partial_max=1.10,partial_effect=.10):
    significant=g5['delta_ci95'][1]<0
    if significant and natural['ratio']<=ni:return 'GO'
    spans=g5['delta_ci95'][0]<=0<=g5['delta_ci95'][1]
    if natural['ratio']<=partial_max and (significant or (g5['improvement']>=partial_effect and spans)):return 'PARTIAL'
    return 'NO-GO'

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);v=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result=np.empty_like(p);result[order]=np.minimum(v,1.);return result.tolist()

def worst(x,ids):
    x=np.asarray(x,float);return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.90)),p95=float(np.quantile(x,.95)),maximum=float(x.max()),worst_case_ids=[str(ids[i]) for i in np.flatnonzero(x==x.max())])
