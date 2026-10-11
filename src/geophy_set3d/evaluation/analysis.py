"""Draw aggregates, physical GT descriptors and descriptive correlations."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import write_json

METRICS = ('mae','rmse','relative_l1','relative_l2')


def write_csv(path, rows):
    if not rows: raise ValueError('No CSV rows')
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def aggregate_draws(summaries, rows):
    metrics=VolumeMetrics();metrics.rows=rows
    result=metrics.compute()
    result['draw_count']=len(summaries)
    result['cases_per_draw']=summaries[0]['case_count']
    result['std_across_draw_means']={key:float(np.std([s['mean_per_case'][key] for s in summaries],ddof=1)) if len(summaries)>1 else 0. for key in METRICS}
    result['case_std_per_draw']={str(s['draw_id']):{key:float(np.std([r[key] for r in rows if r['draw_id']==s['draw_id']],ddof=1)) for key in METRICS} for s in summaries}
    return result


def plume_descriptors(volume, threshold=1e-5):
    g=np.asarray(volume,dtype=np.float64)
    total=float(g.sum());axis=np.linspace(0,.6,g.shape[0])
    center=[float(np.dot(g.sum(axis=tuple(j for j in range(3) if j!=i)),axis)/total) if total>0 else None for i in range(3)]
    return dict(gt_max_concentration=float(g.max()),gt_mean_concentration=float(g.mean()),
                nonzero_voxel_fraction=float(np.count_nonzero(g)/g.size),
                active_voxel_fraction=float(np.mean(g>threshold)),active_voxel_threshold=threshold,
                gt_integrated_concentration_proxy=float(g.mean()*.6**3),
                concentration_center_x_m=center[0],concentration_center_y_m=center[1],concentration_center_z_m=center[2])


def correlation(rows, x, y):
    pairs=[(r[x],r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None and np.isfinite(r[x]) and np.isfinite(r[y])]
    if len(pairs)<3 or len(set(a for a,b in pairs))<2 or len(set(b for a,b in pairs))<2:
        return dict(x=x,y=y,n=len(pairs),rho=None,p_value=None,reason='insufficient variation or sample count')
    statistic,p_value=spearmanr(*zip(*pairs))
    return dict(x=x,y=y,n=len(pairs),rho=float(statistic),p_value=float(p_value),interpretation='descriptive only; no multiple-comparison correction')


def metric_degradation(a,b):
    return {aggregation:{key:dict(a=a[aggregation][key],b=b[aggregation][key],
                                 absolute_delta=b[aggregation][key]-a[aggregation][key],
                                 relative_degradation=b[aggregation][key]/a[aggregation][key]-1)
                         for key in METRICS} for aggregation in ('mean_per_case','pooled')}
