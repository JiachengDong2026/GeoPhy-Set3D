"""Use recorded metadata only; missing source_mode is retained as null."""
import json
import numpy as np
from .analysis import correlation

FIELDS=('geometry_id','source_family','wind_speed_m_s','pressure_pa','active_hole_count','condition_id','source_mode')


def stratify(rows, small_n_threshold=3):
    result=[]
    for field in FIELDS:
        if not any(field in r['condition_meta'] for r in rows): continue
        categories=sorted({json.dumps(r['condition_meta'].get(field),sort_keys=True) for r in rows})
        for category in categories:
            group=[r for r in rows if json.dumps(r['condition_meta'].get(field),sort_keys=True)==category]
            result.append(dict(field=field,value=category,case_count=len(group),small_n=len(group)<small_n_threshold,
                               mean_relative_l1=float(np.mean([r['relative_l1'] for r in group])),
                               median_relative_l1=float(np.median([r['relative_l1'] for r in group])),
                               mean_relative_l2=float(np.mean([r['relative_l2'] for r in group])),
                               mean_rmse=float(np.mean([r['rmse'] for r in group]))))
    continuous=[]
    for field in ('wind_speed_m_s','pressure_pa'):
        numeric=[dict(r,**{field:r['condition_meta'].get(field)}) for r in rows
                 if isinstance(r['condition_meta'].get(field),(float,int))]
        for metric in ('relative_l1','relative_l2'): continuous.append(correlation(numeric,field,metric))
    return result,continuous
