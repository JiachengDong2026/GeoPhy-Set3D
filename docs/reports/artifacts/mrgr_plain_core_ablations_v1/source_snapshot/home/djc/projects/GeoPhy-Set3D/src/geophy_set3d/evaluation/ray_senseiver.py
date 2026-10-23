"""Three frozen methods; physical input identity independent of lifting representation."""
import ast
import numpy as np
import pandas as pd
from .selfsup_c0c1_diagnostic import METRICS, GROUPS
from .deep_comparison import KEYS

MODELS=['Ours','DeepSets','Senseiver']
PAIRS=[('Ours','Senseiver','primary'),('Senseiver','DeepSets','auxiliary')]


def pair_three(senseiver, historical, deep, records):
    if any(len(x)!=900 for x in [senseiver,historical,deep,records]):raise ValueError('Full 900 identity required')
    index=['case_id','group','draw'];expected={(r['case_id'],r['group'],r['draw']):r for r in records}
    frames=[]
    for frame in [senseiver,historical,deep]:
        if frame.duplicated(index).any():raise ValueError('Duplicate identity')
        f=frame.set_index(index)
        if set(f.index)!=set(expected):raise ValueError('Missing or foreign identity')
        frames.append(f)
    rows=[]
    for key,r in expected.items():
        s,h,d=[f.loc[key] for f in frames]
        for row in [s,h,d]:
            for name in ['station_ids','H_station_ids']:
                ids=ast.literal_eval(row[name]) if isinstance(row[name],str) else row[name]
                if ids!=r[name]:raise ValueError('Station order changed')
            for name in ['case_index','geometry_id']:
                if row[name]!=r[name]:raise ValueError('Case/layout identity changed')
        for name in ['complete_input_sha256','complete_H_sha256']:
            if s[name]!=d[name]:raise ValueError('Complete physical information changed: '+name)
        if s['input_identity_sha256']!=h['input_identity_sha256'] or s['H_mask_sha256']!=h['H_mask_sha256']:
            raise ValueError('Historical physical input mismatch')
        if d['input_identity_sha256']!=h['input_identity_sha256'] or d['H_mask_sha256']!=h['H_mask_sha256']:
            raise ValueError('Historical reference chain mismatch')
        if d['prediction_sha256']!=h['deepsets_prediction_sha256'] or d['checkpoint_sha256']!=h['deepsets_checkpoint_sha256']:
            raise ValueError('DeepSets historical prediction mismatch')
        row={k:r[k] for k in KEYS+['station_ids','H_station_ids']}
        for name in ['complete_input_sha256','complete_H_sha256','input_identity_sha256','H_mask_sha256','input_ray_view_sha256']:row[name]=s[name]
        row.update(ours_checkpoint_sha256=h['ours_checkpoint_sha256'],deepsets_checkpoint_sha256=d['checkpoint_sha256'],senseiver_checkpoint_sha256=s['checkpoint_sha256'],senseiver_prediction_sha256=s['prediction_sha256'],ours_prediction_sha256=h['ours_prediction_sha256'],deepsets_prediction_sha256=d['prediction_sha256'])
        for metric in METRICS:
            row['Ours_'+metric]=float(h['Ours_'+metric]);row['DeepSets_'+metric]=float(d[metric]);row['Senseiver_'+metric]=float(s[metric])
            if not np.isfinite([row[n+'_'+metric] for n in MODELS]).all():raise ValueError('Nonfinite metric')
        rows.append(row)
    return pd.DataFrame(rows)


def case_means(paired):
    for key,frame in paired.groupby(['case_id','case_index','group']):
        if sorted(frame.draw.tolist())!=list(range(5)):raise ValueError('Five distinct layouts required')
    cols=[n+'_'+m for n in MODELS for m in METRICS]
    result=paired.groupby(['case_id','case_index','group'],as_index=False)[cols].mean()
    if len(result)!=180:raise ValueError('Full 180 case/groups required')
    return result


def summarize(a,b,ix,epsilon):
    delta=a-b;lo,hi=np.quantile(delta[ix].mean(1),[.025,.975])
    return dict(left_mean=float(a.mean()),right_mean=float(b.mean()),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),ci_low=float(lo),ci_high=float(hi),left_wins=int((delta < -epsilon).sum()),right_wins=int((delta>epsilon).sum()),ties=int((abs(delta)<=epsilon).sum()),relative_reduction=float(1-a.mean()/b.mean()) if b.mean()!=0 else None)


def statistics(case,c):
    ix=np.random.default_rng(c['comparison']['bootstrap_seed']).integers(0,36,(10000,36));eps=c['comparison']['tie_epsilon']
    stats=[];tails=[];contrasts=[]
    for g in GROUPS:
        f=case[case.group==g].sort_values('case_id');assert len(f)==36
        for m in METRICS:
            for left,right,family in PAIRS:
                stats.append(dict(group=g,metric=m,left=left,right=right,family=family,cases=36,**summarize(f[left+'_'+m].to_numpy(),f[right+'_'+m].to_numpy(),ix,eps)))
            for name in MODELS:
                v=f[name+'_'+m].to_numpy();tails.append(dict(group=g,metric=m,model=name,mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),maximum=float(v.max()),worst_case_id=f.iloc[v.argmax()].case_id,worst_case_index=int(f.iloc[v.argmax()].case_index)))
    for m in METRICS:
        arrays={name:case.pivot(index='case_id',columns='group',values=name+'_'+m).sort_index()[GROUPS].to_numpy() for name in MODELS}
        for left,right,family in PAIRS:
            a,b=arrays[left],arrays[right]
            for label,func in [('G1_G4_macro',lambda x:x[:,:4].mean(1)),('G5_minus_G1',lambda x:x[:,4]-x[:,0])]:
                contrasts.append(dict(comparison=label,metric=m,left=left,right=right,family=family,**summarize(func(a),func(b),ix,eps)))
            av=a[:,4].mean()/a[:,0].mean();bv=b[:,4].mean()/b[:,0].mean()
            samples=a[ix,4].mean(1)/a[ix,0].mean(1)-b[ix,4].mean(1)/b[ix,0].mean(1);lo,hi=np.quantile(samples,[.025,.975])
            contrasts.append(dict(comparison='G5_div_G1_grand_mean_ratio',metric=m,left=left,right=right,family=family,left_mean=float(av),right_mean=float(bv),mean_delta=float(av-bv),ci_low=float(lo),ci_high=float(hi)))
    return pd.DataFrame(stats),pd.DataFrame(tails),contrasts
