"""Reusable fixed-input comparison tables and case-level paired statistics."""
import ast
import numpy as np
import pandas as pd
from .selfsup_c0c1_diagnostic import METRICS,GROUPS

KEYS=['case_id','case_index','group','draw','geometry_id']


def pair_reference(baseline,reference,records):
    """Identity, complete membership and ten metrics must agree before pairing."""
    if len(baseline)!=900 or len(reference)!=900 or len(records)!=900:
        raise ValueError('Full 36 x 25 comparison required')
    expected={(r['case_id'],r['group'],r['draw']):r for r in records}
    if len(expected)!=900:raise ValueError('Duplicate evaluation identity')
    if baseline.duplicated(['case_id','group','draw']).any() or reference.duplicated(['case_id','group','draw']).any():
        raise ValueError('Duplicate result row')
    b=baseline.set_index(['case_id','group','draw']);o=reference.set_index(['case_id','group','draw'])
    if set(b.index)!=set(expected) or set(o.index)!=set(expected):raise ValueError('Case/layout mismatch')
    rows=[]
    for key,r in expected.items():
        br=b.loc[key];ours=o.loc[key]
        for name,ids in [('station_ids',r['station_ids']),('H_station_ids',r['H_station_ids'])]:
            for row in [br,ours]:
                actual=row[name];actual=ast.literal_eval(actual) if isinstance(actual,str) else actual
                if actual!=ids:raise ValueError('Station order mismatch')
        for h in ['input_identity_sha256','input_mask_sha256','H_mask_sha256']:
            if br[h]!=ours[h]:raise ValueError('Input/mask identity mismatch: '+h)
        row={k:r[k] for k in KEYS};row.update(station_ids=r['station_ids'],H_station_ids=r['H_station_ids'])
        for h in ['input_identity_sha256','input_mask_sha256','H_mask_sha256']:row[h]=br[h]
        row.update(ours_checkpoint_sha256=ours['checkpoint_sha256'],deepsets_checkpoint_sha256=br['checkpoint_sha256'],
                   deepsets_prediction_sha256=br['prediction_sha256'],ours_prediction_sha256=ours['prediction_sha256'])
        for m in METRICS:
            av=float(ours['C1_'+m]);bv=float(br[m])
            if not np.isfinite([av,bv]).all():raise ValueError('Nonfinite comparison')
            row['Ours_'+m]=av;row['DeepSets_'+m]=bv;row['delta_'+m]=av-bv
        rows.append(row)
    return pd.DataFrame(rows)


def case_means(paired):
    counts=paired.groupby(['case_id','case_index','group']).agg(rows=('draw','size'),draws=('draw','nunique'))
    if len(counts)!=180 or not ((counts['rows']==5)&(counts['draws']==5)).all():
        raise ValueError('Five distinct draws per case/group required')
    columns=[p+m for p in ['Ours_','DeepSets_','delta_'] for m in METRICS]
    return paired.groupby(['case_id','case_index','group'],as_index=False)[columns].mean()


def paired_bootstrap(a,b,indices,tie_epsilon):
    delta=a-b;ci=np.quantile(delta[indices].mean(1),[.025,.975])
    return dict(ours_mean=float(a.mean()),deepsets_mean=float(b.mean()),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                ci_low=float(ci[0]),ci_high=float(ci[1]),ours_wins=int((delta < -tie_epsilon).sum()),
                deepsets_wins=int((delta > tie_epsilon).sum()),ties=int((np.abs(delta)<=tie_epsilon).sum()),
                ours_relative_error_reduction=float(1-a.mean()/b.mean()) if b.mean()!=0 else None)


def statistics(case,c):
    rng=np.random.default_rng(c['comparison']['bootstrap_seed']);ix=rng.integers(0,36,(c['comparison']['bootstrap_samples'],36))
    rows=[];tails=[]
    for group in GROUPS:
        frame=case[case.group==group].sort_values('case_id');assert len(frame)==36
        for m in METRICS:
            a=frame['Ours_'+m].to_numpy();b=frame['DeepSets_'+m].to_numpy()
            rows.append(dict(group=group,metric=m,family='primary' if m=='relative_l1' else 'secondary',cases=36,
                             **paired_bootstrap(a,b,ix,c['comparison']['tie_epsilon'])))
            for name,v in [('Ours',a),('DeepSets',b)]:
                tails.append(dict(group=group,metric=m,model=name,mean=float(v.mean()),median=float(np.median(v)),
                                  p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),maximum=float(v.max()),
                                  worst_case_id=frame.iloc[int(v.argmax())]['case_id'],worst_case_index=int(frame.iloc[int(v.argmax())]['case_index'])))
    contrasts=[]
    for m in ['relative_l1','relative_l2','heldout_relative_l1']:
        arrays={name:case.pivot(index='case_id',columns='group',values=name+'_'+m).sort_index()[GROUPS].to_numpy() for name in ['Ours','DeepSets']}
        for label,func in [('G1_G4_macro',lambda x:x[:,:4].mean(1)),('G5_minus_G1',lambda x:x[:,4]-x[:,0])]:
            contrasts.append(dict(comparison=label,metric=m,**paired_bootstrap(func(arrays['Ours']),func(arrays['DeepSets']),ix,c['comparison']['tie_epsilon'])))
        # Ratio of group grand means, recomputed within each case bootstrap.
        a,b=arrays['Ours'],arrays['DeepSets'];av=a[:,4].mean()/a[:,0].mean();bv=b[:,4].mean()/b[:,0].mean()
        samples=(a[ix,4].mean(1)/a[ix,0].mean(1))-(b[ix,4].mean(1)/b[ix,0].mean(1))
        low,high=np.quantile(samples,[.025,.975])
        contrasts.append(dict(comparison='G5_div_G1_grand_mean_ratio',metric=m,ours=float(av),deepsets=float(bv),delta=float(av-bv),ci_low=float(low),ci_high=float(high)))
    return pd.DataFrame(rows),pd.DataFrame(tails),contrasts
