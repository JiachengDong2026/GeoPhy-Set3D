"""Runtime evaluation summaries; independent review has its own formulas."""
from ctx_v2 import *

def volume(x,truth):
    a=np.asarray(x,dtype=np.float64).reshape(64,64,64);t=truth.astype(np.float64);e=a-t
    peak=lambda v:[float(i*.6/63) for i in np.unravel_index(v.argmax(),v.shape)]
    return dict(relative_l1=float(np.abs(e).sum()/np.abs(t).sum()),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),
        mean=float(a.mean()),std=float(a.std()),spatial_cv=float(a.std()/max(a.mean(),1e-15)),min=float(a.min()),max=float(a.max()),
        nearzero_fraction=float((a<1e-10).mean()),peak_xyz_m=peak(a),truth_peak_xyz_m=peak(t),truth_peak=float(t.max()),zero_L1=1.)
def difference(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);e=a-b
    ac=a-a.mean();bc=b-b.mean();den=np.linalg.norm(ac)*np.linalg.norm(bc)
    return dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),fixed_scale_response=float(np.abs(e).mean()/design()['initial_mass_fraction']),
        spatial_correlation=None if den<1e-30 else float(ac@bc/den),peak_displacement_m=float(np.linalg.norm(np.subtract(np.unravel_index(a.argmax(),(64,)*3),np.unravel_index(b.argmax(),(64,)*3)))*.6/63))
def measurement(p,y,mask):
    p=p.astype(np.float64)[mask];y=y.astype(np.float64)[mask];e=p-y
    return dict(relative_l1=float(np.abs(e).sum()/max(np.abs(y).sum(),1e-6)),relative_l2=float(np.linalg.norm(e)/max(np.linalg.norm(y),1e-6)),zero_L1=1.,zero_L2=1.)
def paired(a,b,ta,tb):
    d=np.asarray(a,dtype=np.float64).reshape(-1)-np.asarray(b,dtype=np.float64).reshape(-1)
    dt=ta.reshape(-1)-tb.reshape(-1);den=np.linalg.norm(d)*np.linalg.norm(dt)
    return dict(**difference(a,b),truth_mae=float(np.abs(dt).mean()),truth_rmse=float(np.sqrt(np.mean(dt**2))),
        relative_to_truth_difference=float(np.abs(d).sum()/np.abs(dt).sum()),direction_cosine=None if den<1e-30 else float(d@dt/den))

def decide(history,pairs,swaps):
    d=design();g=d['gates'];first=history[0]['cases'];last=history[-1]['cases'];checks={}
    checks['seen_training']=all(r['AB_L1']<=g['seen_AB_max'] and first[i]['AB_L1']-r['AB_L1']>=g['seen_AB_absolute_improvement_min'] for i,r in enumerate(last[:6]))
    checks['independent_H']=all(m['relative_l1']<=g['all_H_last_two_max'] for ev in history[-2:] for r in ev['cases'] for m in r['measurement'] if m['target']=='H')
    checks['nontrivial_structure']=all(b['volume']['mean']>=g['all_mean_min'] and b['volume']['spatial_cv']>=g['all_spatial_cv_min'] for r in last for b in r['branches'].values())
    checks['pair_response']=all(p['fixed_scale_response']>=g['all_pairs_MAE_div_fixed_scale_min'] and p['repeat_bitwise'] for p in pairs)
    for branch in ('A','B'):
        for group in ('seen_seen','seen_unseen','unseen_unseen'):
            values=[p['direction_cosine'] or 0. for p in pairs if p['branch']==branch and p['group']==group]
            checks[f'direction_{branch}_{group}']=float(np.mean(np.asarray(values)>0))>=g['pair_cosine_positive_fraction_min'] and float(np.median(values))>=g['pair_cosine_median_min']
        checks['seen_volume_'+branch]=sum(r['branches'][branch]['volume']['relative_l1']<=g['seen_volume_L1_max'] for r in last[:6])>=g['seen_volume_min_cases']
        checks['unseen_volume_'+branch]=all(r['branches'][branch]['volume']['relative_l1']<=g['unseen_volume_L1_max'] for r in last[6:])
        for role in ('seen','unseen'):
            values=[p['matching_advantage_L1'] for p in swaps if p['branch']==branch and p['recipient_role']==role]
            checks['matching_'+branch+'_'+role]=float(np.mean(np.asarray(values)>=g['matching_advantage_min']))>=g['matching_advantage_fraction_min']
    checks['gradients']=all(b['gradient']['measurement_channel']['l2']>g['measurement_gradient_l2_min'] and
        all(b['gradient'][key]['finite_fraction']==1 for key in ('measurement_channel','geometry_channels','readout_weight','readout_bias')) for r in last for b in r['branches'].values())
    nearzero=any(b['volume']['mean']<g['all_mean_min'] or b['volume']['nearzero_fraction']>.95 for r in last for b in r['branches'].values())
    status='GO_DIAGNOSTIC_ONLY' if all(checks.values()) else 'NO_GO' if nearzero or not checks['pair_response'] else 'INCONCLUSIVE_BUDGET_EXHAUSTED'
    return dict(status=status,checks=checks,formal_training_authorized=False)
