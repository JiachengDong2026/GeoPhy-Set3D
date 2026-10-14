"""Independent saved-scalar/array audit. No model, checkpoint, GT, or projection reads."""
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(__file__).parent)
out=parser.parse_args().output
n=[json.loads(l) for l in (out/'normal_evaluated.jsonl').read_text().splitlines()]
records=json.loads((out/'station_subsets.json').read_text())
expected={(r['case_id'],r['geometry_id']):r for r in records}
assert len(n)==len(expected)==900
buckets={}
for row in n:
 source=expected[row['case_id'],row['geometry_id']]
 for k in ['case_id','case_index','draw','group','H_station_ids']:assert row[k]==source[k]
 assert row['input_station_ids']==source['station_ids']
 assert row['valid_context_scalar_count']==32*row['valid_view_voxel_count']
 assert row['ratio_undefined']==0
 assert abs(row['delta_context_ratio']-row['delta_rms']/row['context0_rms'])<1e-14
 for m in ['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2','heldout_relative_l1','heldout_relative_l2','integrated_concentration_relative_error','peak_value_absolute_error']:
  assert row['delta_'+m]==row['C1_'+m]-row['C0_'+m]
 buckets.setdefault((row['case_id'],row['group']),[]).append(row)
case=pd.read_csv(out/'case_group_180.csv');assert len(case)==len(buckets)==180
for row in case.to_dict('records'):
 items=buckets[row['case_id'],row['group']];assert len(items)==5
 for k in ['C0_relative_l1','C1_relative_l1','delta_relative_l1','delta_context_ratio']:
  assert abs(row[k]-sum(x[k] for x in items)/5)<1e-14
indices=np.random.default_rng(2026101701).integers(0,36,(10000,36))
s=json.loads((out/'statistics.json').read_text())
for row in s['primary']:
 a=case[case.group==row['group']].sort_values('case_index')
 delta=a.C1_relative_l1.to_numpy()-a.C0_relative_l1.to_numpy()
 ci=np.quantile(delta[indices].mean(1),[.025,.975])
 assert np.max(np.abs(ci-[row['ci_low'],row['ci_high']]))<1e-14
 assert abs(delta.mean()-row['delta_mean'])<1e-14
historical=Path(json.loads((out/'protocol.json').read_text())['config']['paths']['historical'])
for i in [0,18,35]:
 for g in ['G1','G3','G5']:
  with np.load(historical/'case_images'/f'case{i}_{g}.npz',allow_pickle=False) as old,np.load(out/'case_images'/f'normal_case{i}_V4-{g}-d0.npz',allow_pickle=False) as cur:
   assert np.array_equal(old['prediction'],cur['C1'])
sc=pd.read_csv(out/'shortcut_18.csv');assert len(sc)==18
for mode in ['signflip','wrong_pairing']:
 actual={(r['case_index'],r['group'],r['draw']) for r in sc[sc['mode']==mode].to_dict('records')}
 assert actual=={(i,g,0) for i in [0,18,35] for g in ['G1','G3','G5']}
zero=pd.read_csv(out/'zero_paired_900.csv');assert len(zero)==900
for row in zero.to_dict('records'):
 assert row['c0_identical_to_normal']
 assert abs(row['zero_minus_normal_relative_l1']-(row['C1_relative_l1']-row['C1_relative_l1_normal']))<1e-14
l2=[]
for level,frame in [('case_layout_descriptive',pd.DataFrame(n)),('case_group',case)]:
 for g,a in frame.groupby('group'):
  for metric in ['relative_l2','heldout_relative_l2']:
   mask=(a.delta_observed_relative_l1< -1e-8)&(a['delta_'+metric]>1e-8)
   v=a.loc[mask,'delta_'+metric]
   l2.append(dict(level=level,group=g,event='input_L1_improved_'+metric+'_worsened',count=int(mask.sum()),total=len(a),frequency=float(mask.mean()),mean_worsening=float(v.mean()) if len(v) else None,max_worsening=float(v.max()) if len(v) else None))
relation=[]
for g,a in case.groupby('group'):
 for kind in ['observed','heldout']:
  relation.append(dict(group=g,x='delta_'+kind+'_relative_l1',y='delta_relative_l1',pearson=float(pearsonr(a['delta_'+kind+'_relative_l1'],a.delta_relative_l1).statistic),spearman=float(spearmanr(a['delta_'+kind+'_relative_l1'],a.delta_relative_l1).statistic),case_count=36,p_value=None,note='exploratory supporting description; no p-value or threshold selection'))
for r in [json.loads(l) for l in (out/'inference_sessions.jsonl').read_text().splitlines()]:
 assert r['access']['val_gt64_opens']==r['access']['forbidden_attempts']==0
 assert r['parameters_buffers_unchanged']
for mode in ['normal','zero','signflip','wrong_pairing']:
 a=json.loads((out/f'{mode}_gt_access.json').read_text())
 assert a['checkpoint_opens']==a['val_pool_a_projection_opens']==a['forbidden_attempts']==0
summary=dict(passed=True,paired_rows=900,case_group_rows=180,shortcut_rows=18,zero_rows=900,independent_bootstrap_comparisons=5,historical_saved_C1_byte_exact_cases=9,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             supplementary_L2_discordance=l2,measurement_delta_relation=relation,
             supplementary_analysis='L2 event counts and measurement/voxel delta correlations are exploratory, reported after primary analysis; primary hypotheses and tests unchanged')
(out/'independent_scalar_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('Independent scalar/array/identity/bootstrap/access audit: PASSED')
print('Case/group L2 discordance:',[(r['group'],r['event'],r['count']) for r in l2 if r['level']=='case_group'])
