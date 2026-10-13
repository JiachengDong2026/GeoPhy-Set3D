#!/usr/bin/env python3
"""Independent raw-row arithmetic audit of the completed matched300 trial."""
import json,math,sys
from collections import defaultdict
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
from geophy_set3d.training.utils import write_json
from geophy_set3d.data.measurement_only import DisjointABHSampler
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/experiments/geometry_stratified_selfsup_matched_v1')


def main():
 ev=OUT/'evaluation';p=json.loads((OUT/'protocol.json').read_text());rows=json.loads((ev/'per_case_draw.json').read_text());assert len(rows)==4680
 expected=json.loads((REPO/'docs/reports/artifacts/geometry_stratified_selfsup_matched_v1/evaluation_station_subsets.json').read_text())['records']
 expected={(r['section'],r['V'],r['stratum'],r['case_id'],r['draw']):r for r in expected};groups=defaultdict(list)
 for r in rows:
  key=(r['section'],r['V'],r['stratum'],r['case_id'],r['draw']);s=expected[key]
  assert r['input_station_ids']==s['station_ids'] and r['H_station_ids']==s['H_station_ids']
  groups[(r['model'],*key[:-1])].append(r)
 assert len(groups)==936 and all(len(v)==5 and {r['draw'] for r in v}==set(range(5)) for v in groups.values())
 cases=sorted({r['case_id'] for r in rows});rng=np.random.default_rng(p['config']['bootstrap']['seed']);index=rng.integers(0,36,(10000,36))
 stats=json.loads((ev/'paired_statistics.json').read_text());checked=0
 for r in stats['per_cell']+stats['macro']:
  def means(arm):
   values=[]
   for case in cases:
    views=[3,4,6] if r['V']=='macro3_4_6' else [r['V']]
    values.append(math.fsum(math.fsum(q[r['metric']] for q in groups[arm,r['section'],v,r['stratum'],case])/5 for v in views)/len(views))
   return np.asarray(values)
  a,b=means('Control'),means('Treatment');d=b-a;ci=np.quantile(d[index].mean(1),[.025,.975])
  np.testing.assert_allclose(ci,r['ci'],rtol=0,atol=1e-12)
  assert abs(a.mean()-r['Control'])<1e-12 and abs(b.mean()-r['Treatment'])<1e-12 and abs(d.mean()-r['delta'])<1e-12
  checked+=1
 for arm in ['Control','Treatment']:
  epochs=[json.loads(l) for l in (OUT/arm/'metrics.jsonl').read_text().splitlines()]
  assert len(epochs)==300 and [q['optimizer_steps'] for q in epochs]==[83*(i+1) for i in range(300)]
  assert json.loads((OUT/arm/'complete.json').read_text())['initial_state_sha256']==p['initial_state_sha256'][arm]
 control_plans=json.loads((OUT/'Control_station_plan.json').read_text());original=DisjointABHSampler()
 for epoch,plan in enumerate(control_plans):
  for index,r in enumerate(plan):
   actual=original.sample(64,epoch=epoch,sample_index=index,case_id=r['case_id'])
   assert all(a.tolist()==r[k] for a,k in zip(actual,['A','B','H']))
 audits=json.loads((ev/'access_audit.json').read_text());assert audits['posthoc_GT']['unique_val_gt64_files']==36
 for phase in ['training','measurement']:
  for a in audits[phase].values():assert a['gt_open_attempts']==a['forbidden_npy_attempts']==a['supervised_checkpoint_open_attempts']==a['pool_b_open_attempts']==a['test_or_unknown_npy_attempts']==0
 decision=json.loads((ev/'decision.json').read_text());assert decision['decision']=='NO-GO'
 natural=next(r for r in stats['macro'] if r['section']=='natural' and r['metric']=='relative_l1');low=next(r for r in stats['macro'] if r['stratum']=='low' and r['metric']=='relative_l1')
 assert natural['ci'][0]>0 and low['delta']>0
 result=dict(passed=True,rows=4680,case_groups=936,independent_bootstrap_comparisons=checked,all_49800_Control_samples_match_original_sampler=True,
  identical_frozen_inputs_and_H=True,both300_24900_steps=True,GT_before_frozen_evaluation_opens=0,posthoc_unique_GT64=36,decision='NO-GO')
 write_json(ev/'independent_result_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
