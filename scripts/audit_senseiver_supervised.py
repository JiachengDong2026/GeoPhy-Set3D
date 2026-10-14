#!/usr/bin/env python3
"""Final scalar/source audit, never opens a dataset array or foreign weight."""
import gzip,json,math,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.utils import write_json
import train_ray_senseiver_supervised as t

def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)

def main():
    a=t.ART;r=json.loads((a/'registration.json').read_text());c=json.loads((a/'complete.json').read_text());steps=gz(a/'steps.json.gz');history=gz(a/'history.json.gz')
    assert len(steps)==c['updates'] and [s['updates'] for s in steps]==list(range(1,len(steps)+1))
    minima={k:{g:min(s[k][g] for s in steps) for g in ['encoder','latents','decoder']} for k in ['gradient_l2','update_l2']}
    assert all(math.isfinite(v) and v>0 for d in minima.values() for v in d.values())
    assert all(s['queries_per_case']==64**3 and len(s['cases'])==2 for s in steps)
    assert all(3<=len(row['stations'])<=6 and len(set(row['stations']))==len(row['stations']) and row['valid_rays']>0 for s in steps for row in s['cases'])
    for e in range(c['epochs']):
        ss=[s for s in steps if s['epoch_zero_based']==e];assert len(ss)==83 and [s['position'] for s in ss]==list(range(1,84))
        cases=[z['index'] for s in ss for z in s['cases']];assert sorted(cases)==list(range(166))
    assert min(h['val']['mean']['nmse'] for h in history)==c['best_nmse']
    for p,h in r['source_sha256'].items():assert sha256(REPO/p)==h,p
    gate=json.loads((REPO/'docs/reports/artifacts/ray_senseiver_supervised_stability_v1/registration.json').read_text())
    for p,h in gate['historical_reports_sha256'].items():assert sha256(REPO/p)==h,p
    assert not c['access']['denied_attempts']
    audits={}
    for name in ['Senseiver_inference_access.json','SupervisedMRGR_inference_access.json','evaluation_GT_access.json']:
        if (a/name).exists():
            audit=json.loads((a/name).read_text());assert audit['forbidden_attempts']==0
            if 'inference' in name:assert audit['val_gt64_opens']==0
            else:assert audit['checkpoint_opens']==audit['val_pool_a_projection_opens']==0
            audits[name]=audit
    value=dict(passed=True,completed_epochs=c['epochs'],actual_updates=c['updates'],partial_epoch_updates=c['updates']-83*c['epochs'],all_complete_epochs_have_166_distinct_train_cases=True,all_updates_effective_batch2=True,all_queries_per_case=64**3,all_input_view_counts_3_to_6=True,minimum_group_gradient_and_update_l2=minima,source_manifest_preserved=True,historical_reports_preserved=True,zero_training_denied_attempts=True,initial_hash=r['initial_hash'],selector_actual_min=c['best_nmse'],best_epoch=c['best_epoch'],stop_reason=c['reason'],evaluation_access=audits,audit_source_sha256=sha256(__file__))
    write_json(a/'integrity.json',value);print(json.dumps(value,indent=2))
if __name__=='__main__':main()
