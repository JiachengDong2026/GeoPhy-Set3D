#!/usr/bin/env python3
"""Independent prospective schedule audit; geometry/metadata only, no new inference."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard
from geophy_set3d.training.utils import write_json
from prepare_geometry_augmentation import verify


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(REPO/'configs/geometry_augmentation_preparation_v1.yaml'));a=ap.parse_args()
    c=yaml.safe_load(Path(a.config).read_text());p=verify(c,True);out=Path(c['paths']['output'])
    plans=json.loads((out/'Treatment_station_plan.json').read_text());stats=json.loads((out/'sampler_audit.json').read_text())
    original=json.loads((Path(c['paths']['existing_control'])/'Control_station_plan.json').read_text())
    boundaries=json.loads((out/'boundaries.json').read_text())['boundaries'];counts=Counter();joint=np.zeros((3,3),int)
    modes=['natural','moderate','hard'];frequency={k:np.zeros(64,int) for k in modes};max_attempt=0;flat=[]
    with ClusteredEvaluationReadGuard(c['paths']['dataset'],[],out,geometry_only=True) as guard:
        g=dict(np.load(Path(c['paths']['dataset'])/'geometry/train_geometry.npz',allow_pickle=False))
        center=.3-g['station_origin_m'];center/=np.linalg.norm(center,axis=1,keepdims=True)
        for epoch,rows in enumerate(plans):
            assert len(rows)==166
            for i,r in enumerate(rows):
                assert r['case_id']==original[epoch][i]['case_id']
                assert len(r['A'])==len(original[epoch][i]['A']) and len(r['B'])==len(original[epoch][i]['B'])
                sets=[set(r[k]) for k in ['A','B','H']];assert len(sets[2])==2
                assert not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
                for k in ['A','B']:
                    ids=r[k];mode=r[k+'_mode'];assert len(ids)==len(set(ids)) and set(ids)<=set(range(64))
                    d=center[ids];x,y=np.triu_indices(len(ids),1);score=float((1-np.abs(d@d.T)[x,y]).mean())
                    az=np.sort(g['station_azimuth_deg'][ids]%360);span=360-float(np.diff(np.r_[az,az[0]+360]).max())
                    assert abs(score-r[k+'_score'])<1e-12 and abs(span-r[k+'_span_deg'])<1e-10
                    if mode=='moderate':assert boundaries[str(len(ids))][0]<=score<boundaries[str(len(ids))][1]
                    if mode=='hard':assert span<=c['hard_span_max_deg']+1e-10
                    counts[mode]+=1;frequency[mode][ids]+=1
                joint[modes.index(r['A_mode']),modes.index(r['B_mode'])]+=1
                max_attempt=max(max_attempt,r['joint_attempts']);flat.append(r)
        assert len(flat)==49800 and dict(counts)==stats['mode_input_counts']
        np.testing.assert_array_equal(joint,stats['AB_joint_mode_counts'])
        for mode in modes:np.testing.assert_array_equal(frequency[mode],stats['station_frequency_by_mode'][mode])
        assert max_attempt==stats['joint_max_attempts']<=c['rejection_cap']
        assert guard.counts['val_pool_a_projection_opens']==guard.counts['val_gt64_opens']==guard.counts['checkpoint_opens']==0
        write_json(out/'verification_access_audit.json',guard.summary())
    smoke=json.loads((out/'smoke_result.json').read_text())
    assert smoke['passed'] and smoke['optimizer_steps']==0 and smoke['parameters_buffers_unchanged'] and not smoke['formal_training_run']
    assert {(r['A_mode'],r['B_mode']) for r in smoke['rows']}=={(a,b) for a in modes for b in modes}
    assert all(r['finite_nonzero_gradient'] and np.isfinite(r['loss']) for r in smoke['rows'])
    assert smoke['audit']['gt_open_attempts']==smoke['audit']['supervised_checkpoint_open_attempts']==smoke['audit']['pool_b_open_attempts']==smoke['audit']['test_or_unknown_npy_attempts']==0
    result=dict(passed=True,planned_pairs_checked=49800,planned_input_categories_checked=99600,
        disjoint_and_control_V_exact=True,mode_counts_and_station_frequencies_recomputed=True,
        geometry_score_and_span_independently_recomputed=True,smoke_backward_passes=9,optimizer_steps=0,
        formal_training_run=False,GT_checkpoint_poolB_test_attempts=0,verification_source_sha256=digest(Path(__file__)))
    dest=REPO/'docs/reports/artifacts/geometry_augmentation_preparation_v1'
    write_json(out/'independent_verification.json',result);write_json(dest/'independent_verification.json',result)
    for name in ['smoke_result.json','prepare.log','smoke.log','verification_access_audit.json']:shutil.copy2(out/name,dest/name)
    (dest/'SHA256SUMS').write_text('\n'.join(f'{digest(q)}  {q.relative_to(REPO)}' for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='SHA256SUMS')+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
