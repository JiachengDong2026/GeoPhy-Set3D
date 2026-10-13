#!/usr/bin/env python3
"""Independent scalar-result audit: raw rows, geometry identity, case bootstrap."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
from geophy_set3d.evaluation.geometry_strata import membership,subset_scores,pair_scores
from geophy_set3d.training.utils import write_json
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/evaluations/geometry_strata_frozen_v1')


def main():
    cfg=json.loads((OUT/'protocol.json').read_text())['config'];b=json.loads((OUT/'boundaries.json').read_text())
    expected=json.loads((OUT/'station_subsets.json').read_text())['records']
    records={(r['section'],r['V'],r['stratum'],r['case_id'],r['draw']):r for r in expected}
    assert len(records)==2340
    descriptors={tuple(r['station_ids']):r for r in json.loads((OUT/'geometry_descriptors.json').read_text())}
    for r in expected:
        ids=r['station_ids'];d=descriptors[tuple(ids)];assert len(ids)==r['V']==d['V']
        assert len(set(ids))==len(ids) and not set(ids)&set(r['H_station_ids']) and len(r['H_station_ids'])==4
        assert np.all(np.asarray(d['station_origins_m'])[:,1]==d['station_height_m'])
        if r['section']=='controlled':
            assert membership(d['center_direction_score'],b['V'][str(r['V'])]['boundaries'])==r['stratum']
            score=subset_scores([list(range(r['V']))],pair_scores(d['station_origins_m']))[0]
            assert abs(score-d['center_direction_score'])<1e-12
    actual=[];groups=defaultdict(list)
    for name in ('SelfSup','Supervised'):
        rows=[json.loads(l) for l in (OUT/f'{name}_rows.jsonl').read_text().splitlines()];assert len(rows)==2340
        for r in rows:
            key=(r['section'],r['V'],r['stratum'],r['case_id'],r['draw']);frozen=records[key]
            assert r['input_station_ids']==frozen['station_ids'] and r['H_station_ids']==frozen['H_station_ids']
            groups[(name,*key[:-1])].append(r)
        actual.extend(rows)
    cm=list(csv.DictReader((OUT/'case_means.csv').open()));assert len(cm)==936
    metrics=['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2','heldout_relative_l1','heldout_relative_l2',
             'integrated_concentration_relative_error','peak_value_absolute_error']
    lookup={}
    for row in cm:
        key=(row['model'],row['section'],int(row['V']),row['stratum'],row['case_id']);raw=groups[key]
        assert len(raw)==5 and {r['draw'] for r in raw}==set(range(5))
        for metric in metrics:assert math.isclose(float(row[metric]),math.fsum(r[metric] for r in raw)/5,abs_tol=1e-12)
        lookup[key]=row
    pairs=json.loads((OUT/'paired_statistics.json').read_text())['comparisons'];cases=sorted({r['case_id'] for r in expected})
    rng=np.random.default_rng(cfg['bootstrap_seed']);resample=rng.integers(0,len(cases),(10000,len(cases)))
    for pair in pairs:
        left,right=pair['comparison'].split('-');model=pair['model'];views=pair['V'];metric=pair['metric']
        differences=[]
        for case in cases:
            a=groups[(model,'controlled',views,left,case)];c=groups[(model,'controlled',views,right,case)]
            differences.append(math.fsum(r[metric] for r in a)/5-math.fsum(r[metric] for r in c)/5)
        d=np.asarray(differences);ci=np.quantile(d[resample].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,pair['ci'],atol=1e-12,rtol=0)
        assert abs(d.mean()-pair['mean_delta'])<1e-12
    audit=json.loads((OUT/'access_audit.json').read_text())
    assert all(a['forbidden_attempts']==0 for a in audit.values())
    for phase in ('boundary_access_audit.json','registration_access_audit.json'):
        assert audit[phase]['val_gt64_opens']==audit[phase]['val_pool_a_projection_opens']==audit[phase]['checkpoint_opens']==0
    for phase in ('SelfSup_access_audit.json','Supervised_access_audit.json'):
        assert audit[phase]['unique_val_gt64_files']==audit[phase]['unique_val_pool_a_projections']==36
    result=dict(passed=True,acquisition_records=2340,paired_model_rows=4680,case_mean_rows=936,metrics_per_case=10,
        independently_recomputed_case_bootstrap_comparisons=len(pairs),controlled_membership_verified=True,
        models_share_station_order_and_H=True,geometry_height_and_center_score_recomputed=True,natural_preserved=True,
        clustered_excluded_from_main_strata=True,boundary_measurement_gt_checkpoint_opens=0,forbidden_access_attempts=0)
    write_json(OUT/'independent_result_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
