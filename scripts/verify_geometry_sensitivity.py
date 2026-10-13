#!/usr/bin/env python3
"""Independent scalar audit; no dataset content or checkpoint loading."""
import argparse
import json
import math
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json
from run_geometry_sensitivity import configuration,verify,inventory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(REPO/'configs/geometry_sensitivity_v1.yaml'))
    args=parser.parse_args();cfg=configuration(args.config);p=verify(cfg);out=Path(cfg['paths']['output'])
    raw=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]
    records=json.loads((out/'station_subsets.json').read_text());lookup={(r['case_id'],r['geometry_id']):r for r in records}
    assert len(raw)==len(lookup)==900
    gd=json.loads((out/'geometry_layouts.json').read_text());descriptors={r['geometry_id']:r for r in gd['descriptors']}
    spectra={r['geometry_id']:r for r in json.loads((out/'spectra.json').read_text())}
    for row in raw:
        e=lookup[row['case_id'],row['geometry_id']];g=descriptors[row['geometry_id']]
        assert row['station_ids']==e['station_ids'] and row['H_station_ids']==e['H_station_ids']
        assert not set(e['station_ids'])&set(e['H_station_ids'])
        assert len(e['station_ids'])==len(e['H_station_ids'])==4
        assert row['checkpoint_sha256']==cfg['checkpoint_sha256']
        assert row['valid_input_rays']==g['valid_ray_count']
        assert math.isclose(row['mae'],row['absolute_sum']/262144,abs_tol=1e-14)
        assert math.isclose(row['relative_l1'],row['absolute_sum']/max(row['gt_absolute_sum'],1e-6),abs_tol=1e-14)
        assert math.isclose(row['rmse'],math.sqrt(row['squared_sum']/262144),abs_tol=1e-14)
        assert math.isclose(row['relative_l2'],math.sqrt(row['squared_sum']/max(row['gt_squared_sum'],1e-12)),abs_tol=1e-14)
    for r in descriptors.values():
        origin=np.asarray(r['station_origins_m']);np.testing.assert_array_equal(origin[:,1],r['station_height_m'])
        d=np.linalg.norm(origin[:,None]-origin[None],axis=-1)[np.triu_indices(4,1)]
        np.testing.assert_allclose(d,r['pairwise_station_distance_m'],atol=1e-12,rtol=0)
        center=.3-origin;center/=np.linalg.norm(center,axis=1,keepdims=True)
        score=np.mean(1-np.abs(center@center.T)[np.triu_indices(4,1)])
        assert abs(score-r['center_direction_score'])<1e-12
        q05,q40,q60,q80=gd['boundaries'];group=r['group']
        assert {'G1':score>=q80,'G2':q60<=score<q80,'G3':q40<=score<q60,'G4':q05<=score<q40,'G5':score<q05 and r['station_azimuth_span_deg']<=cfg['cluster_span_max_deg']}[group]
        s=np.asarray(spectra[r['geometry_id']]['singular_values_ppm_m']);assert len(s)==512 and (np.diff(s)<=0).all()
        rank=int((s>cfg['spectral_relative_sigma_tolerance']*s[0]).sum());assert rank==r['coarse_numeric_rank']
        assert r['coarse_nullity']==512-rank and r['full64_nullity_lower_bound']==262144-r['valid_ray_count']>0
        if rank<512:assert r['coarse_condition_number'] is None
    df=pd.DataFrame(raw);cases=sorted(df.case_id.unique());assert len(cases)==36
    exported=pd.read_csv(out/'geometry_sensitivity_results.csv');assert len(exported)==900
    source_metadata=[json.loads(l) for l in (Path(cfg['paths']['dataset'])/'source_index.jsonl').read_text().splitlines()]
    sample_ids={r['case_id']:str(r['sample_id']) for r in source_metadata if r['split']=='val'}
    keyed=exported.set_index(['case_id','geometry_id'])
    assert keyed.index.is_unique
    for row in raw:
        exported_row=keyed.loc[row['case_id'],row['geometry_id']]
        for metric in ['relative_l1','relative_l2','mae','rmse','observed_relative_l1','heldout_relative_l1']:
            assert abs(exported_row[metric]-row[metric])<1e-12
        assert json.loads(exported_row['station_ids'])==row['station_ids']
        assert json.loads(exported_row['H_station_ids'])==row['H_station_ids']
        assert str(exported_row['sample_id'])==sample_ids[row['case_id']]
        assert exported_row['protocol_sha256']==digest(out/'protocol.json')
        for name in ['original_seed','population_seed','subset_seed','heldout_seed']:assert exported_row[name]==cfg[name]
    cm=pd.read_csv(out/'case_means.csv');metrics=[k for k in cm.columns if k not in ['case_id','group']]
    assert len(cm)==180
    for row in cm.to_dict('records'):
        rows=[r for r in raw if r['case_id']==row['case_id'] and r['group']==row['group']]
        assert len(rows)==5 and sorted(r['draw'] for r in rows)==list(range(5))
        for key in metrics:assert abs(row[key]-math.fsum(r[key] for r in rows)/5)<1e-12
    ids=np.random.default_rng(cfg['bootstrap_seed']).integers(0,36,(cfg['bootstrap_samples'],36))
    pairs=json.loads((out/'paired_statistics.json').read_text());assert len(pairs)==100
    for pair in pairs:
        right,left=pair['comparison'].split('-');key=pair['metric']
        a=cm[cm.group==left].set_index('case_id').loc[cases,key].to_numpy();b=cm[cm.group==right].set_index('case_id').loc[cases,key].to_numpy();d=b-a
        np.testing.assert_allclose(np.quantile(d[ids].mean(1),[.025,.975]),pair['case_bootstrap_95ci'],atol=1e-12,rtol=0)
        assert abs(d.mean()-pair['mean_delta'])<1e-12
        assert pair['worsened_cases']==int((d>0).sum()) and pair['improved_cases']==int((d<0).sum())
        pv=float(wilcoxon(d,method='auto').pvalue) if np.any(d) else 1.
        assert abs(pv-pair['wilcoxon_p'])<1e-12
    correlations=json.loads((out/'geometry_correlations.json').read_text());checked=0
    for r in correlations:
        if r['coefficient'] is None:continue
        keys=sorted(k for k,g in descriptors.items() if r['scope']=='all_G1_G5' or g['group']!='G5')
        x=np.array([descriptors[k][r['descriptor']] for k in keys]);y=np.array([df[df.geometry_id==k].relative_l1.mean() for k in keys])
        actual=(pearsonr if r['correlation']=='Pearson' else spearmanr)(x,y).statistic
        assert abs(actual-r['coefficient'])<1e-12;checked+=1
    for case in cfg['representative_case_indices']:
        volumes=[np.load(out/'case_images'/f'case{case}_{g}.npz') for g in cfg['representative_groups']]
        assert all(np.array_equal(volumes[0]['GT'],v['GT']) for v in volumes)
        for group,v in zip(cfg['representative_groups'],volumes):
            rawrow=next(r for r in raw if r['case_index']==case and r['group']==group and r['draw']==cfg['representative_draw'])
            delta=np.abs(v['prediction'].astype(float)-v['GT'].astype(float)).sum()/max(np.abs(v['GT'].astype(float)).sum(),1e-6)
            assert abs(delta-rawrow['relative_l1'])<1e-12
    assert inventory(cfg)==json.loads((out/'inventory_before.json').read_text())
    audit=json.loads((out/'access_audit.json').read_text());assert all(a['forbidden_attempts']==0 for a in audit.values())
    assert audit['registration_access_audit.json']['checkpoint_opens']==audit['registration_access_audit.json']['val_gt64_opens']==audit['registration_access_audit.json']['val_pool_a_projection_opens']==0
    assert audit['full_access_audit.json']['unique_val_pool_a_projections']==audit['full_access_audit.json']['unique_val_gt64_files']==36
    result=dict(passed=True,actual_rows=900,layouts=25,cases=36,case_group_means=180,independent_paired_bootstrap_checks=100,
        scipy_geometry_correlations_checked=checked,representative_volume_L1_checks=9,physical_axes_and_group_membership_checked=True,
        source_checkpoint_history_and_data_metadata_unchanged=True,GT_or_H_not_used_for_inference=True,forbidden_attempts=0,
        report_source_sha256=digest(REPO/'scripts/report_geometry_sensitivity.py'))
    write_json(out/'independent_verification.json',result)
    dest=REPO/'docs/reports/artifacts/geometry_sensitivity_v1';write_json(dest/'independent_verification.json',result)
    manifest=dict(registered_sources=p['source_sha256'],input_sha256=p['registration_sha256'],historical_sha256=p['historical_sha256'],
        report_sources={str(q.relative_to(REPO)):digest(q) for q in [REPO/'scripts/report_geometry_sensitivity.py',Path(__file__)]})
    write_json(out/'source_manifest.json',manifest);write_json(dest/'source_manifest.json',manifest)
    sums=[f'{digest(q)}  {q.relative_to(REPO)}' for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='SHA256SUMS']
    for q in [REPO/'docs/reports/geometry_sensitivity_summary.md',REPO/'docs/geometry_sensitivity_v1_protocol_zh.md']:sums.append(f'{digest(q)}  {q.relative_to(REPO)}')
    (dest/'SHA256SUMS').write_text('\n'.join(sums)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
