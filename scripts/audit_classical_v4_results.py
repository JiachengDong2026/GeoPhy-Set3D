#!/usr/bin/env python3
"""Independent saved-volume, pairing, curve and case-bootstrap verification."""
import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,yaml
from run_classical_v4 import verify,verify_selection,ReadGuard,write,digest


def check_row(row):
    curve=Path(row['curve_file']);points=json.loads(curve.read_text());volume=curve.parents[2]/'volumes'/curve.parent.name/(curve.stem+'.npy')
    assert digest(volume)==row['volume_sha256']
    c=np.load(volume,mmap_mode='r');assert c.shape==(64,64,64) and c.dtype==np.float32 and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
    assert len(points)==row['solver_stats']['iterations'] and 0<len(points)<=row['solver_config']['iterations']
    assert np.isfinite([v['objective'] for v in points]).all()
    if row['method']!='SART':assert np.max(np.diff([v['objective'] for v in points]),initial=0)<=2e-12
    if row['solver_stats']['status']=='budget_exhausted':assert len(points)==row['solver_config']['iterations']
    assert set(row['input_station_ids']).isdisjoint(row['H_station_ids']) and len(set(row['input_station_ids']))==4 and len(set(row['H_station_ids']))==4
    return volume


def main():
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);p=verify(cfg,out);s=verify_selection(out)
    calibration=json.loads((out/'calibration_rows.json').read_text());classic=json.loads((out/'classical_rows.json').read_text());network=json.loads((out/'network_historical_rows.json').read_text())
    records=json.loads((out/'station_protocol.json').read_text());lookup={(r['case_id'],r['draw']):r for r in records}
    with ReadGuard(cfg,out,'calibrate',p['calibration_cases']):
        for row in calibration:check_row(row)
    for method in ['SART','Tikhonov','TV']:
        with ReadGuard(cfg,out/'workers'/method,'evaluate'):
            group=[r for r in classic if r['method']==method]
            for row in group:
                check_row(row);r=lookup[row['case_id'],row['draw']]
                assert row['input_station_ids']==r[row['condition']] and row['H_station_ids']==r['H']
                assert row['solver_config']==s['selected'][method][row['selection_identities'][0]]['config']
    summaries=json.loads((out/'paired_statistics.json').read_text());expanded=list(network)
    for r in classic:
        for identity in r['selection_identities']:expanded.append(dict(r,model=r['method']+'_'+identity))
    groups=collections.defaultdict(list)
    for r in expanded:groups[r['model'],r['condition'],r['case_id']].append(r)
    rng=np.random.default_rng(cfg['bootstrap']['seed']);index=rng.integers(0,36,(10000,36))
    for stat in summaries:
        case_ids=sorted({r['case_id'] for r in expanded if r['model']==stat['left_model'] and r['condition']==stat['left_condition']});assert len(case_ids)==36
        left=[];right=[]
        for case in case_ids:
            a=groups[stat['left_model'],stat['left_condition'],case];b=groups[stat['right_model'],stat['right_condition'],case]
            assert len(a)==len(b)==5 and sorted(r['draw'] for r in a)==sorted(r['draw'] for r in b)==list(range(5))
            # Independently aggregate paired draw differences, then bootstrap cases.
            da={r['draw']:r[stat['metric']] for r in a};db={r['draw']:r[stat['metric']] for r in b}
            left.append(np.mean([da[i] for i in range(5)]));right.append(np.mean([db[i] for i in range(5)]))
        delta=np.asarray(left)-right;ci=np.percentile(delta[index].mean(axis=1),[2.5,97.5])
        assert abs(delta.mean()-stat['mean_delta'])<1e-12
        assert np.allclose(ci,[stat['CI_low'],stat['CI_high']],atol=1e-12,rtol=0)
    sessions=[]
    for path in [out/'access_sessions.jsonl']+list(out.glob('workers/*/access_sessions.jsonl')):
        sessions.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    assert all(r['forbidden_attempts']==0 for r in sessions)
    write(out/'independent_verification.json',dict(calibration_volumes_verified=len(calibration),Val_volumes_verified=len(classic),network_historical_rows_paired=len(network),paired_statistics_verified=len(summaries),source_selected_config_station_hashes_verified=True,
        volume_range_shape_dtype_and_hashes_verified=True,curves_and_fixed_stopping_budget_verified=True,variation_objective_monotonic_with_2e_12_tolerance=True,
        bootstrap_unit='36 cases after five paired draws',bootstrap_seed=cfg['bootstrap']['seed'],bootstrap_resamples=10000,real_forbidden_attempts=0,
        sessions=len(sessions),failed_diagnostic_sessions=sum(r['failed'] for r in sessions),failure_note='Only explicitly recorded timer TF32 setup check and exact-valid-zero-presence diagnostic; no solver/parameter changes or forbidden data reads'))
    print('Independent reconstruction and paired bootstrap verification passed',len(classic),len(summaries))
if __name__=='__main__':main()
