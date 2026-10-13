#!/usr/bin/env python3
"""Audit preregistration, all actual schedules and independent evaluation boundary."""
import argparse,json,sys,gzip
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json
from diagnose_clustered_one_swap import snapshot
from run_geometry_augmentation_formal import configuration,verify,ARCHIVE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--final',action='store_true');a=ap.parse_args();c,out=configuration(a.config);p=verify(c,out)
    for name in ('dataset_before','history_before'):
        full=out/'registration'/f'{name}.json';compressed=ARCHIVE/'registration'/f'{name}.json.gz'
        assert gzip.decompress(compressed.read_bytes())==full.read_bytes()
    assert snapshot(c['paths']['dataset'])==json.loads((out/'registration/dataset_before.json').read_text())
    hist=json.loads((out/'registration/history_before.json').read_text())
    for path,old in hist.items():assert snapshot(path)==old,path
    # Full input plan archive round trip; no datasets/checkpoints loaded.
    for arm in c['arms']:
        assert gzip.decompress((ARCHIVE/f'{arm}_station_plan.json.gz').read_bytes())==(out/f'{arm}_station_plan.json').read_bytes()
    old=json.loads((Path(c['paths']['sensitivity'])/'geometry_layouts.json').read_text());new=json.loads((out/'confirmatory_geometries/geometry_layouts.json').read_text());assert old['boundaries']==new['boundaries']
    o={tuple(sorted(r['station_ids'])) for r in old['layouts']};n={tuple(sorted(r['station_ids'])) for r in new['layouts']};assert len(n)==50 and not o&n
    records=json.loads((out/'confirmatory_geometries/evaluation_records.json').read_text());assert len(records)==2880
    for r in records:assert len(r['station_ids'])==len(r['H_station_ids'])==4 and not set(r['station_ids'])&set(r['H_station_ids'])
    result=dict(preregistration_valid=True,dataset_metadata_unchanged=True,history_metadata_unchanged=True,historical_scalar_hashes_unchanged=True,full_plan_archive_roundtrip=True,
        confirmatory_subsets_new=True,GT128_content_reads=0,final=a.final)
    if a.final:
        ev=out/'posthoc_evaluation';mm=json.loads((ev/'measurement_complete.json').read_text());gg=json.loads((ev/'posthoc_complete.json').read_text());assert mm['rows']==gg['rows']==5760 and gg['both300_before_any_GT_read']
        for arm in c['arms']:
            complete=json.loads((out/arm/'complete.json').read_text());assert complete['epochs']==300 and complete['optimizer_steps']==24900
            for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts','pool_b_open_attempts','test_or_unknown_npy_attempts','data_write_attempts'):assert complete['access'][k]==0,k
            assert digest(out/arm/'checkpoints/best.pt')==complete['best_sha256']
            logs=[json.loads(l) for l in (out/arm/'metrics.jsonl').read_text().splitlines()];assert [r['epoch'] for r in logs]==list(range(300))
            assert sum(len(r['train']['gradient_norms']) for r in logs)==24900
            for r in logs:
                assert r['optimizer_steps']==83*(r['epoch']+1) and r['train']['sample_count']==166 and r['val']['sample_count']==36
                assert all(np.isfinite([v['preclip'],v['postclip']]).all() and v['postclip']<=1.00001 for v in r['train']['gradient_norms'])
            assert digest(ev/f'{arm}_measurement_rows.jsonl')==mm['row_sha256'][arm]
        assert json.loads((ev/'selector_audit.json').read_text())['passed']
        access=json.loads((ev/'posthoc_access_audit.json').read_text());assert access['unique_val_gt64_files']==36 and access['forbidden_attempts']==0
        result.update(both300_completed=True,optimizer_steps_per_arm=24900,all_actual_epoch_schedules_verified_in_runner=True,measurement_rows=5760,posthoc_val_GT64_unique=36,
            selector_measurement_only_verified=True,best_hashes_verified=True,training_forbidden_attempts=0,posthoc_forbidden_attempts=0)
    dest=out/('final_integrity_audit.json' if a.final else 'registration/integrity_pre_training.json');write_json(dest,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
