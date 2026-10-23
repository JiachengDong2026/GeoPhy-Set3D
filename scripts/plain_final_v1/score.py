"""Separate CPU process: recompute every sealed field with official GT64 metrics."""
from context import *


def main():
    backend();verify_hashes(verified(BASE/'protocol.json')['sources']);seal=verified(BASE/'all_predictions_sealed.json')
    assert seal['all_predictions_saved_before_GT64'] and sha(BASE/'inference/measurement.jsonl')==seal['measurement_sha256']
    manifest=verified(BASE/'prediction_manifest.json');cases=verified(BASE/'cases.json');source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    archived={(r['method'],r['condition'],r['draw'],r['case_id'],r['stage']):r for r in rows(HISTORY/'evaluation/test/metrics.jsonl') if r['method'] in CLASSICS}
    assert sha(HISTORY/'evaluation/test/metrics.jsonl')==verified(BASE/'comparison_manifest.json')['source_metrics']['sha256']
    measured={(r['condition'],r['draw'],r['case_id'],r['stage']):r['measurement'] for r in rows(BASE/'inference/measurement.jsonl')}
    previous={(r['method'],r['condition'],int(r['draw']),r['case_id'],r['stage']):r for r in csv.DictReader((TARGETED/'metrics.csv').open()) if r['method']==MAIN}
    known=read(HISTORY/'evaluation/test/gt_access.json')['GT64_sha256_after_seal'];records=[];flatrows=[];gt_hashes={};maximum=0.;compared=0;first=None
    assert not (BASE/'per_case_metrics.csv').exists()
    with io_guard(gt=True,predictions=manifest) as guard:
        for r in manifest:assert sha(r['prediction'])==r['prediction_sha256']
        for case in cases:
            path=DATA/source[case]['gt_volume_64_path'];assert source[case]['split']=='test'
            if first is None:first=now()
            target=np.load(path,allow_pickle=False).astype(np.float64).squeeze();gt_hashes[str(path)]=sha(path);assert gt_hashes[str(path)]==known[str(path)]
            for r in [r for r in manifest if r['case_id']==case]:
                with np.load(r['prediction'],allow_pickle=False) as z:
                    for stage in z.files:
                        c=z[stage].astype(np.float64);vol=official.volume_metrics(c,target);morph=official.safe_shape(c,target);morph['absolute_peak_ratio_error']=abs(morph['peak_ratio']-1)
                        key=r['method'],r['condition'],r['draw'],case,stage
                        measurement=archived[key]['measurement'] if r['method'] in CLASSICS else measured[r['condition'],r['draw'],case,stage]
                        values=flatten(vol,morph,measurement)
                        if key in archived:
                            old=flatten(archived[key]['volume'],archived[key]['morphology'])
                            for k,v in old.items():
                                if v is None:assert values[k] is None;continue
                                diff=abs(v-values[k]);maximum=max(maximum,diff);assert diff<1e-11,(key,k,diff)
                            compared+=1
                        if key in previous:
                            old=previous[key];mapping={'relative_l1':vol['relative_l1'],'relative_l2':vol['relative_l2'],'peak_ratio':morph['peak_ratio'],
                                'absolute_peak_ratio_error':morph['absolute_peak_ratio_error'],'peak_absolute_error':vol['peak_absolute_error'],
                                'peak_location_error_mm':vol['peak_location_error_mm'],'centroid_error_mm':vol['centroid_error_mm'],
                                'centroid_distance_mm':morph['centroid_distance_mm'],'Dice50':morph['dice_0.5']}
                            for k,v in mapping.items():diff=abs(v-float(old[k]));maximum=max(maximum,diff);assert diff<1e-11
                            compared+=1
                        rr=dict(method=r['method'],stage=stage,condition=r['condition'],draw=r['draw'],case_id=case,
                            volume=vol,morphology=morph,measurement=measurement,reconstruct_seconds=r.get('reconstruct_seconds',r.get('seconds')),
                            peak_allocated_bytes=r['peak_allocated_bytes'],setup_seconds=r.get('setup_seconds'),solver_status=r.get('solver_status'),iterations=r.get('iterations'),
                            prediction=r['prediction'],prediction_sha256=r['prediction_sha256'],reused=r['reused'],measurement_origin='frozen historical diagnostics, identity audited +20 spotchecks' if r['method'] in CLASSICS else 'current GT-free projection diagnostics')
                        records.append(rr);flatrows.append(dict(method=r['method'],stage=stage,condition=r['condition'],draw=r['draw'],case_id=case,**values,
                            reconstruct_seconds=rr['reconstruct_seconds'],peak_allocated_bytes=rr['peak_allocated_bytes'],setup_seconds=rr['setup_seconds'],solver_status=rr['solver_status'],iterations=rr['iterations'],
                            prediction=r['prediction'],prediction_sha256=r['prediction_sha256'],reused=r['reused']))
            print('scored',case,len(records),'/3570',flush=True)
        assert not guard.denied and len(records)==3570
    jsonl(BASE/'metrics.jsonl',records);csv_write(BASE/'per_case_metrics.csv',flatrows)
    freeze(BASE/'GT64_access.json',dict(at=now(),first_GT64_at=first,seal_at=seal['at'],unique_GT64_files=len(gt_hashes),GT64_sha256=gt_hashes,
        fields=3570,historical_fields_recomputed_and_matched=compared,maximum_old_metric_difference=maximum,
        metrics_sha256=sha(BASE/'metrics.jsonl'),csv_sha256=sha(BASE/'per_case_metrics.csv'),GT128_read=False,Pool_B_read=False,audit=guard.summary()))


if __name__=='__main__':main()
