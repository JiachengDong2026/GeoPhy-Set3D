"""Optional CPU scoring of disjoint future layouts using frozen scoring functions.

This adapter changes no numerical/statistical definition. The primary analyzer
checks each completed shard's manifest/CSV hashes and merges it as usual.
"""
from diagnostic import *
from analyze import canonical_metrics,independent_metrics,write_csv


def main(indices):
    backend()
    seal=verified(BASE/'all_predictions_sealed.json')
    verify_hashes(seal['manifest_hashes'])
    verify_hashes(verified(BASE/'experiment_registry.json')['sources'])
    amendment=verified(BASE/'cpu_scoring_execution.json')
    assert amendment['adapter_sha256']==sha(__file__)
    allowed={i for job in amendment['jobs'] for i in job['layouts']}
    assert set(indices)<=allowed
    sources={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='test'}
    with Guard(gt_splits=['test']) as guard:
        gt={case:np.load(DATA/src['gt_volume_64_path'],allow_pickle=False).squeeze().astype(np.float64) for case,src in sources.items()}
        for i in indices:
            root=BASE/'predictions'/f'layout{i:02d}'
            out=BASE/'analysis';out.mkdir(exist_ok=True)
            dest=out/f'layout{i:02d}_metrics.csv';verification=out/f'layout{i:02d}_verified.json'
            if verification.exists():
                assert sha(dest)==verified(verification)['metrics_sha256'];continue
            assert not dest.exists(),'Do not overwrite a partially scored primary/shard layout'
            task_lock=(out/f'layout{i:02d}_cpu.lock').open('a')
            fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            records=rows(root/'manifest.jsonl');current=[];errors={};count=0
            for record in records:
                pred=root/record['prediction'];assert sha(pred)==record['sha256'];g=gt[record['case_id']]
                with np.load(pred,allow_pickle=False) as z:
                    for stage in ('C0','C1'):
                        p=z[stage].astype(np.float64)
                        metrics=canonical_metrics(p,g);ind=independent_metrics(p,g)
                        for key,value in ind.items():
                            if value is None:assert metrics[key] is None
                            else:
                                assert math.isclose(value,metrics[key],rel_tol=1e-8,abs_tol=1e-9),(i,record,key)
                                errors[key]=max(errors.get(key,0.),abs(value-metrics[key]))
                        current.append(dict(case_id=record['case_id'],case_index=record['case_index'],
                            condition=record['condition'],draw=record['draw'],views=record['views'],stage=stage,
                            experiment=record['experiment'],permutation_index=record['permutation_index'],
                            permutation=json.dumps(record['permutation']),checkpoint_sha256=EXPECTED,
                            prediction_sha256=record['sha256'],prediction_path=str(pred),**metrics))
                        count+=1
            write_csv(dest,current)
            freeze(verification,dict(at=now(),passed=True,fields=count,metrics_max_abs_difference=errors,
                metrics_sha256=sha(dest),manifest_sha256=sha(root/'manifest.jsonl'),
                CPU_shard=True,adapter_sha256=sha(__file__),frozen_scoring_sources=amendment['numerical_sources']))
            task_lock.close()
            print('CPU SHARD VERIFIED',i,count,flush=True)
        assert not guard.denied
    freeze(BASE/f'CPU_shard_{indices[0]:02d}_GT_audit.json',dict(at=now(),layouts=indices,
        after_seal_sha256=sha(BASE/'all_predictions_sealed.json'),GT128_read=False,
        model_forward_calls=0,no_training=True,audit=guard.summary()))


if __name__=='__main__':main([int(i) for i in sys.argv[1:]])
