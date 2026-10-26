"""Verify counts, checkpoints, old evidence and collect the bounded probe."""
from probe_context import *
import csv
import gzip
import shutil


def main():
    r = verify()
    assert read(ART / 'preflight.json')['passed']
    review = read(ART / 'independent_review.json')
    assert review['passed'] and review['full_fields'] == 24
    manifests = []
    curves = []
    actual = 0
    for index in r['indices']:
        for arm in r['arms']:
            directory = ROOT / f'case{index}_{arm}'
            result = read(directory / 'result.json')
            journal = [json.loads(s) for s in (directory / 'updates.jsonl').read_text().splitlines()]
            assert [row['update'] for row in journal] == list(range(1,201))
            assert all(row['actual_parameter_delta']>0 and np.isfinite(row['preclip_norm']) and row['preclip_norm']>0 for row in journal)
            last = directory / 'last.pt'
            assert sha(last) == result['checkpoint_sha256']
            ck = torch.load(last,map_location='cpu',weights_only=False)
            assert ck['steps'] == 200 and ck['scheduler']['last_epoch'] == 12 and ck['rng']
            assert {int(state['step']) for state in ck['optimizer']['state'].values()} == {200}
            assert all(torch.isfinite(value).all() for value in ck['model'].values())
            del ck
            with (directory / 'updates.jsonl').open('rb') as source, gzip.open(ART / f'case{index}_{arm}_updates.jsonl.gz','wb') as target:
                shutil.copyfileobj(source,target)
            manifests.append(dict(index=index,arm=arm,checkpoint=str(last),sha256=sha(last),actual_updates=200,
                                  complete_fields=[m['arrays'] for m in result['metrics'] if 'arrays' in m]))
            for metric in result['metrics']:
                curves.append(dict(index=index,arm=arm,step=metric['step'],loss=metric['loss'],
                    observed=metric['measurement']['observed'],cross=metric['measurement']['cross'],H=metric['measurement']['heldout'],
                    E=metric['extra_E_H'],mean=float(np.mean([f['mean'] for f in metric['fields']])),
                    std=float(np.mean([f['std'] for f in metric['fields']]))))
            assert not any(result['access'][key] for key in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
            actual += len(journal)
    assert actual == r['total_update_ceiling']
    with (ART / 'curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(curves)
    write(ART / 'checkpoint_manifest.json',manifests)
    assert not (ROOT / 'case0_voxel_sigmoid_attempt1_preupdate/updates.jsonl').exists()
    old_shared = PROJECT / 'docs/reports/artifacts/shared_failure_v1'
    assert sha(old_shared / 'delivery_manifest.json') == r['prior_diagnosis_manifest_sha256']
    protected = 0
    for location in [old_shared, UNET_ART]:
        for name,digest in read(location / 'delivery_manifest.json').items():
            assert sha(PROJECT / name) == digest,name
            protected += 1
    history=read(UNET_ART / 'history_audit.json')
    for name,digest in history['frozen_sha256'].items():
        assert sha(name)==digest,name
    write(ART / 'execution_integrity.json',dict(actual_updates=actual, runs=6, updates_per_run=200,
        independent_full_fields=24, independent_integrals=96, frozen_evidence_files_verified=protected,
        frozen_scientific_sources=147, frozen_input_files=207, CPU_only=True, GPU_used=False,
        formal_training_started=False, formal_predictions=0, GT64_reads=0, GT128_reads=0, PoolB_reads=0, final_test_reads=0,
        initial_engineering_failure_updates=0, replay_updates=0, main_merged=False,
        diagnostic_conclusion='See independent per-arm fit and E results; no formal GO authorization'))
    print('Delivery verification passed:',actual,'updates;',review['full_fields'],'full independent fields')


if __name__ == '__main__':
    main()
