"""SART method dispatch, seal validation and data-boundary regression checks."""
from common import *
import tempfile
from unittest.mock import patch
import numpy as np
import evaluation as ev


def main():
    m=dict(key='SART',kind='sart',config=dict(iterations=200))
    with patch('sart.method',return_value=m),patch('sart.work') as worker:
        ev.predict_neural(m,'test');worker.assert_called_once_with('test')
    assert ev.target_for(m,'test')==OUT/'classical/test/SART/frozen'
    case='fixture';settings=[dict(condition='Val-A-V4',draw=0,input_ids={case:[1,2,3,4]},heldout_ids={case:[5,6,7,8]})]
    registration=dict(cases=dict(val=[case]))
    with tempfile.TemporaryDirectory(dir=OUT/'sart') as tmp:
        d=Path(tmp);p=d/'prediction.npz';np.savez_compressed(p,C1=np.zeros((64,)*3,dtype=np.float32))
        row=dict(condition='Val-A-V4',draw=0,case_id=case,input_ids=[1,2,3,4],heldout_ids=[5,6,7,8],prediction=p.name,
            sha256=sha(p),config=m['config'],iterations=200)
        def save(rr):
            jsonl(d/'manifest.jsonl',rr);write(d/'complete.json',dict(manifest_sha256=sha(d/'manifest.jsonl'),reconstructions=1,fields=1,audit=dict(denied=[],counts={}),config=m['config'],method='SART',split='val'))
        def fixture(path):
            if Path(path).name=='registration.json':return registration
            if Path(path).name=='layouts.json':return dict(val=settings)
            raise AssertionError(path)
        with patch.object(ev,'target_for',return_value=d),patch.object(ev,'verified',side_effect=fixture):
            save([row]);ev.verify_predictions(m,'val',arrays=True)
            negatives=[[],[row,row],[dict(row,config=dict(iterations=100))],[dict(row,heldout_ids=[1,6,7,8])],[dict(row,sha256='0'*64)]]
            for rr in negatives:
                save(rr);rejected=False
                try:ev.verify_predictions(m,'val',arrays=True)
                except AssertionError:rejected=True
                assert rejected
    denied=[];source=rows(DATA/'source_index.jsonl');test=next(r for r in source if r['split']=='test')
    for label,path in [('test_projection',DATA/test['train_projection_path']),('test_GT64',DATA/test['gt_volume_64_path']),('GT128',Path(test['source_target_path']))]:
        with AccessGuard(splits=['train']) as guard:
            rejected=False
            try:
                with path.open('rb') as f:f.read(1)
            except PermissionError:rejected=True
            assert rejected and guard.denied
            denied.append(dict(label=label,denied_before_read=True))
    for p in (OUT/'training').glob('*/protocol.json'):verify_hashes(verified(p)['sources'])
    assert not (OUT/'final_lock.json').exists() and not (OUT/'final_method_choices_closed.json').exists()
    freeze(OUT/'sart/integration_checks.json',dict(at=now(),passed=True,dispatch=True,seal_validation=True,
        corrupt_and_missing_predictions_rejected=True,boundary_denials=denied,training_sources_unchanged=True,
        final_methods=13,final_reconstructions=7735,final_fields=13090,primary_comparators=12,source_sha256=sha(__file__)))
    print('SART integration checks PASSED; final test files denied before read')


if __name__=='__main__':main()
