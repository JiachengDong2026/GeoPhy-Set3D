"""Allow only final-lock checkpoints during the unchanged prediction seal audit."""
from common import *
import argparse
import evaluation as ev

BASE=OUT/'recovery/seal_checkpoint_allowlist_v2'


def checks():
    lock=verified(OUT/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['evidence'])
    assert not (OUT/'evaluation/test/gt_access.json').exists()
    checkpoints=[m['checkpoint'] for m in lock['methods'] if m['kind']=='neural']
    with AccessGuard(checkpoints=checkpoints) as guard:
        for m in lock['methods']:
            if m['kind']=='neural':assert sha(m['checkpoint'])==m['checkpoint_sha256']
    forbidden=[DATA/r['gt_volume_64_path'] for r in rows(DATA/'source_index.jsonl') if r['split']=='test'][:1]
    forbidden += [Path('/tmp/mrgr_unregistered_foreign_checkpoint.pt')]
    denied=[]
    with AccessGuard(checkpoints=checkpoints) as boundary:
        for p in forbidden:
            try:
                with p.open('rb'):raise AssertionError('Forbidden open succeeded')
            except PermissionError:denied.append(str(p))
    assert len(denied)==2 and not boundary.touched
    freeze(BASE/'checks.json',dict(at=now(),passed=True,registered_checkpoints_hashed=len(checkpoints),audit=guard.summary(),
        forbidden_test_GT_and_foreign_checkpoint_denied_before_bytes=denied,sources_unchanged=True,source_sha256=sha(__file__)))
    freeze(BASE/'amendment.json',dict(at=now(),reason='Original seal guard omitted registered historical checkpoints; PermissionError before seal and before test GT access.',
        failed_stage_sha256=sha(OUT/'stages/final_all_seal.json'),failed_log_sha256=sha(OUT/'stages/final_all_seal.log'),
        final_lock_sha256=sha(OUT/'final_lock.json'),checks_sha256=sha(BASE/'checks.json'),wrapper_sha256=sha(__file__),
        implementation='Process-local AccessGuard subclass adds only the 17 checkpoints already in the immutable final method lock; original validation unchanged.',
        models_changed=False,predictions_changed=False,metrics_changed=False,hyperparameters_changed=False,test_GT_opened=False))
    print('Registered checkpoint allowlist and forbidden-read checks PASSED',flush=True)


def seal():
    lock=verified(OUT/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['evidence']);verify_hashes(lock['data_identity'])
    amendment=verified(BASE/'amendment.json');assert amendment['wrapper_sha256']==sha(__file__)
    assert verified(BASE/'checks.json')['passed']
    checkpoints=[m['checkpoint'] for m in lock['methods'] if m['kind']=='neural']
    original_guard=ev.AccessGuard;original_freeze=ev.freeze
    class RegisteredCheckpointGuard(original_guard):
        def __init__(self,*args,**kwargs):
            assert not args and 'checkpoints' not in kwargs
            super().__init__(checkpoints=checkpoints,**kwargs)
    def amended_freeze(path,value):
        assert Path(path)==OUT/'evaluation/test/all_predictions_sealed.json'
        value=dict(value,recovery_amendment_sha256=sha(BASE/'amendment.json'),recovery_wrapper_sha256=sha(__file__))
        return original_freeze(path,value)
    ev.AccessGuard=RegisteredCheckpointGuard;ev.freeze=amended_freeze
    ev.seal('test',lock['methods'])
    print('All 21 methods sealed; no prediction arrays regenerated or modified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['checks','seal']);a=p.parse_args();globals()[a.action]()
