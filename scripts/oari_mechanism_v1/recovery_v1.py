"""Explicit preserved-failure recovery; same sampling/model, numerical checker only."""
from runtime import *
import fcntl


def register():
    old=BASE/'prediction_manifest.jsonl';rr=rows(old);assert len(rr)==12
    for r in rr:assert sha(BASE/r['prediction'])==r['sha256']
    # Renaming preserves every original record and leaves all original prediction
    # paths valid. New predictions use a separate directory, never overwrite.
    archived=BASE/'failed_attempt1_prediction_manifest.jsonl';assert not archived.exists();old.rename(archived)
    freeze(BASE/'numerical_recovery_v1.json',dict(at=now(),reason='independent SciPy Jensen-Shannon distance sqrt of negative floating roundoff near0; reproduced on GT-free synthetic data',
        original_protocol_sha256=sha(BASE/'protocol.json'),original_sources_unchanged=True,
        sources={str(HERE/n):sha(HERE/n) for n in ('probe_recovery_v1.py','finish_recovery_v1.py','recovery_v1.py')},
        original_failure_files={str(BASE/n):sha(BASE/n) for n in ('probe_failure.json','controller_failure.json')},
        original_forward_calls=12,original_manifest=str(archived),original_manifest_sha256=sha(archived),recovery_forward_calls=45,total_planned_calls=57,
        only_changes=['independent JS via SciPy rel_entr without sqrt; nonnegative roundoff correction bounded at1e-12','separate recovery output directory','explicit failure and total-cost accounting'],
        unchanged=['checkpoint','model structure/weights','A/P1/P2 semantics','case indices/layouts/draws','case-level statistics','all training jobs'],
        repeats_reason='attention summaries existed only in process memory at failure; archived volumes cannot reconstruct missing attention',GT64_read=False,GT128_read=False))


def main():
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register();s=scheduler();s.run_gpu_once('probe_recovery_v1',[sys.executable,str(HERE/'probe_recovery_v1.py')],BASE/'probe_complete.json')
        with (BASE/'scheduling/finish_recovery_v1.log').open('x') as log:
            subprocess.run([sys.executable,str(HERE/'finish_recovery_v1.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert verified(BASE/'acceptance.json')['completed'];write(BASE/'status.json',dict(at=now(),state='complete',preserved_failure_recovery=True))


if __name__=='__main__':main()
