"""Queue module checks/training behind the live scheduler, without stopping workers."""
from common import *
import fcntl
from scheduler import run_jobs
from module_registry_v2 import BASE,methods


def status(state,**kw):
    v=dict(at=now(),pid=os.getpid(),state=state,**kw);write(BASE/'status.json',v);append(BASE/'events.jsonl',v);print(json.dumps(v),flush=True)


def overlap_preflight():
    # The original scheduler owns active jobs, but has no unlaunched jobs left.
    # Its next stage blocks in module_registry.methods until we finish all jobs.
    assert read(OUT/'pipeline_status.json')['state']=='formal_ablation_training'
    for arm,seed in [('multiscale',20261019),('single_scale',20261011),('no_feedback',20261011),
                     ('no_feedback',20261019),('single_scale',20261019),('no_consistency',20261011),('no_consistency',20261019)]:
        record=read(OUT/'scheduling/formal_ablation_training'/f'{arm}_seed{seed}.json')
        assert record['pid'] and (record.get('exit_code') in (None,0))
    assert not (OUT/'final_lock.json').exists()
    assert not (OUT/'stages/ablation_development_predictions.json').exists()


def main():
    mutex=(BASE/'controller.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=verified(BASE/'registration.json');verify_hashes(reg['protocol_hashes'])
    overlap_preflight()
    status('module_GPU_checks',scheduler='idle UUIDs only; independent phase lock after all original jobs launched; original downstream evaluation gated on module completion')
    queue=verified(BASE/'queue.json')
    run_jobs(queue[:1],'module_checks_v2',lock_path=BASE/'scheduler.lock',preflight=overlap_preflight)
    assert verified(BASE/'checks.json')['passed']
    status('module_ablation_training')
    run_jobs(queue[1:],'module_ablations_v2',lock_path=BASE/'scheduler.lock',preflight=overlap_preflight)
    assert verified(BASE/'checks.json')['passed'];completed=methods(wait=False)
    evidence={str(BASE/'checks.json'):sha(BASE/'checks.json'),str(BASE/'registration.json'):sha(BASE/'registration.json')}
    for m in completed:
        for n in ('protocol.json','summary.json'):p=Path(m['run'])/n;evidence[str(p)]=sha(p)
    freeze(BASE/'training_complete.json',dict(at=now(),passed=True,runs=8,epochs_each=300,evidence=evidence))
    status('complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
