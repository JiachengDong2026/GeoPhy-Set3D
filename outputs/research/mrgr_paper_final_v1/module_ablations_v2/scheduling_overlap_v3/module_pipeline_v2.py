"""Queue module checks/training behind the live scheduler, without stopping workers."""
from common import *
import fcntl
from scheduler import run_jobs
from module_registry_v2 import BASE,methods


def status(state,**kw):
    v=dict(at=now(),pid=os.getpid(),state=state,**kw);write(BASE/'status.json',v);append(BASE/'events.jsonl',v);print(json.dumps(v),flush=True)


def main():
    mutex=(BASE/'controller.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=verified(BASE/'registration.json');verify_hashes(reg['protocol_hashes'])
    status('queued_checks_and_eight_trainings',scheduler='same exclusive UUID scheduler; begins when preceding phase releases lock')
    queue=verified(BASE/'queue.json')
    run_jobs(queue[:1],'module_checks_v2')
    assert verified(BASE/'checks.json')['passed']
    run_jobs(queue[1:],'module_ablations_v2')
    assert verified(BASE/'checks.json')['passed'];completed=methods(wait=False)
    evidence={str(BASE/'checks.json'):sha(BASE/'checks.json'),str(BASE/'registration.json'):sha(BASE/'registration.json')}
    for m in completed:
        for n in ('protocol.json','summary.json'):p=Path(m['run'])/n;evidence[str(p)]=sha(p)
    freeze(BASE/'training_complete.json',dict(at=now(),passed=True,runs=8,epochs_each=300,evidence=evidence))
    status('complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
