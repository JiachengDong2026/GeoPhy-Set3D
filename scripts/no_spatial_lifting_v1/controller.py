"""Existing UUID reservation helpers; two independent full-budget seed jobs."""
from runtime import *
import fcntl
import time
import traceback


def progress(state,**extra):
    runs=[]
    for seed in SEEDS:
        p=run_protocol(seed)/'status.json'
        runs.append(dict(seed=seed,**(read(p) if p.exists() else dict(state='registered',epoch=0))))
    write(BASE/'status.json',dict(at=now(),state=state,controller_pid=os.getpid(),runs=runs,**extra))


def train_all(s):
    pending=list(SEEDS);active={};stable={};failures=[]
    while pending or active:
        for seed,item in list(active.items()):
            if not s.done(item):continue
            try:
                s.finish(item)
                summary=read(run_protocol(seed)/'summary.json')
                assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900
            except Exception:
                failures.append(seed)
            del active[seed]
        if failures:pending=[]
        snap=s.snapshot();occupied={item['record']['gpu_uuid'] for item in active.values()}
        for d in s.devices(snap):
            uuid=d['uuid'];stable[uuid]=stable.get(uuid,0)+1 if d['idle'] and uuid not in occupied else 0
            if not pending or len(active)>=2 or stable[uuid]<2:continue
            lock=s.reserve(uuid)
            if lock is None:continue
            seed=pending.pop(0);target=run_protocol(seed)
            assert not (target/'checkpoints/last.pt').exists(),'No silent retry/resume of formal task'
            verify_hashes(verified(target/'protocol.json')['sources'])
            active[seed]=s.launch(f'train_seed{seed}',[sys.executable,str(HERE/'worker.py'),'--target',str(target)],uuid,lock)
            occupied.add(uuid)
        progress('training',pending=pending,active={str(seed):item['record'] for seed,item in active.items()},failures=failures)
        if pending or active:time.sleep(15)
    assert not failures,failures
    evidence={str(run_protocol(seed)/n):sha(run_protocol(seed)/n) for seed in SEEDS for n in ('summary.json','metrics.jsonl','checkpoints/best.pt','checkpoints/last.pt')}
    freeze(BASE/'training_complete.json',dict(at=now(),passed=True,seeds=list(SEEDS),evidence=evidence))


def main():
    (BASE/'scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            s=scheduler();s.run_gpu_once('engineering',[sys.executable,str(HERE/'checks.py')],BASE/'engineering_checks.json')
            assert verified(BASE/'engineering_checks.json')['passed']
            if not (BASE/'training_complete.json').exists():train_all(s)
            progress('training_complete_preparing_evaluation')
            # Evaluation implementation is independently locked before inference/GT.
            while not (HERE/'evaluate.py').exists():time.sleep(15)
            with (BASE/'scheduling/evaluation.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate.py'),'register'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            for seed in SEEDS:
                s.run_gpu_once(f'predict_seed{seed}',[sys.executable,str(HERE/'evaluate.py'),'predict','--seed',str(seed)],BASE/'evaluation'/str(seed)/'complete.json')
            s.run_gpu_once('benchmark',[sys.executable,str(HERE/'evaluate.py'),'benchmark'],BASE/'benchmark.json')
            progress('sealing_and_scoring')
            with (BASE/'scheduling/scoring.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate.py'),'score'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert verified(BASE/'acceptance.json')['completed'];progress('complete')
        except BaseException:
            progress('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
