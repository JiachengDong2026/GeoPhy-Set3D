"""Durable UUID-locked queue; stop future launches after any engineering failure."""
from stage import *
import fcntl
import time
import traceback


def command(script,*args):return [sys.executable,str(HERE/script),*map(str,args)]


def run():
    BASE.joinpath('scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=scheduler()
        exclusion=BASE/'scheduling/resource_exclusions.json'
        if exclusion.exists():
            excluded=set(verified(exclusion)['excluded_gpu_uuids'])
            original_devices=s.devices
            s.devices=lambda snap:[dict(d,idle=False) if d['uuid'] in excluded else d for d in original_devices(snap)]
        try:
            verify_hashes(verified(BASE/'protocol.json')['sources'])
            assert verified(BASE/'checks/cpu.json')['passed']
            s.run_gpu_once('gpu_engineering',command('checks.py','gpu'),BASE/'checks/gpu.json')
            jobs=[dict(key=f'{a}_seed{seed}',command=command('stage.py','train','--target',BASE/'training'/f'{a}_seed{seed}'),
                artifact=BASE/'training'/f'{a}_seed{seed}'/'summary.json') for a,seed in TASKS]
            jobs.append(dict(key='frozen_obs_token_seed20261019',command=command('predict.py','frozen'),artifact=BASE/'frozen/complete.json'))
            active=[];stable={};failed=False;expanded=(BASE/'scheduling/concurrency_expansion.json').exists()
            while jobs or active:
                jobs=[j for j in jobs if not j['artifact'].exists()]
                snap=s.snapshot()
                for item,job in list(active):
                    if not s.done(item):continue
                    valid=item['process'].returncode==0 and job['artifact'].exists()
                    try:s.finish(item,valid=valid)
                    except AssertionError:failed=True
                    active.remove((item,job))
                if failed:jobs=[]
                if not expanded:
                    prefixes=[BASE/'training'/f'{a}_seed{seed}'/'status.json' for a,seed in TASKS[:2]]
                    if all(p.exists() and read(p).get('epoch',0)>=3 for p in prefixes) and s.host_ok(expand=True):
                        expanded=True
                        freeze(BASE/'scheduling/concurrency_expansion.json',dict(at=now(),initial_limit=2,new_limit=4,
                            rule='firsttwo training jobs completed>=3epochs; host>=64GiB RAM/150GiB scratch and load<75%cores',snapshot=snap))
                for d in s.devices(snap):
                    stable[d['uuid']]=stable.get(d['uuid'],0)+1 if d['idle'] else 0
                    if not jobs or stable[d['uuid']]<2:continue
                    if len(active)>=(4 if expanded else 2):break
                    if any(i['record']['gpu_uuid']==d['uuid'] for i,_ in active):continue
                    while jobs and jobs[0]['artifact'].exists():jobs.pop(0)
                    if not jobs:break
                    gpu_lock=s.reserve(d['uuid'])
                    if gpu_lock is None:continue
                    job=jobs.pop(0);item=s.launch(job['key'],job['command'],d['uuid'],gpu_lock)
                    active.append((item,job));stable[d['uuid']]=0
                statuses={}
                for arm,seed in TASKS:
                    p=BASE/'training'/f'{arm}_seed{seed}'/'status.json'
                    if p.exists():statuses[p.parent.name]=read(p)
                write(BASE/'status.json',dict(at=now(),state='failed_waiting_for_siblings' if failed else 'running',
                    training=statuses,active=[i['record'] for i,_ in active],pending=[j['key'] for j in jobs],controller_pid=os.getpid()))
                if jobs or active:time.sleep(15)
            assert not failed,'One or more jobs failed; see preserved logs'
            if not (BASE/'training_complete.json').exists():
                freeze(BASE/'training_complete.json',dict(at=now(),completed=True,runs=[str(BASE/'training'/f'{a}_seed{s}') for a,s in TASKS]))
            for arm,seed in TASKS:
                key=f'{arm}_seed{seed}'
                s.run_gpu_once('predict_'+key,command('predict.py','trained','--key',key),BASE/'evaluation'/key/'complete.json')
            s.run_gpu_once('benchmark',command('predict.py','benchmark'),BASE/'benchmark.json')
            with (BASE/'scheduling/scoring.log').open('a') as log:
                subprocess.run(command('evaluate.py'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert verified(BASE/'acceptance.json')['completed']
            write(BASE/'status.json',dict(at=now(),state='complete',report=str(REPORT),controller_pid=os.getpid()))
        except BaseException:
            write(BASE/'status.json',dict(at=now(),state='failed',traceback=traceback.format_exc(),controller_pid=os.getpid()));raise


if __name__=='__main__':run()
