"""Durable UUID-bound resource-aware queue; no other process is signalled."""
from runtime import *
import argparse
import fcntl
import shutil
import time
import traceback

def snapshot():
    s=resources();append(BASE/'scheduling/resources.jsonl',s);return s

def free_devices(s):
    occupied={line.split(',')[0].strip() for line in s['gpu_processes'].splitlines() if line.strip()}
    answer=[]
    for line in s['gpus'].splitlines():
        i,uuid,name,total,used,free,util,mu=[x.strip() for x in line.split(',')]
        if uuid not in occupied and int(used)<100 and int(free)>=20000 and int(util)<=5:
            answer.append(dict(index=int(i),uuid=uuid,total_MiB=int(total),free_MiB=int(free)))
    return answer

def done(key):
    p=BASE/'scheduling'/f'{key}.json'
    if not p.exists():return False
    r=read(p)
    if r['state']=='failed':raise RuntimeError('Preserved failed task; inspect '+str(p))
    if r['state']=='running':
        # A controller crash must not launch a second copy of a surviving worker.
        try:os.kill(r['pid'],0)
        except ProcessLookupError:raise RuntimeError('Orphan task requires inspection/resume: '+str(p))
        raise RuntimeError('Worker still alive from previous controller: '+str(p))
    return r['state']=='complete'

def queue(tasks):
    pending=[t for t in tasks if not done(t['key'])];running={};stable={};failed=[]
    while pending or running:
        s=snapshot()
        for key,job in list(running.items()):
            proc=job['proc'];code=proc.poll()
            if code is None:continue
            record=job['record'];record.update(state='complete' if code==0 else 'failed',exit_code=code,finished_at=now(),wall_seconds=time.monotonic()-job['started'])
            write(BASE/'scheduling'/f'{key}.json',record);append(BASE/'scheduling/events.jsonl',record)
            job['log'].close();job['lock'].close();del running[key]
            if code!=0:failed.append(key)
        if failed:
            pending=[]
            if not running:raise RuntimeError('Failed tasks retained: '+str(failed))
        occupied={j['record']['gpu_uuid'] for j in running.values()}
        devices=free_devices(s)
        current={d['uuid'] for d in devices}
        stable={u:stable.get(u,0)+1 for u in current}
        for dev in devices:
            if not pending or dev['uuid'] in occupied or stable[dev['uuid']]<2:continue
            # Reserve substantial host memory/disk and confirm fresh GPU state.
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
            if mem['MemAvailable']<16*1024**3 or shutil.disk_usage(BASE).free<20*1024**3:continue
            lockdir=BASE.parent/'gpu_locks';lockdir.mkdir(exist_ok=True)
            lock=(lockdir/(dev['uuid']+'.lock')).open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            fresh=snapshot()
            if dev['uuid'] not in {d['uuid'] for d in free_devices(fresh)}:lock.close();continue
            task=pending.pop(0);key=task['key'];path=BASE/'scheduling'/f'{key}.log'
            log=path.open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,*task['args']]
            proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(key=key,state='running',started_at=now(),pid=proc.pid,controller_pid=os.getpid(),gpu_uuid=dev['uuid'],physical_index_at_start=dev['index'],
                visible_device='cuda:0',memory_reservation_MiB=20000,command=cmd,log=str(path),resource_snapshot_at_start=fresh['at'])
            write(BASE/'scheduling'/f'{key}.json',record);append(BASE/'scheduling/events.jsonl',record)
            running[key]=dict(proc=proc,record=record,log=log,lock=lock,started=time.monotonic());occupied.add(dev['uuid'])
            print('Started',key,'on',dev['uuid'],flush=True)
        write(BASE/'queue_status.json',dict(at=now(),pending=[t['key'] for t in pending],running={k:j['record'] for k,j in running.items()},failed=failed,poll_seconds=15))
        if pending or running:time.sleep(15)

def cpu(name):
    log=BASE/'scheduling'/f'{name}.log'
    with log.open('a') as f:
        result=subprocess.run([sys.executable,*job_args(name+'.py')],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
    if result.returncode:raise RuntimeError(f'{name} failed: {log}')

def main():
    from register import setup
    setup();BASE.joinpath('scheduling').mkdir(exist_ok=True)
    lock=(BASE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        write(BASE/'pipeline_status.json',dict(state='auditing',at=now(),pid=os.getpid()))
        cpu('reuse')
        queue([dict(key='correctness_checks',args=job_args('checks.py'))])
        verify_hashes(verified(BASE/'design.json')['sources'])
        cpu('report')
        write(BASE/'pipeline_status.json',dict(state='training',at=now(),pid=os.getpid()))
        runs=verified(BASE/'design.json')['runs']
        queue([dict(key=Path(r).name,args=job_args('train_worker.py','--target',r)) for r in runs])
        cpu('methods')
        cpu('report')
        write(BASE/'pipeline_status.json',dict(state='development_prediction',at=now(),pid=os.getpid()))
        queue([dict(key='predict_'+Path(r).name,args=job_args('evaluate.py','predict','--key',Path(r).name)) for r in runs])
        queue([dict(key='development_score',args=job_args('evaluate.py','score'))])
        queue([dict(key='matched_benchmark',args=job_args('evaluate.py','benchmark'))])
        cpu('analysis');cpu('verify');cpu('report')
        freeze(BASE/'acceptance.json',dict(at=now(),completed=True,development_only=True,evidence={str(p):sha(p) for p in [BASE/'design.json',BASE/'reuse_audit.json',BASE/'checks/passed.json',BASE/'evaluation/val/metrics.jsonl',BASE/'analysis/summary.json',BASE/'analysis/independent_verification.json',BASE/'benchmark.json',REPORT]}))
        write(BASE/'pipeline_status.json',dict(state='complete',at=now(),pid=os.getpid()))
    except BaseException:
        write(BASE/'pipeline_status.json',dict(state='failed',at=now(),pid=os.getpid(),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
