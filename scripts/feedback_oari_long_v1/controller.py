"""One reserved dynamic GPU for a fair short pilot; cooperates with C/D locks."""
from runtime import *
import fcntl
import shutil
import time
import traceback

def snapshot():
    s=resources();append(BASE/'scheduling/resources.jsonl',s);return s

def devices(s):
    busy={line.split(',')[0].strip() for line in s['gpu_processes'].splitlines() if line.strip()};answer=[]
    for line in s['gpus'].splitlines():
        i,uuid,name,total,used,free,util,mu=[x.strip() for x in line.split(',')]
        answer.append(dict(index=int(i),uuid=uuid,free=int(free),used=int(used),util=int(util),
            idle=uuid not in busy and int(free)>=20000 and int(used)<100 and int(util)<=5))
    return sorted(answer,key=lambda d:(not d['idle'],-d['free']))

def reserve(s):
    GPU_LOCKS.mkdir(parents=True,exist_ok=True)
    for dev in devices(s):
        lock=(GPU_LOCKS/(dev['uuid']+'.lock')).open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        append(BASE/'scheduling/events.jsonl',dict(event='reservation',at=now(),gpu_uuid=dev['uuid'],index=dev['index'],idle=dev['idle'],note='cooperative queue priority only; no CUDA allocation and no external process signalling'))
        return dev,lock
    return None,None

def wait_idle(dev,lock):
    stable=0
    while True:
        s=snapshot();dd=devices(s);current=next(d for d in dd if d['uuid']==dev['uuid'])
        if not current['idle']:
            # Transfer the reservation if a different truly idle card appears.
            candidates=[d for d in dd if d['idle'] and d['uuid']!=dev['uuid']]
            for other in candidates:
                candidate=(GPU_LOCKS/(other['uuid']+'.lock')).open('a')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close();continue
                lock.close();dev=other;lock=candidate;current=other;stable=0
                append(BASE/'scheduling/events.jsonl',dict(event='reservation_transfer',at=now(),gpu_uuid=dev['uuid']));break
        stable=stable+1 if current['idle'] else 0
        write(BASE/'status.json',dict(state='waiting_for_idle_GPU',at=now(),pid=os.getpid(),reserved_uuid=dev['uuid'],snapshot_gpu=current,poll_seconds=15,CPU_checks_passed=(BASE/'cpu_checks.json').exists()))
        if stable>=2:
            fresh=next(d for d in devices(snapshot()) if d['uuid']==dev['uuid'])
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
            if fresh['idle'] and mem['MemAvailable']>=16*1024**3 and shutil.disk_usage(BASE).free>=20*1024**3:return dev,lock
            stable=0
        time.sleep(15)

def run_task(key,command,dev=None):
    path=BASE/'scheduling'/f'{key}.json'
    if path.exists():
        record=read(path)
        if record['state']=='complete':return
        raise RuntimeError('Preserve failed/orphan task for inspection: '+str(path))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='' if dev is None else dev['uuid'])
    log_path=BASE/'scheduling'/f'{key}.log';started=time.monotonic()
    with log_path.open('x') as log:
        proc=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,env=env)
        record=dict(key=key,state='running',started_at=now(),pid=proc.pid,gpu_uuid=None if dev is None else dev['uuid'],command=command,log=str(log_path))
        write(path,record);append(BASE/'scheduling/events.jsonl',record)
        while proc.poll() is None:
            snapshot();write(BASE/'status.json',dict(state='running_'+key,at=now(),pid=os.getpid(),task=record));time.sleep(15)
        record.update(state='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode,finished_at=now(),wall_seconds=time.monotonic()-started)
        write(path,record);append(BASE/'scheduling/events.jsonl',record)
        if proc.returncode:raise RuntimeError('Task failed; retained '+str(log_path))

def main():
    from register import main as register
    register();BASE.joinpath('scheduling').mkdir(exist_ok=True)
    controller_lock=(BASE/'controller.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dev=None;lock=None
    try:
        if (BASE/'acceptance.json').exists():verify_hashes(verified(BASE/'acceptance.json')['evidence']);return
        while lock is None:
            dev,lock=reserve(snapshot())
            if lock is None:time.sleep(15)
        verify_hashes(verified(BASE/'design.json')['sources']);verified(BASE/'migration_verification.json');verified(BASE/'gpu_checks.json')
        from long_report import main as report
        report()
        # Identical architecture reuses the passed pilot gradient and recovery checks.
        for target in verified(BASE/'design.json')['runs']:
            report();dev,lock=wait_idle(dev,lock);run_task(Path(target).name,args('worker.py','--target',target),dev)
        report();dev,lock=wait_idle(dev,lock);run_task('development_predict',args('evaluate.py','predict'),dev)
        dev,lock=wait_idle(dev,lock);run_task('development_score',args('evaluate.py','score'),dev)
        lock.close();lock=None
        run_task('statistics_and_verification',args('analyze.py'))
        write(BASE/'status.json',dict(state='long_training_complete',at=now(),pid=os.getpid(),report=str(REPORT)))
        report()
        freeze(BASE/'acceptance.json',dict(at=now(),completed_long_training=True,full300=True,evidence={str(p):sha(p) for p in [BASE/'design.json',BASE/'migration_verification.json',BASE/'gpu_checks.json',BASE/'independent_verification.json',BASE/'analysis/summary.json',REPORT]}))
        write(BASE/'status.json',dict(state='long_training_complete',at=now(),pid=os.getpid(),report=str(REPORT)))
    except BaseException:
        write(BASE/'status.json',dict(state='failed',at=now(),pid=os.getpid(),traceback=traceback.format_exc()))
        try:
            from long_report import main as report
            report()
        except Exception:pass
        raise
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':main()
