"""Scheduling amendment only: attach ongoing OARI and use a second idle GPU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'feedback_oari_long_v1'))
from runtime import *
import controller as serial
import fcntl
import time
import traceback

AMENDMENT=BASE/'parallel_scheduling_amendment.json'
PREFERRED='GPU-fdd41417-bc2c-971b-b84d-6464cf7c5749'

def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def lock_gpu(uuid):
    f=(GPU_LOCKS/(uuid+'.lock')).open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f

def complete(target):
    c=target/'continuation_complete.json'
    if not c.exists():return False
    r=verified(c);s=read(target/'summary.json')
    assert r['passed'] and r['prefix_metrics_exact'] and r['epochs']==s['epochs']==300
    assert r['updates']==s['updates']==24900
    assert sha(target/'checkpoints/last.pt')==r['last_checkpoint_sha256']
    assert sha(target/'summary.json')==r['summary_sha256']
    return True

def train_parallel():
    d=verified(BASE/'design.json');active={};locks={};pending=[];stable={}
    try:
        for name in d['runs']:
            target=Path(name);path=BASE/'scheduling'/f'{target.name}.json'
            if path.exists():
                r=read(path)
                if r['state']=='complete':assert complete(target);continue
                assert r['state']=='running' and alive(r['pid']),r
                cmd=(Path('/proc')/str(r['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                assert str(target) in cmd and 'feedback_oari_long_v1/worker.py' in cmd
                lock=lock_gpu(r['gpu_uuid']);assert lock is not None,'GPU lock held by another controller'
                locks[r['gpu_uuid']]=lock;active[target.name]=dict(target=target,record=r,proc=None)
                append(BASE/'scheduling/events.jsonl',dict(event='parallel_attach',at=now(),pid=r['pid'],gpu_uuid=r['gpu_uuid'],key=target.name))
            else:pending.append(target)
        while active or pending:
            for key,item in list(active.items()):
                r=item['record'];proc=item['proc']
                finished=not alive(r['pid']) if proc is None else proc.poll() is not None
                if not finished:continue
                ok=complete(item['target']) and (proc is None or proc.returncode==0)
                r.update(state='complete' if ok else 'failed',finished_at=now(),
                    exit_code=None if proc is None else proc.returncode,
                    completion_verified=ok,attached_process=proc is None)
                write(BASE/'scheduling'/f'{key}.json',r);append(BASE/'scheduling/events.jsonl',r)
                assert ok,'Training failed; preserve logs: '+key
                locks.pop(r['gpu_uuid']).close();del active[key]
            snapshot=serial.snapshot();dd=serial.devices(snapshot)
            if pending and len(active)<2:
                occupied={i['record']['gpu_uuid'] for i in active.values()}
                candidates=sorted(dd,key=lambda x:(x['uuid']!=PREFERRED,x['index']))
                for dev in candidates:
                    uuid=dev['uuid']
                    if uuid in occupied:continue
                    stable[uuid]=stable.get(uuid,0)+1 if dev['idle'] else 0
                    if stable[uuid]<2:continue
                    lock=lock_gpu(uuid)
                    if lock is None:continue
                    fresh=next(x for x in serial.devices(serial.snapshot()) if x['uuid']==uuid)
                    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
                    if not fresh['idle'] or memory['MemAvailable']<16*1024**3 or serial.shutil.disk_usage(BASE).free<20*1024**3:
                        lock.close();stable[uuid]=0;continue
                    target=pending.pop(0);key=target.name
                    command=args('worker.py','--target',target);logpath=BASE/'scheduling'/f'{key}.log'
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=uuid)
                    verify_hashes(d['sources'])
                    with logpath.open('x') as log:proc=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,env=env)
                    record=dict(key=key,state='running',started_at=now(),pid=proc.pid,gpu_uuid=uuid,command=command,log=str(logpath),scheduling_amendment=str(AMENDMENT))
                    write(BASE/'scheduling'/f'{key}.json',record);append(BASE/'scheduling/events.jsonl',record)
                    locks[uuid]=lock;active[key]=dict(target=target,record=record,proc=proc);stable[uuid]=0
                    break
            write(BASE/'status.json',dict(state='parallel_training',at=now(),pid=os.getpid(),max_concurrent_training=2,
                running={k:v['record'] for k,v in active.items()},pending=[p.name for p in pending],amendment=str(AMENDMENT)))
            if active or pending:time.sleep(15)
    finally:
        for lock in locks.values():lock.close()

def main():
    controller_lock=(BASE/'controller.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    amendment=verified(AMENDMENT);verify_hashes(amendment['sources'])
    verify_hashes(verified(BASE/'design.json')['sources']);dev=None;lock=None
    try:
        train_parallel()
        while lock is None:
            dev,lock=serial.reserve(serial.snapshot())
            if lock is None:time.sleep(15)
        for key,command in [('development_predict',args('evaluate.py','predict')),('development_score',args('evaluate.py','score'))]:
            dev,lock=serial.wait_idle(dev,lock);serial.run_task(key,command,dev)
        lock.close();lock=None
        serial.run_task('statistics_and_verification',args('analyze.py'))
        write(BASE/'status.json',dict(state='long_training_complete',at=now(),pid=os.getpid(),report=str(REPORT),amendment=str(AMENDMENT)))
        from long_report import main as report
        report()
        with REPORT.open('a') as f:f.write('\n调度修订：用户明确授权使用空闲GPU3后，本实验改为最多两组训练并行；相同的300轮预算、模型和数据协议保持不变。以 parallel_scheduling_amendment.json 为准。\n')
        evidence=[BASE/'design.json',AMENDMENT,BASE/'migration_verification.json',BASE/'gpu_checks.json',BASE/'independent_verification.json',BASE/'analysis/summary.json',REPORT]
        freeze(BASE/'acceptance.json',dict(at=now(),completed_long_training=True,full300=True,evidence={str(p):sha(p) for p in evidence}))
    except BaseException:
        write(BASE/'status.json',dict(state='failed',at=now(),pid=os.getpid(),traceback=traceback.format_exc(),amendment=str(AMENDMENT)))
        raise
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':main()
