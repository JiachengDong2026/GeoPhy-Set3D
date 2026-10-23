"""GPU UUID scheduler: never share a device with an existing compute process."""
from common import *
import argparse
import fcntl
import time


def run_jobs(jobs,phase,*,lock_path=None,preflight=None):
    log=OUT/'scheduling'/phase;log.mkdir(parents=True,exist_ok=True)
    lock=Path(lock_path or OUT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    pending=list(jobs);active=[];failed=[];last_report=0.
    while pending or active:
        if preflight is not None:preflight()
        snap=resources();append(log/'resources.jsonl',snap)
        if time.monotonic()-last_report>=60:
            try:
                import report
                report.progress()
            except Exception as error:append(log/'report_warnings.jsonl',dict(at=now(),error=repr(error)))
            last_report=time.monotonic()
        for job,proc,handle,started,gpu,record in list(active):
            code=proc.poll()
            if code is None:continue
            handle.close();record.update(ended_at=now(),seconds=time.monotonic()-started,exit_code=code)
            write(log/(job['key']+'.json'),record);active.remove((job,proc,handle,started,gpu,record))
            print(json.dumps(dict(event='finished',key=job['key'],code=code,at=now())),flush=True)
            if code:failed.append(job['key'])
        # Stop launching after failure, but retain and wait for already authorized siblings.
        if failed:pending=[]
        occupied={x.split(',')[0].strip() for x in snap['gpu_processes'].splitlines() if x.strip()}
        owned={x[4] for x in active}
        for line in snap['gpus'].splitlines():
            cols=[s.strip() for s in line.split(',')];gpu=cols[1];free=int(cols[5]);util=int(cols[6])
            if not pending:break
            if gpu in occupied or gpu in owned or free<20000 or util>5:continue
            # New resource snapshot immediately before each start, including process ownership.
            fresh=resources()
            if gpu in {x.split(',')[0].strip() for x in fresh['gpu_processes'].splitlines()}:continue
            job=pending.pop(0);attempt=1
            while (log/f"{job['key']}_attempt{attempt}.log").exists():attempt+=1
            path=log/f"{job['key']}_attempt{attempt}.log";handle=path.open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen([sys.executable,*job['args']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            rec=dict(key=job['key'],pid=proc.pid,gpu_uuid=gpu,started_at=now(),args=job['args'],log=str(path),preflight=fresh)
            write(log/(job['key']+'.json'),rec);active.append((job,proc,handle,time.monotonic(),gpu,rec));owned.add(gpu)
            print(json.dumps(dict(event='started',key=job['key'],gpu=gpu,pid=proc.pid,at=now())),flush=True)
        if pending or active:time.sleep(15)
    fcntl.flock(lock,fcntl.LOCK_UN)
    if failed:raise RuntimeError('Failed jobs; see logs: '+', '.join(failed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('queue');p.add_argument('--phase',required=True);a=p.parse_args();run_jobs(read(a.queue),a.phase)
