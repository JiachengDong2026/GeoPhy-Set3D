"""UUID-locked, restartable queue; engineering prefix gates before OARI promotion."""
from runtime import *
from scope import effective_runs, scope_evidence, deferred_runs, evaluation_stage
import fcntl
import shutil
import time
import traceback

def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def snapshot():
    result=resources()
    append(BASE/'scheduling/resources.jsonl',result)
    return result

def devices(s):
    active={r.split(',')[0].strip() for r in s['gpu_processes'].splitlines() if r.strip()}
    out=[]
    for line in s['gpus'].splitlines():
        i,u,name,total,used,free,util,mem=[v.strip() for v in line.split(',')]
        out.append(dict(index=int(i),uuid=u,used=int(used),free=int(free),util=int(util),
            idle=u not in active and int(used)<100 and int(free)>=20000 and int(util)<=5))
    return out

def host_ok(expand=False):
    memory={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    return memory['MemAvailable']>=(64 if expand else 16)*1024**3 and shutil.disk_usage(BASE).free>=(150 if expand else 30)*1024**3 and os.getloadavg()[0]<os.cpu_count()*.75

def reserve(uuid):
    GPU_LOCKS.mkdir(exist_ok=True)
    f=(GPU_LOCKS/(uuid+'.lock')).open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    if not next(d for d in devices(snapshot()) if d['uuid']==uuid)['idle'] or not host_ok():
        f.close();return None
    return f

def wait_gpu():
    stable={}
    while True:
        for d in devices(snapshot()):
            stable[d['uuid']]=stable.get(d['uuid'],0)+1 if d['idle'] else 0
            if stable[d['uuid']]>=2:
                f=reserve(d['uuid'])
                if f is not None:return d['uuid'],f
        write(BASE/'status.json',dict(at=now(),state='waiting_for_idle_gpu',controller_pid=os.getpid()))
        time.sleep(15)

def task_path(key):return BASE/'scheduling'/f'{key}.json'

def progress():
    import status_report
    status_report.main(quiet=True)

def launch(key,command,uuid,lock):
    path=task_path(key);old=read(path) if path.exists() else None
    attempt=1 if old is None else old['attempt']+1
    log=BASE/'scheduling'/f'{key}_attempt{attempt}.log'
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=uuid)
    with log.open('x') as f:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT,
            pass_fds=(lock.fileno(),) if lock is not None else ())
    record=dict(key=key,state='running',attempt=attempt,command=command,started_at=now(),
        gpu_uuid=uuid,pid=process.pid,log=str(log),controller_pid=os.getpid())
    write(path,record);append(BASE/'scheduling/events.jsonl',record)
    return dict(record=record,process=process,lock=lock)

def done(item):
    process=item['process']
    return process.poll() is not None if process is not None else not alive(item['record']['pid'])

def finish(item,valid=None):
    r=item['record'];p=item['process']
    if valid is None:valid=p is not None and p.returncode==0
    r.update(state='complete' if valid else 'failed',finished_at=now(),exit_code=p.returncode if p else None)
    write(task_path(r['key']),r);append(BASE/'scheduling/events.jsonl',r)
    if item['lock'] is not None:item['lock'].close()
    assert valid,'Task failed; preserve evidence and block next phase: '+r['key']+' log='+r['log']

def run_gpu_once(key,command,artifact):
    if artifact.exists():return
    uuid,lock=wait_gpu();item=launch(key,command,uuid,lock)
    while not done(item):
        write(BASE/'status.json',dict(at=now(),state=key,task=item['record'],controller_pid=os.getpid()))
        progress()
        snapshot();time.sleep(15)
    finish(item)
    assert artifact.exists(),artifact

def gate(seed):
    path=BASE/'gates'/f'seed{seed}.json'
    if path.exists():return verified(path)['promote']
    a=BASE/'training'/f'uniform_seed{seed}'/'prefix_complete.json'
    b=BASE/'training'/f'uniform_feedback_oari_seed{seed}'/'prefix_complete.json'
    if not a.exists() or not b.exists():return None
    control,treat=read(a),read(b)
    ok=treat['best_H4']<=1.10*control['best_H4'] and treat['seconds']<=2*control['seconds'] and treat['peak_bytes']<=20*1024**3
    freeze(path,dict(at=now(),seed=seed,promote=ok,control=control,treatment=treat,
        rule=verified(BASE/'design.json')['pilot']['gate'],data='validation measurement H4 only; no volume GT',
        evidence={str(a):sha(a),str(b):sha(b)}))
    return ok

def training_jobs():
    d=verified(BASE/'design.json');jobs=[]
    for seed in NEW_SEEDS:
        for arm in ('uniform','uniform_feedback_oari'):
            target=BASE/'training'/f'{arm}_seed{seed}'
            jobs.append(dict(key=target.name+'_prefix20',target=target,stop=20))
    # Main controls continue before lower-priority mechanism ablations.
    for name in effective_runs():
        target=Path(name);p=verified(target/'protocol.json')
        jobs.append(dict(key=target.name+'_full300',target=target,stop=300))
    return jobs

def completed(job):
    target=job['target']
    if job['stop']==20:return (target/'prefix_complete.json').exists()
    if not (target/'summary.json').exists():return False
    s=read(target/'summary.json')
    assert s['completed'] and s['epochs']==300 and s['updates']==24900
    assert sha(target/'checkpoints/best.pt')==s['best_checkpoint_sha256']
    assert sha(target/'checkpoints/last.pt')==s['last_checkpoint_sha256']
    return True

def ready(job):
    p=verified(job['target']/'protocol.json')
    if job['stop']==300 and p['arm'] in ('uniform','uniform_feedback_oari'):
        if not (job['target']/'prefix_complete.json').exists():return False
        if p['arm']=='uniform_feedback_oari':return gate(p['seed']) is True
    return True

def train_all():
    jobs=training_jobs();active={};pending=[];skipped=[];stable={};limit=2
    included=set(effective_runs())
    deferred=[Path(n).name+'_full300' for n in deferred_runs()]
    skipped.extend(Path(n).name+'_full300' for n in verified(BASE/'design.json')['runs'] if n not in included and n not in deferred_runs())
    for job in jobs:
        if completed(job):continue
        p=verified(job['target']/'protocol.json')
        if job['stop']==300 and p['arm']=='uniform_feedback_oari' and gate(p['seed']) is False:
            skipped.append(job['key']);continue
        path=task_path(job['key'])
        if path.exists():
            record=read(path)
            if record['state']=='running' and alive(record['pid']):
                cmd=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                assert str(job['target']) in cmd and 'uniform_final_v1/worker.py' in cmd
                active[job['key']]=dict(record=record,process=None,lock=None,job=job)
                continue
            # Failed work is never silently resumed. An inspected recovery approval
            # file can authorize replay from canonical last.pt without changing protocol.
            if record['state']!='complete':
                approval=BASE/'recovery'/f'{job["key"]}.json'
                assert approval.exists() and verified(approval)['allow_resume'],path
        pending.append(job)
    while pending or active:
        for key,item in list(active.items()):
            if not done(item):continue
            finish(item,valid=(item['process'] is None or item['process'].returncode==0) and completed(item['job']))
            del active[key]
        if limit==2 and len(active)>=2 and host_ok(expand=True):
            states=[read(i['job']['target']/'status.json') if (i['job']['target']/'status.json').exists() else {} for i in active.values()]
            if all(s.get('epoch',0)>=3 for s in states):
                limit=4
                append(BASE/'scheduling/events.jsonl',dict(at=now(),event='expand_parallelism',previous=2,maximum=4,
                    reason='two initial jobs each completed>=3epochs without exception; >=64GiB available RAM, >=150GiB scratch, host load below75%CPU cores',
                    resources=snapshot()))
        for job in list(pending):
            p=verified(job['target']/'protocol.json')
            if job['stop']==300 and p['arm']=='uniform_feedback_oari' and gate(p['seed']) is False:
                skipped.append(job['key']);pending.remove(job)
        snap=snapshot();occupied={i['record']['gpu_uuid'] for i in active.values()}
        for d in devices(snap):
            u=d['uuid']
            stable[u]=stable.get(u,0)+1 if d['idle'] and u not in occupied else 0
            if len(active)>=limit or stable[u]<2:continue
            # A single run must not have its prefix and continuation overlap.
            targets={i['job']['target'] for i in active.values()}
            candidates=[j for j in pending if j['target'] not in targets and ready(j)]
            if not candidates:continue
            lock=reserve(u)
            if lock is None:continue
            job=candidates[0];pending.remove(job)
            verify_hashes(verified(BASE/'design.json')['training_sources'])
            item=launch(job['key'],args('worker.py','--target',job['target'],'--stop-after',job['stop']),u,lock)
            item['job']=job;active[job['key']]=item;occupied.add(u);stable[u]=0
        write(BASE/'status.json',dict(at=now(),state='training',controller_pid=os.getpid(),maximum_parallel=limit,
            running={k:i['record'] for k,i in active.items()},pending=[j['key'] for j in pending],skipped=skipped,deferred=deferred))
        progress()
        if pending or active:time.sleep(15)
    freeze(BASE/'training_complete.json',dict(at=now(),complete=True,skipped=skipped,deferred=deferred,
        completed_runs=[str(Path(n)) for n in effective_runs() if (Path(n)/'summary.json').exists()],
        scope_evidence=scope_evidence(),
        OARI_gates={str(s):gate(s) for s in NEW_SEEDS}))

def main():
    BASE.joinpath('scheduling').mkdir(parents=True,exist_ok=True)
    controller=(BASE/'controller.lock').open('a')
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        verify_hashes(verified(BASE/'design.json')['training_sources'])
        if not (BASE/'checks/cpu.json').exists():
            attempt=len(list((BASE/'scheduling').glob('cpu_checks*.log')))+1
            with (BASE/'scheduling'/f'cpu_checks_attempt{attempt}.log').open('x') as log:
                subprocess.run(args('checks.py','cpu'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        run_gpu_once('gpu_checks',args('checks.py','gpu'),BASE/'checks/gpu.json')
        if not (BASE/'training_complete.json').exists():train_all()
        run_gpu_once('benchmark',args('benchmark.py'),BASE/'benchmark.json')
        # Evaluation code/protocol is independently frozen by evaluate.py only
        # after all currently included training finishes, before current test GT.
        with (BASE/'scheduling/evaluation_freeze.log').open('a') as log:
            subprocess.run(args('evaluate.py','register'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        lock=verified(BASE/'final_lock.json')
        for m in lock['methods']:
            run_gpu_once('predict_'+m['key'],args('evaluate.py','predict','--key',m['key']),
                BASE/'evaluation/test'/m['key']/'complete.json')
        run_gpu_once('seal_score',args('evaluate.py','score'),BASE/'evaluation/test/gt_access.json')
        if not (BASE/'acceptance.json').exists():
            with (BASE/'scheduling/finish.log').open('a') as log:
                subprocess.run(args('finish.py'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        else:assert verified(BASE/'acceptance.json')['completed']
        write(BASE/'status.json',dict(at=now(),state='accepted_pending_sync',controller_pid=os.getpid()))
        progress()
        with (BASE/'scheduling/github_sync.log').open('a') as log:
            subprocess.run(args('sync_results.py'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(BASE/'status.json',dict(at=now(),state='stage_complete_awaiting_user_decision' if evaluation_stage() else 'complete',
            acceptance=str(BASE/'acceptance.json'),controller_pid=os.getpid(),deferred_runs=deferred_runs(),
            OARI_promoted_to_final_model=False))
        progress()
    except BaseException:
        write(BASE/'status.json',dict(at=now(),state='failed',controller_pid=os.getpid(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
