"""Durable two-batch queue; exclusive GPUs, inherited locks and no silent retries."""
from common_stage import *
import signal


def workflow_alive(record):
    try:
        stat=Path(f"/proc/{record['workflow_pid']}/stat").read_text().split(') ',1)[1].split()
        return stat[0]!='Z' and stat[19]==record['workflow_start_ticks']
    except FileNotFoundError:return False


def configure_scheduler():
    s=scheduler();original=s.devices;registration=verified(BASE/'resource_registration.json')
    def devices(snap):
        pinned={r['gpu_uuid'] for r in registration['foreign_workflows'] if workflow_alive(r)}
        return [dict(d,idle=False,foreign_workflow_pending=True) if d['uuid'] in pinned else d for d in original(snap)]
    s.devices=devices
    return s


def verify_or_fail_previous(s,key,artifact):
    path=s.task_path(key)
    if not path.exists():return
    old=read(path)
    assert not (old['state']=='running' and s.alive(old['pid'])),'Existing live worker: '+str(path)
    if artifact.exists():assert old['state']=='complete';return
    recovery=BASE/'recovery'/f'{key}.json'
    assert recovery.exists(),'Preserved failed/interrupted task requires inspected recovery registration: '+str(path)
    r=verified(recovery);assert r['allow_resume'] and r['original_record_sha256']==sha(path)
    if r.get('checkpoint'):
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==sha(Path(r['checkpoint']).parent.parent/'protocol.json')
    append(BASE/'scheduling/events.jsonl',dict(at=now(),event='audited_recovery_authorized',key=key,recovery_sha256=sha(recovery)))


def own_descendants(pid):
    table=subprocess.check_output(['ps','-eo','pid,ppid,uid'],text=True)
    children={}
    for line in table.splitlines()[1:]:
        p,pp,u=map(int,line.split())
        if u==os.getuid():children.setdefault(pp,[]).append(p)
    found=set();todo=[pid]
    while todo:
        p=todo.pop()
        for child in children.get(p,[]):
            if child not in found:found.add(child);todo.append(child)
    return found


def conflicts(snap,item):
    uuid=item['record']['gpu_uuid'];allowed={item['record']['pid']}|own_descendants(item['record']['pid'])
    return [line for line in snap['gpu_processes'].splitlines() if line.strip() and line.split(',')[0].strip()==uuid and int(line.split(',')[1]) not in allowed]


def batch(s,tasks,index):
    pending=[];active=[];stable={};failed=False
    for a,seed in tasks:
        key=f'{a}_seed{seed}';target=BASE/'training'/key;artifact=target/'summary.json'
        if artifact.exists():validate_training(target);continue
        verify_or_fail_previous(s,key,artifact);pending.append(dict(key=key,target=target))
    while pending or active:
        snap=s.snapshot()
        for item,job in list(active):
            collision=conflicts(snap,item) if not s.done(item) else []
            if collision and not item.get('collision_recorded'):
                item['collision_recorded']=True;failed=True
                children=own_descendants(item['record']['pid'])
                incident=BASE/'recovery'/f"collision_{job['key']}_attempt{item['record']['attempt']}.json"
                freeze(incident,dict(at=now(),task=item['record'],foreign_GPU_processes=collision,snapshot=snap,
                    action='SIGINT own worker and identified own data-loader children only; stop new launches; preserve last.pt and logs',children=sorted(children),foreign_signals=0))
                for pid in [item['record']['pid'],*children]:
                    try:os.kill(pid,signal.SIGINT)
                    except ProcessLookupError:pass
            if not s.done(item):continue
            valid=item['process'].returncode==0 and (job['target']/'summary.json').exists() and not item.get('collision_recorded')
            if valid:
                try:validate_training(job['target'])
                except BaseException:
                    valid=False;append(BASE/'scheduling/events.jsonl',dict(at=now(),event='validation_failed',key=job['key'],traceback=traceback.format_exc()))
            try:s.finish(item,valid=valid)
            except AssertionError:failed=True
            active.remove((item,job))
        if not failed:
            occupied={i['record']['gpu_uuid'] for i,_ in active}
            for d in s.devices(snap):
                stable[d['uuid']]=stable.get(d['uuid'],0)+1 if d['idle'] and d['uuid'] not in occupied else 0
                if not pending or stable[d['uuid']]<2 or len(active)>=4:continue
                if len(active)>=2 and not s.host_ok(expand=True):continue
                lock=s.reserve(d['uuid'])
                if lock is None:continue
                job=pending.pop(0);verify_hashes(verified(BASE/'protocol.json')['sources'])
                item=s.launch(job['key'],command('train_worker.py','--target',job['target']),d['uuid'],lock)
                active.append((item,job));stable[d['uuid']]=0
        write(BASE/'status.json',dict(at=now(),state='failed_waiting_for_siblings' if failed else 'training',batch=index,
            active=[i['record'] for i,_ in active],pending=[j['key'] for j in pending],controller_pid=os.getpid()))
        update_tracker()
        if failed and not active:raise RuntimeError('Task failed; preserved evidence and blocked next batch. Pending: '+str([j['key'] for j in pending]))
        if pending or active:time.sleep(15)
    path=BASE/f'batch{index}_accepted.json'
    if not path.exists():freeze(path,dict(at=now(),completed=True,runs=[validate_training(BASE/'training'/f'{a}_seed{seed}') for a,seed in tasks]))


def run_once(s,key,cmd,artifact):
    if artifact.exists():return
    verify_or_fail_previous(s,key,artifact)
    s.run_gpu_once(key,cmd,artifact)


def main():
    BASE.joinpath('scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_hashes(verified(BASE/'protocol.json')['sources']);assert verified(BASE/'checks/cpu.json')['passed']
            s=configure_scheduler()
            run_once(s,'gpu_engineering',command('checks.py','gpu'),BASE/'checks/gpu.json')
            run_once(s,'resume_engineering',command('resume_check.py'),BASE/'checks/resume.json')
            batch(s,TASKS[:4],1);batch(s,TASKS[4:],2)
            if not (BASE/'training_complete.json').exists():freeze(BASE/'training_complete.json',dict(at=now(),completed=True,runs=methods()))
            subprocess.run(command('evaluate.py','register'),cwd=PROJECT,check=True)
            for a,seed in TASKS:
                key=f'{a}_seed{seed}'
                run_once(s,'predict_'+key,command('predict.py','trained','--key',key),BASE/'evaluation'/key/'complete.json')
            run_once(s,'benchmark',command('predict.py','benchmark'),BASE/'benchmark.json')
            with (BASE/'scheduling/scoring.log').open('a') as log:subprocess.run(command('evaluate.py','score'),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert verified(BASE/'acceptance.json')['completed']
            write(BASE/'status.json',dict(at=now(),state='complete',report=str(REPORT),controller_pid=os.getpid()))
            update_tracker()
        except BaseException:
            write(BASE/'status.json',dict(at=now(),state='failed',traceback=traceback.format_exc(),controller_pid=os.getpid()));raise


if __name__=='__main__':main()
