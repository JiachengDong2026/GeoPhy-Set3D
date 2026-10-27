"""Stop only our FNO immediately after a complete epoch, then resume exact state."""
from feasible_context import *
import ctypes,struct,signal,subprocess,select


def main():
    verify();assert read(ART/'fno_workers_checks.json')['passed']
    for path,digest in read(ART/'fno_workers_registration.json')['sources'].items():
        assert sha(PROJECT/path)==digest,path
    record=read(ROOT/'scheduling/pilot_fno.json');pid=record['pid']
    assert record['state']=='running'
    assert str(PROJECT/'scripts/classical_feasible_v1/train_multicase_debug.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    assert fd>=0
    assert libc.inotify_add_watch(fd,os.fsencode(ROOT/'pilot_fno'),8)>=0  # IN_CLOSE_WRITE
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        ready,_,_=select.select([fd],[],[],5)
        if not ready:continue
        data=os.read(fd,65536);offset=0;matched=False
        while offset<len(data):
            _,mask,_,length=struct.unpack_from('iIII',data,offset)
            name=data[offset+16:offset+16+length].split(b'\0')[0]
            if mask&8 and name==b'metrics.jsonl':matched=True
            offset+=16+length
        if not matched:continue
        os.kill(pid,signal.SIGSTOP)
        rows=[json.loads(line) for line in (ROOT/'pilot_fno/metrics.jsonl').read_text().splitlines()]
        latest=rows[-1]
        if latest['stage']!='preheat' or latest['epoch']>=60:
            os.kill(pid,signal.SIGCONT);raise RuntimeError('Expected preheat boundary before epoch60')
        write(ROOT/'pilot_fno/workers_stop_request.json',dict(pid=pid,epoch=latest['epoch'],updates=latest['updates'],
            reason='Exact original CPU backprojection in four loader workers; identical RNG/order; no scientific change'))
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        break
    else:raise TimeoutError('No epoch boundary within300s; original worker untouched')
    os.close(fd)
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        if (ROOT/'pilot_fno/interrupted.pt').exists() and read(ROOT/'scheduling/pilot_fno.json')['state']!='running':break
        time.sleep(2)
    else:raise TimeoutError('Inspect original worker/controller; no resume attempted')
    checkpoint=ROOT/'pilot_fno/interrupted.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['stage']=='preheat' and ck['epoch']==latest['epoch']
    assert ck['optimizer_steps']==latest['updates']==83*latest['epoch']
    assert ck['scheduler']['last_epoch']==latest['epoch']
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={latest['updates']}
    journal=[json.loads(line) for line in (ROOT/'pilot_fno/preheat_updates.jsonl').read_text().splitlines()]
    assert len(journal)==latest['updates'] and [r['update'] for r in journal]==list(range(1,latest['updates']+1))
    boundary=dict(epoch=ck['epoch'],updates=ck['optimizer_steps'],checkpoint_sha256=sha(checkpoint),
        no_partial_updates=True,no_update_replay=True,original_scheduler_preserved=True,
        original_optimizer_preserved=True,original_rng_preserved=True)
    write(ROOT/'pilot_fno/workers_boundary.json',boundary);write(ART/'fno_workers_boundary.json',boundary)
    call('fno_workers_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'pilot_fno_workers','train_multicase_workers.py','fno'])
    assert (ART/'fno_pilot_summary.json').exists()
    for arm in ['senseiver','gnot','deepsets']:
        call(arm+'_pilot_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'pilot_'+arm,'train_multicase_debug.py',arm])
        assert (ART/(arm+'_pilot_summary.json')).exists()
    write(ART/'other_algorithms_status.json',dict(state='bounded_multicase_checks_complete',fno_execution='worker-prepared exact continuation'))


def call(name,arguments):
    with (ROOT/(name+'.log')).open('x') as stream:
        child=subprocess.Popen([sys.executable,*arguments],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        write(ART/'other_algorithms_status.json',dict(state=name,pid=child.pid,log=str(ROOT/(name+'.log'))))
        code=child.wait()
    assert code==0,(name,code,'Inspect preserved log')


if __name__=='__main__':main()
