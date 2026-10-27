"""Continue remaining untouched pilots after the existing FNO worker finishes."""
from feasible_context import *
import subprocess


def main():
    verify()
    for path,digest in read(ART/'gnot_chunk_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        state=read(ROOT/'scheduling/pilot_fno_workers.json')['state']
        assert state not in ['failed','interrupted'],'FNO failed; inspect before another pilot'
        if state=='complete':break
        time.sleep(10)
    else:raise TimeoutError('FNO dependency wait limit')
    assert (ART/'fno_pilot_summary.json').exists()
    for arm in ['senseiver','gnot','deepsets']:
        script='train_gnot_chunked.py' if arm=='gnot' else 'train_multicase_debug.py'
        path=ROOT/(arm+'_pilot_controller.log')
        with path.open('x') as stream:
            process=subprocess.Popen([sys.executable,str(PROJECT/'scripts/classical_feasible_v1/launch.py'),
                'pilot_'+arm,script,arm],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
            write(ART/'other_algorithms_status.json',dict(state=arm+'_pilot',pid=process.pid,log=str(path)))
            code=process.wait()
        assert code==0,(arm,'Inspect preserved pilot/controller log')
        assert (ART/(arm+'_pilot_summary.json')).exists()
    write(ART/'other_algorithms_status.json',dict(state='bounded_multicase_checks_complete'))
    verify()


if __name__=='__main__':main()
