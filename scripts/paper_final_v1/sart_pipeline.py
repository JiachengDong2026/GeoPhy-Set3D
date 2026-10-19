"""Detached CPU SART development; main workflow gates on immutable completion."""
from common import *
import fcntl
import time
from sart import BASE


def status(name,**kwargs):
    value=dict(state=name,at=now(),pid=os.getpid(),**kwargs);write(BASE/'pipeline_status.json',value);append(BASE/'pipeline_events.jsonl',value);print(json.dumps(value),flush=True)


def stage(key,args):
    marker=BASE/'stages'/(key+'.json')
    if marker.exists():assert read(marker)['exit_code']==0;return
    status(key);log=BASE/'stages'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True);assert not log.exists()
    started=time.monotonic();preflight=resources()
    with log.open('x') as f:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        p=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record=dict(key=key,pid=p.pid,gpu_uuid=None,device='cpu',started_at=now(),args=args,log=str(log),preflight=preflight)
        task=OUT/'scheduling/sart_development'/(key+'.json');write(task,record)
        last=0
        while p.poll() is None:
            if time.monotonic()-last>60:append(BASE/'resources.jsonl',resources());last=time.monotonic()
            time.sleep(5)
    record.update(ended_at=now(),seconds=time.monotonic()-started,exit_code=p.returncode);write(task,record);write(marker,record)
    assert p.returncode==0,f'{key} failed; see {log}'


def finish():
    import evaluation as ev
    from sart import method
    methods=[method()];ev.SCOPE='sart_dev';ev.seal('val',methods);ev.score('val',device='cpu');ev.summarize('val',ev.historical_val_metrics())
    from verify import main as verify
    verify('val','sart_dev')
    from report import export
    export(OUT/'evaluation/sart_dev')
    from sart_report import main as report_sart
    report_sart()


def main():
    mutex=(BASE/'pipeline.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert verified(BASE/'numerical_checks.json')['passed']
    for candidate in range(2):stage(f'calibration_{candidate}',['scripts/paper_final_v1/sart.py','work','--split','calibration','--candidate',str(candidate)])
    stage('select',['scripts/paper_final_v1/sart.py','select'])
    stage('validation_predictions',['scripts/paper_final_v1/sart.py','work','--split','val'])
    stage('validation_score_verify_report',['scripts/paper_final_v1/sart_pipeline.py','finish'])
    evidence={str(OUT/p):sha(OUT/p) for p in ('sart/registration.json','sart/selected.json','sart/numerical_checks.json','evaluation/sart_dev/all_predictions_sealed.json','evaluation/sart_dev/summary.json','evaluation/sart_dev/independent_verification.json')}
    freeze(BASE/'development_complete.json',dict(at=now(),passed=True,evidence=evidence,calibration_reconstructions=24,validation_reconstructions=612,final_GT_read=False))
    status('complete')


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='finish':finish()
        else:main()
    except BaseException as e:status('failed',error=repr(e));raise
