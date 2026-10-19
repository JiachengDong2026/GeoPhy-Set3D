"""Durable ordered paper workflow, with strict development-before-test gates."""
from common import *
import fcntl
import time
from scheduler import run_jobs


def state(name,**kw):
    write(OUT/'pipeline_status.json',dict(state=name,at=now(),pid=os.getpid(),**kw))
    append(OUT/'pipeline_events.jsonl',read(OUT/'pipeline_status.json'))
    import report;report.progress()
    print(json.dumps(read(OUT/'pipeline_status.json')),flush=True)

def stage(key,args):
    marker=OUT/'stages'/(key+'.json')
    if marker.exists():assert read(marker)['exit_code']==0 and read(marker)['args']==args;return
    state(key)
    log=OUT/'stages'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    assert not log.exists(),'Inspect failed stage and use explicit recovery; do not overwrite logs'
    with log.open('x') as f:p=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    write(marker,dict(at=now(),args=args,exit_code=p.returncode))
    assert p.returncode==0,f'{key} failed; see {log}'

def jobs(key,jobs):
    marker=OUT/'stages'/(key+'.json')
    if marker.exists():assert read(marker)['exit_code']==0;return
    state(key);run_jobs(jobs,key);write(marker,dict(at=now(),exit_code=0,jobs=jobs))

def main():
    mutex=(OUT/'pipeline.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state('waiting_classical_development_predictions')
    for _ in range(3600):
        paths=[OUT/'classical/val'/m/'frozen/complete.json' for m in ('NNLS-NL','Tikhonov','TV')]
        if all(p.exists() for p in paths):break
        for p in (OUT/'scheduling/classical_val').glob('*.json'):
            if read(p).get('exit_code',0)!=0:raise RuntimeError('Classical dev failed; inspect scheduling logs')
        if _%4==0:
            import report;report.progress()
        time.sleep(15)
    else:raise RuntimeError('No classical completion within 15h; preserve state for review')
    methodsfile=str(OUT/'classical_val_methods.json')
    stage('classical_dev_seal',['scripts/paper_final_v1/evaluation.py','seal','--split','val','--scope','classical_dev','--methods',methodsfile])
    jobs('classical_dev_score',[dict(key='score',args=['scripts/paper_final_v1/evaluation.py','score','--split','val','--scope','classical_dev','--methods',methodsfile])])
    stage('classical_dev_summary',['scripts/paper_final_v1/evaluation.py','summarize','--split','val','--scope','classical_dev','--methods',methodsfile])
    stage('classical_dev_verify',['scripts/paper_final_v1/verify.py','--split','val','--scope','classical_dev'])
    stage('classical_dev_report',['scripts/paper_final_v1/report.py','export','--base',str(OUT/'evaluation/classical_dev')])
    # Classical comparison is completed before ablation registration/training.
    from train import register
    # Pair the structural questions first and use the shorter no-feedback jobs
    # to balance the three devices before launching consistency controls.
    arms=[('multiscale',20261019),('single_scale',20261011),('no_feedback',20261011),
          ('no_feedback',20261019),('single_scale',20261019),('no_consistency',20261011),('no_consistency',20261019)]
    for arm,seed in arms:register(arm,seed)
    queue=[dict(key=f'{arm}_seed{seed}',args=['scripts/paper_final_v1/train.py','train','--target',str(OUT/'training'/f'{arm}_seed{seed}')]) for arm,seed in arms]
    jobs('formal_ablation_training',queue)
    from evaluation import training_methods
    neural=training_methods();devmethods=neural[2:]+[dict(key=m,kind='classical') for m in ('NNLS-NL','Tikhonov','TV')]
    # The original live controller may have cached this function before the
    # user requested SART; evaluation.seal independently enforces the same gate.
    if (OUT/'sart/registration.json').exists():
        from sart import wait_development
        devmethods.append(wait_development())
    devfile=OUT/'development_methods.json'
    if not devfile.exists():freeze(devfile,devmethods)
    jobs('ablation_development_predictions',[dict(key=m['key'],args=['scripts/paper_final_v1/evaluation.py','predict','--split','val','--key',m['key'],'--methods',str(devfile)]) for m in neural[2:]])
    stage('all_development_seal',['scripts/paper_final_v1/evaluation.py','seal','--split','val','--methods',str(devfile)])
    jobs('all_development_score',[dict(key='score',args=['scripts/paper_final_v1/evaluation.py','score','--split','val','--methods',str(devfile)])])
    stage('all_development_summary',['scripts/paper_final_v1/evaluation.py','summarize','--split','val','--methods',str(devfile)])
    stage('ablation_development_statistics',['scripts/paper_final_v1/ablation_statistics.py','--split','val'])
    stage('all_development_verify',['scripts/paper_final_v1/verify.py','--split','val'])
    stage('all_development_report',['scripts/paper_final_v1/report.py','export','--base',str(OUT/'evaluation/val')])
    stage('final_method_lock',['scripts/paper_final_v1/finalize.py','lock'])
    methods=verified(OUT/'final_lock.json')['methods'];testqueue=[]
    for m in methods:
        args=['scripts/paper_final_v1/classical.py','work','--split','test','--method',m['key']] if m['kind']=='classical' else ['scripts/paper_final_v1/evaluation.py','predict','--split','test','--key',m['key']]
        testqueue.append(dict(key=m['key'],args=args))
    jobs('final_all_predictions',testqueue)
    stage('final_all_seal',['scripts/paper_final_v1/evaluation.py','seal','--split','test'])
    jobs('final_independent_scoring',[dict(key='score',args=['scripts/paper_final_v1/evaluation.py','score','--split','test'])])
    stage('final_statistics',['scripts/paper_final_v1/evaluation.py','summarize','--split','test'])
    stage('ablation_final_statistics',['scripts/paper_final_v1/ablation_statistics.py','--split','test'])
    stage('final_independent_verification',['scripts/paper_final_v1/verify.py','--split','test'])
    stage('final_figures_csv',['scripts/paper_final_v1/report.py','export','--base',str(OUT/'evaluation/test')])
    stage('final_acceptance',['scripts/paper_final_v1/finalize.py','accept'])
    state('complete',acceptance=str(OUT/'final_acceptance.json'))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('failed',error=repr(e));raise
