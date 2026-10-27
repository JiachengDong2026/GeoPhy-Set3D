"""Independent qualification and sealed test for the explicitly revised guard variant."""
from sparse_guard_v2_context import *
import subprocess,importlib


def action(name):
    verify()
    if name=='validation':
        module=importlib.import_module('review_validation');module.ART=ART;module.ROOT=ROOT;module.verify=verify;module.main()
    elif name=='interventions':
        from review_selected_sparse_v2 import main as selected;selected()
    elif name in ['freeze','predict','score']:
        import test_sparse_guard_v2 as test
        if name=='freeze':
            assert read(ART/'selected_unet_causal_review.json')['passed'];test.registration()
        else:getattr(test,name)()
    elif name=='compare':
        module=importlib.import_module('compare_frozen_main');module.ART=ART;module.ROOT=ROOT;module.verify=verify;module.main()
    else:raise ValueError(name)
    verify()


def call(name,args):
    with (ROOT/(name+'.log')).open('x') as log:
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        write(ART/'pipeline_status.json',dict(state=name,pid=process.pid,log=str(ROOT/(name+'.log'))))
        code=process.wait()
    assert code==0,(name,code,'Preserve failed phase; no automatic test continuation')


def pipeline():
    verify();deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        state=read(BASE_ROOT/'scheduling/unet_sparse_guard_v2.json')['state']
        assert state not in ['failed','interrupted']
        if state=='complete':break
        time.sleep(10)
    else:raise TimeoutError('Sparse guard training wait limit')
    summary=read(ART/'unet_summary.json')
    if not summary['formal_complete']:
        write(ART/'pipeline_status.json',dict(state='stopped_for_analysis',summary=summary));return
    script=PROJECT/'scripts/classical_feasible_v1'
    for name in ['validation','interventions','freeze']:
        call(name,[str(script/'sparse_guard_v2_pipeline.py'),name])
    call('predict_controller',[str(script/'launch.py'),'unet_sparse_guard_v2_test','sparse_guard_v2_pipeline.py','predict'])
    for name in ['score','compare']:call(name,[str(script/'sparse_guard_v2_pipeline.py'),name])
    write(ART/'pipeline_status.json',dict(state='formal_training_and_test_complete',guard_version='sparse_guard_v2',
        original_v1_status_preserved=True,paired_main_comparison=True,test=read(ART/'unet_test_score.json')))
    verify()


if __name__=='__main__':
    if len(sys.argv)>1:action(sys.argv[1])
    else:pipeline()
