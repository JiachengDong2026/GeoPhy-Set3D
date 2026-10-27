"""Finish the unchanged training, qualify its selected best, seal/score, compare."""
from feasible_context import *
import subprocess


def call(name,arguments):
    with (ROOT/(name+'.log')).open('x') as stream:
        process=subprocess.Popen([sys.executable,*arguments],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        write(ART/'pipeline_status.json',dict(state=name,pid=process.pid,log=str(ROOT/(name+'.log'))))
        code=process.wait()
    assert code==0,(name,code,'Inspect preserved log; no test continuation after a failed check')


def main():
    verify()
    for registration in ['test_registration.json','selected_checkpoint_review_registration.json','frozen_main_comparison_registration.json']:
        for path,digest in read(ART/registration)['sources'].items():assert sha(PROJECT/path)==digest,path
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        task=read(ROOT/'scheduling/unet_workers.json')
        assert task['state'] not in ['failed','interrupted'],'Training engine failed; preserve state for diagnosis'
        if task['state']=='complete':break
        time.sleep(10)
    else:raise TimeoutError('Formal training dependency wait limit')
    summary=read(ART/'unet_summary.json')
    if not summary['formal_complete']:
        write(ART/'pipeline_status.json',dict(state='stopped_for_analysis',summary=summary));return
    script=PROJECT/'scripts/classical_feasible_v1'
    call('independent_final_unet_validation',[str(script/'review_validation.py')])
    call('independent_selected_unet_interventions',[str(script/'review_selected_unet.py')])
    assert read(ART/'selected_unet_causal_review.json')['passed']
    call('freeze_unet_test',[str(script/'test_unet.py'),'freeze'])
    call('test_unet_controller',[str(script/'launch.py'),'unet_test','test_unet.py','predict'])
    call('score_unet_test',[str(script/'test_unet.py'),'score'])
    call('compare_unet_frozen_main',[str(script/'compare_frozen_main.py')])
    write(ART/'pipeline_status.json',dict(state='formal_training_and_test_complete',summary=summary,test=read(ART/'unet_test_score.json'),paired_main_comparison=True))
    verify()


if __name__=='__main__':main()
