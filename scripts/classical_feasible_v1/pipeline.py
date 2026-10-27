"""Durable serial U-Net workflow; test only after successful formal completion."""
from feasible_context import *
import subprocess


def run(name,args):
    path=ROOT/(name+'.log')
    with path.open('x') as stream:
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        write(ART/'pipeline_status.json',dict(state=name,pid=process.pid,log=str(path)))
        code=process.wait()
    assert code==0,(name,code,str(path))


def main():
    verify();deadline=time.monotonic()+3600
    while not (ART/'teachers.json').exists():
        task=ROOT/'scheduling/teachers.json'
        if task.exists():assert read(task)['state'] not in ['failed','interrupted'],'Teacher job failed; preserve evidence'
        assert time.monotonic()<deadline,'Bounded teacher dependency wait expired'
        time.sleep(15)
    while read(ROOT/'scheduling/teachers.json')['state']=='running':time.sleep(5)
    assert read(ROOT/'scheduling/teachers.json')['state']=='complete'
    run('unet_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'unet','train_unet.py'])
    summary=read(ART/'unet_summary.json')
    if not summary['formal_complete']:
        write(ART/'pipeline_status.json',dict(state='stopped_for_analysis',summary=summary));return
    run('freeze_unet_test',[str(PROJECT/'scripts/classical_feasible_v1/test_unet.py'),'freeze'])
    run('test_unet_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'unet_test','test_unet.py','predict'])
    run('score_unet_test',[str(PROJECT/'scripts/classical_feasible_v1/test_unet.py'),'score'])
    write(ART/'pipeline_status.json',dict(state='formal_training_and_test_complete',summary=summary,test=read(ART/'unet_test_score.json')))


if __name__=='__main__':main()
