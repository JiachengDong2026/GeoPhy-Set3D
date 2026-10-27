"""Only a train-only positive paired scale diagnosis can promote the fresh formal variant."""
from feasible_context import *
import subprocess


def call(name,args):
    with (ROOT/(name+'.log')).open('x') as stream:
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        write(ART/'amplitude_pipeline_status.json',dict(state=name,pid=process.pid))
        code=process.wait()
    assert code==0,(name,code,'inspect preserved log')


def main():
    verify()
    pair=read(ART/'amplitude_pair_summary.json')['results']
    assert pair[1]['final']['mean_normalized_MSE']<=.5*pair[0]['final']['mean_normalized_MSE']
    assert pair[1]['final']['mean_H']<=.9*pair[0]['final']['mean_H']
    call('unet_amp_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'unet_amp_v2','train_unet_amp_v2.py'])
    summary=read(ART/'unet_amp_summary.json')
    if not summary['formal_complete']:
        write(ART/'amplitude_pipeline_status.json',dict(state='stopped_for_analysis',summary=summary));return
    call('freeze_unet_amp_test',[str(PROJECT/'scripts/classical_feasible_v1/test_unet_amp_v2.py'),'freeze'])
    call('test_unet_amp_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'unet_amp_test','test_unet_amp_v2.py','predict'])
    call('score_unet_amp_test',[str(PROJECT/'scripts/classical_feasible_v1/test_unet_amp_v2.py'),'score'])
    write(ART/'amplitude_pipeline_status.json',dict(state='formal_training_and_test_complete',summary=summary))


if __name__=='__main__':main()
