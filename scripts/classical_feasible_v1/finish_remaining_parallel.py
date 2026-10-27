"""Use the released U-Net GPU for remaining pilots; leave the running GNOT untouched."""
from sparse_guard_v2_context import *
import subprocess


def call(name,args,cpu=False):
    with (ROOT/(name+'.log')).open('x') as log:
        env=dict(os.environ)
        if cpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(ART/'remaining_status.json',dict(state=name,pid=process.pid,log=str(ROOT/(name+'.log'))))
        code=process.wait()
    assert code==0,(name,code,'Preserve log and inspect before any continuation')


def main():
    verify()
    for path,digest in read(ART/'remaining_queue_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        path=ART/'pipeline_status.json'
        if path.exists() and read(path)['state'] in ['formal_training_and_test_complete','stopped_for_analysis']:break
        state=read(BASE_ROOT/'scheduling/unet_sparse_guard_v2.json')['state']
        assert state not in ['failed','interrupted'],'U-Net engine failed; preserve resources for diagnosis'
        time.sleep(10)
    else:raise TimeoutError('U-Net test priority wait limit')
    script=PROJECT/'scripts/classical_feasible_v1'
    call('senseiver_v2_controller',[str(script/'launch.py'),'pilot_senseiver_sparse_guard_v2','pilot_sparse_guard_v2.py'])
    call('senseiver_v2_independent_review',[str(script/'review_sparse_guard_pilot.py')],cpu=True)
    write(ART/'senseiver_continuation_status.json',dict(state='complete',summary=read(ART/'senseiver_pilot_summary.json')))
    assert not (BASE_ROOT/'pilot_deepsets').exists(),'Never duplicate an already-started experiment'
    call('deepsets_controller',[str(script/'launch.py'),'pilot_deepsets','train_multicase_debug.py','deepsets'])
    assert (BASE_ART/'deepsets_pilot_summary.json').exists()
    while time.monotonic()<deadline:
        state=read(BASE_ROOT/'scheduling/pilot_gnot.json')['state']
        assert state not in ['failed','interrupted'],'GNOT engine failed; preserve diagnostic evidence'
        if state=='complete':break
        time.sleep(10)
    else:raise TimeoutError('GNOT wait limit')
    assert (BASE_ART/'gnot_pilot_summary.json').exists()
    write(BASE_ART/'other_algorithms_status.json',dict(state='bounded_multicase_checks_complete',parallel_execution_disclosed=True,
        original_supervisor_replaced_without_training_interruption=True))
    write(ART/'remaining_status.json',dict(state='remaining_training_decisions_complete'))
    verify()


if __name__=='__main__':main()
