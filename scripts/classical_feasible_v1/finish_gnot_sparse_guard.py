"""Gate one GNOT final epoch on independent saved-field review; never change criteria."""
from sparse_guard_v2_context import *
from review_gnot_sparse_guard_pilot import numpy_health
import subprocess


def call(name,args,cpu=False):
    with (ROOT/(name+'.log')).open('x') as log:
        env=dict(os.environ)
        if cpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(ART/'gnot_continuation_status.json',dict(state=name,pid=process.pid))
        code=process.wait()
    assert code==0,(name,code,'Preserve failure; do not replay')


def main():
    verify();registration=read(ART/'gnot_continuation_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        path=BASE_ART/'gnot_multicase_review.json'
        if path.exists():break
        time.sleep(5)
    else:raise TimeoutError('GNOT independent boundary review wait')
    review=read(path);assert review['passed'] and review['terminal_epoch']==29
    stats=[];guard=GTReadGuard(DATA,BASE_ROOT)
    with guard:
        for row in review['records']:
            if row['index']>=(4 if row['split']=='val' else 3):continue
            assert sha(row['path'])==row['sha256']
            with np.load(row['path'],allow_pickle=False) as arrays:
                for branch in ['A','B']:stats.append(dict(split=row['split'],index=row['index'],branch=branch,**numpy_health(arrays[branch+'_normal'])))
    passed=healthy(stats)
    write(ART/'gnot_guard_boundary_audit.json',dict(passed=passed,health=stats,independent_normal_fields=14,
        access=guard.summary(),GT_used=False,new_updates=0,original_review_sha256=sha(path)))
    if not passed:
        write(ART/'gnot_continuation_status.json',dict(state='boundary_revised_guard_failed',new_updates=0));return
    script=PROJECT/'scripts/classical_feasible_v1'
    call('gnot_v2_controller',[str(script/'launch.py'),'pilot_gnot_sparse_guard_v2','pilot_gnot_sparse_guard_v2.py'])
    call('gnot_v2_independent_review',[str(script/'review_gnot_sparse_guard_pilot.py')],cpu=True)
    write(ART/'gnot_continuation_status.json',dict(state='complete',summary=read(ART/'gnot_pilot_summary.json')))
    verify()


if __name__=='__main__':main()
