"""Read-only stage observer: verify completed pilots and write truthful progress."""
from feasible_context import *
import subprocess


def main():
    verify();deadline=time.monotonic()+12*3600;reviewed=set()
    # One independent CPU review per completed result. Never launches training.
    while time.monotonic()<deadline:
        for arm in ['deeponet','fno','senseiver','gnot','deepsets','deeponet_lr2e5']:
            summary=ART/(arm+'_summary.json' if arm.endswith('lr2e5') else arm+'_pilot_summary.json')
            review=ART/(arm+'_review.json' if arm.endswith('lr2e5') else arm+'_multicase_review.json')
            if not summary.exists() or arm in reviewed:continue
            if not review.exists():
                script='review_deeponet_stabilization.py' if arm.endswith('lr2e5') else 'review_multicase.py'
                with (ROOT/(arm+'_completion_review.log')).open('x') as log:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    result=subprocess.run([sys.executable,str(PROJECT/'scripts/classical_feasible_v1'/script),arm],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,env=env)
                assert result.returncode==0,(arm,'Independent review failed; preserve log')
            assert read(review)['passed'];reviewed.add(arm)
        with (ROOT/'progress_refresh.log').open('a') as log:
            subprocess.run([sys.executable,str(PROJECT/'scripts/classical_feasible_v1/status_report.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status=dict(reviewed=sorted(reviewed),elapsed_seconds=12*3600-(deadline-time.monotonic()))
        if (ART/'unet_summary.json').exists():
            unet=read(ART/'unet_summary.json');status['unet_decision']=unet['decision']
            test_done=not unet['formal_complete'] or (ART/'unet_test_score.json').exists()
            stabilization=read(ART/'deeponet_stabilization_status.json')['state']
            if test_done and all(arm in reviewed for arm in ['deeponet','fno','senseiver','gnot','deepsets']) and stabilization in ['complete','deferred_resource_wait_limit']:
                if stabilization=='complete':assert 'deeponet_lr2e5' in reviewed
                status['state']='all_training_decisions_and_reviews_complete'
                write(ART/'completion_observer_status.json',status);verify();return
        status['state']='monitoring'
        write(ART/'completion_observer_status.json',status);time.sleep(45)
    raise TimeoutError('12h observer limit; inspect training/controller logs; no new training launched')


if __name__=='__main__':main()
