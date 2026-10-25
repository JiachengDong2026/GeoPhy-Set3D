"""Finite two-run pipeline; no retries or queue resumption."""
from ctx_v2 import *
import subprocess

def main():
    reg=verified(ROOT/'registration.json');verify(reg['sources']);verify(reg['inputs'])
    assert read(ROOT.parent/'deep_baselines_v1/queue_status.json')['state']=='paused_by_user'
    assert verified(ROOT.parent/'deep_baselines_identifiability_v1/final_status.json')['total_updates']==2000
    write(ROOT/'status.json',dict(at=now(),state='RUNNING',new_update_limit=400,prior_updates=2000,formal_training_authorized=False))
    try:
        # Finish both training workers before exposing any H/truth result.
        for arm in design()['arms']:
            subprocess.run([sys.executable,str(HERE/'worker_v2.py'),'--arm',arm],cwd=PROJECT,check=True)
        for arm in design()['arms']:
            subprocess.run([sys.executable,str(HERE/'evaluate_v2.py'),'--arm',arm],cwd=PROJECT,check=True)
    except BaseException as e:
        write(ROOT/'status.json',dict(at=now(),state='FAILED_STOPPED_NO_AUTO_RETRY',error=repr(e),formal_training_authorized=False))
        raise
    write(ROOT/'status.json',dict(at=now(),state='COMPLETE_PENDING_INDEPENDENT_REVIEW',new_updates=400,cumulative_updates=2400,remaining_budget=100,formal_training_authorized=False))

if __name__=='__main__':main()
