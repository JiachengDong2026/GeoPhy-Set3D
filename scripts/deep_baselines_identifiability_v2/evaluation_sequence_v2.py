"""Explicit evaluation-only recovery; never imports or invokes training worker."""
from ctx_v2 import *
import subprocess

def main():
    for arm in design()['arms']:
        training=verified(ROOT/'runs'/arm/'training_result.json');assert training['updates']==200
    write(ROOT/'status.json',dict(at=now(),state='EVALUATION_ONLY_RECOVERY',completed_training_updates=400,
        cumulative_updates=2400,extra_training_authorized=False,reason='normalized companion swap rounding; raw and network outputs invariant'))
    try:
        for arm in design()['arms']:
            subprocess.run([sys.executable,str(HERE/'evaluation_recovery_v2.py'),'--arm',arm],cwd=PROJECT,check=True)
    except BaseException as e:
        write(ROOT/'status.json',dict(at=now(),state='FAILED_EVALUATION_NO_AUTO_RETRY',error=repr(e),training_updates_preserved=400));raise
    write(ROOT/'status.json',dict(at=now(),state='COMPLETE_PENDING_INDEPENDENT_REVIEW',new_updates=400,cumulative_updates=2400,remaining_budget=100,formal_training_authorized=False))

if __name__=='__main__':main()
