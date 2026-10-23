"""Continue after the documented seal-only software error; never retrain."""
from common import *
import fcntl
from pipeline import state,stage,jobs


def main():
    mutex=(OUT/'pipeline.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=verified(OUT/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['evidence'])
    assert read(OUT/'stages/final_all_predictions.json')['exit_code']==0
    assert verified(OUT/'recovery/seal_checkpoint_allowlist_v2/checks.json')['passed']
    stage('final_all_seal_recovery_v2',['scripts/paper_final_v1/seal_recovery_v2.py','seal'])
    jobs('final_independent_scoring',[dict(key='score',args=['scripts/paper_final_v1/evaluation.py','score','--split','test'])])
    stage('final_statistics',['scripts/paper_final_v1/evaluation.py','summarize','--split','test'])
    stage('ablation_final_statistics',['scripts/paper_final_v1/ablation_statistics.py','--split','test'])
    stage('final_independent_verification',['scripts/paper_final_v1/verify.py','--split','test'])
    stage('final_figures_csv',['scripts/paper_final_v1/report.py','export','--base',str(OUT/'evaluation/test')])
    stage('final_acceptance',['scripts/paper_final_v1/finalize.py','accept'])
    state('complete',acceptance=str(OUT/'final_acceptance.json'),seal_recovery=str(OUT/'recovery/seal_checkpoint_allowlist_v2/amendment.json'))


if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise
