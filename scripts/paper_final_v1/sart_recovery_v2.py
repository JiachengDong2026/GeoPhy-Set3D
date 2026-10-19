"""Versioned continuation of failed SART development, keeping all original logs."""
from common import *
import fcntl
from sart import BASE
from sart_pipeline import stage,status


def main():
    mutex=(BASE/'pipeline.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert verified(BASE/'cache_checks_v2.json')['passed'];verified(BASE/'implementation_amendment_v2.json')
    stage('validation_predictions_recovery_v2',['scripts/paper_final_v1/sart.py','work','--split','val'])
    stage('validation_score_verify_report',['scripts/paper_final_v1/sart_pipeline.py','finish'])
    evidence={str(OUT/p):sha(OUT/p) for p in ('sart/registration.json','sart/selected.json','sart/numerical_checks.json','sart/integration_checks.json','sart/cache_checks_v2.json','sart/implementation_amendment_v2.json','evaluation/sart_dev/all_predictions_sealed.json','evaluation/sart_dev/summary.json','evaluation/sart_dev/independent_verification.json')}
    freeze(BASE/'development_complete.json',dict(at=now(),passed=True,evidence=evidence,calibration_reconstructions=24,validation_reconstructions=612,final_GT_read=False,software_recovery='cache eviction, v2; numerical solver unchanged'))
    status('complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
