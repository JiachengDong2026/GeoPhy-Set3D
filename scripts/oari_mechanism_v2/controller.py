"""Single locked GPU, engineering gate, resumable inference, offline finish."""
from runtime import *
import fcntl
import time
import traceback


def main():
    validate()
    (BASE / 'scheduling').mkdir(exist_ok=True)
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = scheduler()
        s.run_gpu_once('engineering', [sys.executable, str(HERE / 'worker.py'), '--limit', '2'], BASE / 'engineering_checks.json')
        assert verified(BASE / 'engineering_checks.json')['passed']
        s.run_gpu_once('inference', [sys.executable, str(HERE / 'worker.py')], BASE / 'inference_complete.json')
        for script, sentinel in [('analyze.py', 'summary.json'), ('verify.py', 'independent_verification.json'), ('report_results.py', 'acceptance.json')]:
            if (BASE / sentinel).exists():
                continue
            logpath = BASE / 'scheduling' / f'{script}_{time.time_ns()}.log'
            with logpath.open('x') as log:
                subprocess.run([sys.executable, str(HERE / script)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert verified(BASE / 'acceptance.json')['completed']
        write(BASE / 'status.json', dict(at=now(), state='accepted_pending_git_sync', acceptance=str(BASE / 'acceptance.json')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        freeze(BASE / f'failure_controller_{time.time_ns()}.json', dict(at=now(), traceback=traceback.format_exc()))
        write(BASE / 'status.json', dict(at=now(), state='failed_preserved'))
        raise
