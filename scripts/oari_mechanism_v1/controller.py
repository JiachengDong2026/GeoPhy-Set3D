"""CPU reuse first; one independently idle locked GPU; no training interference."""
from runtime import *
import fcntl


def main():
    (BASE/'scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (BASE/'scheduling/B.log').open('x') as log:
            subprocess.run([sys.executable,str(HERE/'attenuation.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        s=scheduler();s.run_gpu_once('probe',[sys.executable,str(HERE/'probe.py')],BASE/'probe_complete.json')
        with (BASE/'scheduling/finish.log').open('x') as log:
            subprocess.run([sys.executable,str(HERE/'finish.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert verified(BASE/'acceptance.json')['completed']
        write(BASE/'status.json',dict(at=now(),state='complete'))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(BASE/'controller_failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
