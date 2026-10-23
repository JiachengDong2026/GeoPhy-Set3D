"""No training: one exclusive inference job, then CPU scoring/analysis/reporting."""
from context import *
import signal
import traceback


def main():
    (BASE/'scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_hashes(verified(BASE/'protocol.json')['sources']);s=scheduler()
        try:
            artifact=BASE/'all_predictions_sealed.json'
            if not artifact.exists():
                assert not s.task_path('inference').exists(),'Existing attempt requires inspected recovery; do not replay silently'
                uuid,fd=s.wait_gpu();item=s.launch('inference',command('infer.py'),uuid,fd)
                while not s.done(item):
                    snap=s.snapshot()
                    foreign=[r for r in snap['gpu_processes'].splitlines() if r.strip() and r.split(',')[0].strip()==uuid and int(r.split(',')[1])!=item['record']['pid']]
                    if foreign:
                        freeze(BASE/'resource_collision.json',dict(at=now(),foreign=foreign,action='interrupt own inference only, no foreign signal',task=item['record']))
                        os.kill(item['record']['pid'],signal.SIGINT);item['process'].wait();s.finish(item,valid=False)
                    time.sleep(15)
                s.finish(item);assert artifact.exists()
            for script,done in [('score.py','GT64_access.json'),('analysis.py','independent_statistics_verification.json'),('reports.py','acceptance.json')]:
                if (BASE/done).exists():continue
                with (BASE/'scheduling'/f'{script}.log').open('x') as log:subprocess.run(command(script),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            freeze(BASE/'final_resources.json',resources())
            write(BASE/'status.json',dict(at=now(),state='complete_exploratory_results',independent_test='BLOCKED_NO_INDEPENDENT_DATA'))
        except BaseException:
            write(BASE/'status.json',dict(at=now(),state='failed_preserved',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
