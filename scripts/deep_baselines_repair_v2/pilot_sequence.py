"""Finite sequential pilots only; no automatic formal/second-seed launches."""
from repair_context import *
import signal
import subprocess


def main():
    active=None
    def stop(signum,frame):raise KeyboardInterrupt('User/controller stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        for arm in ARMS:
            target=ROOT/'pilots'/f'{arm}_v2_seed{SEED}'
            assert not (target/'decision.json').exists()
            cmd=[sys.executable,str(HERE/'launch.py'),'--key',arm+'_repair_v2_pilot','--',sys.executable,str(HERE/'pilot.py'),'run','--arm',arm,'--version','v2']
            with (ROOT/(arm+'_v2_pilot_controller.log')).open('x') as f:
                active=subprocess.Popen(cmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
                c.write(ROOT/'repair_queue_status.json',dict(at=c.now(),state='pilot_only',arm=arm,pid=active.pid))
                code=active.wait()
            assert code==0,'Preserve failed pilot and inspect before any retry'
            assert (target/'decision.json').exists()
            active=None
    finally:
        if active is not None and active.poll() is None:
            active.terminate();active.wait(timeout=45)
    c.write(ROOT/'repair_queue_status.json',dict(at=c.now(),state='pilots_complete_awaiting_evidence_review',formal_training_started=False))


if __name__=='__main__':main()
