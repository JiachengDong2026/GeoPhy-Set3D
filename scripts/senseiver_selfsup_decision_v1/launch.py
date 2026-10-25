"""Reuse the existing locked controller in a new zero-update audit namespace."""
import sys
import time
from audit import PROJECT, ROOT
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller

UUID='GPU-37424437-e208-62d1-e617-53c421c94a49'

def main():
    original=controller.c.scheduler
    def restricted():
        s=original();devices=s.devices
        s.devices=lambda snap:[v for v in devices(snap) if v['uuid']==UUID]
        def wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                snap=s.snapshot();controller.c.append(ROOT/'gpu_snapshots.jsonl',snap)
                row=next(v for v in s.devices(snap) if v['uuid']==UUID)
                stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(UUID)
                    if lock is not None:return UUID,lock
                time.sleep(15)
            raise RuntimeError('GPU0 not idle or locked; stop without other tasks')
        s.wait_gpu=wait
        return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    controller.run('zero_update_diagnostic',[sys.executable,str(PROJECT/'scripts/senseiver_selfsup_decision_v1/diagnose.py')])

if __name__=='__main__':
    main()
