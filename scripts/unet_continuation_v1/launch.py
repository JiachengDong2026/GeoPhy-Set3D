"""Reuse existing UUID lock, double-idle checks, and foreign-conflict monitoring."""
import sys,os,time
from continuation_context import *
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller


def main():
    verify_all()
    assert read(ART/'preflight.json')['passed']
    original=controller.c.scheduler
    def restricted():
        s=original();s.BASE=ROOT
        def wait():
            previous=set();began=time.monotonic()
            while time.monotonic()-began<60:
                snap=s.snapshot();append(ROOT/'gpu_snapshots.jsonl',snap)
                idle={v['uuid'] for v in s.devices(snap) if v['idle']}
                for uuid in sorted(idle & previous):
                    lock=s.reserve(uuid)
                    if lock is not None:
                        os.environ['GEOPHY_CONTINUATION_UUID_LOCKED']=uuid
                        return uuid,lock
                previous=idle;time.sleep(15)
            raise RuntimeError('No safely available GPU within 60s')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    controller.run('case0_continuation',[sys.executable,str(PROJECT/'scripts/unet_continuation_v1/train.py')])


if __name__=='__main__':main()
