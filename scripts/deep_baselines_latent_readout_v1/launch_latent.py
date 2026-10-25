"""Reuse existing UUID controller, isolated state; never call a training entrypoint."""
from latent_context import *
import argparse,time
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=list(EXPECTED));a=ap.parse_args()
    original=controller.c.scheduler
    def restricted():
        s=original();devices=s.devices;s.devices=lambda snap:[v for v in devices(snap) if v['uuid']==old.UUID]
        def wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                row=next(v for v in s.devices(s.snapshot()) if v['uuid']==old.UUID);stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(old.UUID)
                    if lock is not None:return old.UUID,lock
                time.sleep(15)
            raise RuntimeError('GPU0 unavailable; stop without touching other devices')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    controller.run('extract_'+a.arm,[sys.executable,str(HERE/'extract_latent.py'),'--arm',a.arm])
if __name__=='__main__':main()
