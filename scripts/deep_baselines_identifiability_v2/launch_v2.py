"""Existing global resource controller, restricted to GPU0, bounded wait."""
from ctx_v2 import *
import argparse
import time
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--recover-reason');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    original=controller.c.scheduler
    def restricted():
        s=original();devices=s.devices;s.devices=lambda snap:[v for v in devices(snap) if v['uuid']==UUID]
        def wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                row=next(v for v in s.devices(s.snapshot()) if v['uuid']==UUID);stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(UUID)
                    if lock is not None:return UUID,lock
                time.sleep(15)
            write(ROOT/'status.json',dict(at=now(),state='BLOCKED_GPU0_UNAVAILABLE',key=a.key,formal_training_authorized=False))
            raise RuntimeError('GPU0 unavailable in60seconds; other GPUs forbidden')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    controller.run(a.key,command,a.recover_reason)

if __name__=='__main__':main()
