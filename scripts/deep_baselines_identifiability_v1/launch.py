"""Bounded GPU0-only adapter over existing lock and conflict-safe controller."""
from diag_context import *
import argparse
import time
import controller

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--recover-reason');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    ROOT.mkdir(exist_ok=True);original=c.scheduler
    def restricted():
        s=original();devices=s.devices;s.devices=lambda snap:[d for d in devices(snap) if d['uuid']==UUID]
        def bounded_wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                row=next(d for d in s.devices(s.snapshot()) if d['uuid']==UUID)
                stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(UUID)
                    if lock is not None:return UUID,lock
                time.sleep(15)
            c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='BLOCKED_GPU0_UNAVAILABLE',key=a.key,formal_training_authorized=False))
            raise RuntimeError('GPU0 unavailable after bounded60second acquisition; no other GPU allowed')
        s.wait_gpu=bounded_wait
        return s
    c.scheduler=restricted;controller.ROOT=ROOT
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    controller.run(a.key,cmd,a.recover_reason)

if __name__=='__main__':main()
