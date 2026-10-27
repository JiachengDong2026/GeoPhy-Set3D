"""One GPU per job, existing UUID locks and conflict handling; no hidden replay."""
from feasible_context import *
import argparse
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller


def main():
    ap=argparse.ArgumentParser();ap.add_argument('key');ap.add_argument('script');ap.add_argument('arguments',nargs='*');a=ap.parse_args()
    verify();ROOT.joinpath('scheduling').mkdir(exist_ok=True)
    original=controller.c.scheduler
    def restricted():
        s=original();s.BASE=ROOT
        def wait():
            previous=set();began=time.monotonic()
            while time.monotonic()-began<60:
                snap=s.snapshot();append(ROOT/f'{a.key}_gpu_snapshots.jsonl',snap)
                idle={d['uuid'] for d in s.devices(snap) if d['idle']}
                for uuid in sorted(idle & previous):
                    lock=s.reserve(uuid)
                    if lock is not None:return uuid,lock
                previous=idle;time.sleep(15)
            raise RuntimeError('No safely available GPU within bounded60s; no preemption')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    controller.run(a.key,[sys.executable,str(PROJECT/'scripts/classical_feasible_v1'/a.script),*a.arguments])


if __name__=='__main__':main()
