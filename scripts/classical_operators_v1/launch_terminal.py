"""New one-shot audit namespace, same existing UUID reservation controller."""
import sys,time,argparse
from audit import PROJECT,ROOT,sha,write
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller
UUID='GPU-37424437-e208-62d1-e617-53c421c94a49'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['deeponet','fno']);a=ap.parse_args()
    target=ROOT/a.arm/'terminal_source_registration.json';assert not target.exists()
    write(target,dict(sources={p.name:sha(p) for p in [PROJECT/'scripts/classical_operators_v1/terminal_diagnostic.py',__file__]},updates=0))
    original=controller.c.scheduler
    def restricted():
        s=original();devices=s.devices;s.devices=lambda snap:[r for r in devices(snap) if r['uuid']==UUID]
        def wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                snap=s.snapshot();controller.c.append(ROOT/'gpu_snapshots.jsonl',snap)
                row=next(r for r in s.devices(snap) if r['uuid']==UUID);stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(UUID)
                    if lock is not None:return UUID,lock
                time.sleep(15)
            raise RuntimeError('GPU0 unavailable; no preemption')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    controller.run('terminal_'+a.arm,[sys.executable,str(PROJECT/'scripts/classical_operators_v1/terminal_diagnostic.py'),a.arm])
if __name__=='__main__':main()
