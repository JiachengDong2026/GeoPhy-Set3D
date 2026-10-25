"""Reuse existing global UUID lock and own-process conflict protection."""
import sys,time,argparse
from audit import PROJECT,ROOT
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller
UUID='GPU-37424437-e208-62d1-e617-53c421c94a49'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','pilot','formal','evaluate']);ap.add_argument('arm',choices=['deeponet','fno']);ap.add_argument('--recover-reason');ap.add_argument('--gpu',choices=['0','1'],default='0');a=ap.parse_args()
    global UUID
    if a.gpu=='1':UUID='GPU-cd80a78f-7d90-c291-4680-4895db9aac8b'
    original=controller.c.scheduler
    def restricted():
        s=original();devices=s.devices;s.devices=lambda snap:[r for r in devices(snap) if r['uuid']==UUID]
        def wait():
            began=time.monotonic();stable=0
            while time.monotonic()-began<60:
                snap=s.snapshot();controller.c.append(ROOT/'gpu_snapshots.jsonl',snap)
                row=next(r for r in s.devices(snap) if r['uuid']==UUID)
                stable=stable+1 if row['idle'] else 0
                if stable>=2:
                    lock=s.reserve(UUID)
                    if lock is not None:return UUID,lock
                time.sleep(15)
            raise RuntimeError('GPU0 unavailable; no preemption or fallback')
        s.wait_gpu=wait;return s
    controller.c.scheduler=restricted;controller.ROOT=ROOT
    script='evaluate.py' if a.phase=='evaluate' else 'worker.py'
    command=[sys.executable,str(PROJECT/'scripts/classical_operators_v1'/script),a.phase,a.arm]
    controller.run(a.phase+'_'+a.arm,command,a.recover_reason)
if __name__=='__main__':main()
