"""Global UUID flock; 15s double-idle snapshots; never resume or replay training."""
import sys,time,os,argparse
from unet_paths import *
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
import controller

def main():
 arm=sys.argv[1];assert arm=='unet'
 assert not (ROOT/arm/'pilot').exists(),'No automatic resume/replay'
 original=controller.c.scheduler
 def restricted():
  s=original()
  def wait():
   began=time.monotonic();previous=set()
   while time.monotonic()-began<60:
    snap=s.snapshot();controller.c.append(ROOT/'gpu_snapshots.jsonl',snap)
    idle={r['uuid'] for r in s.devices(snap) if r['idle']}
    for uuid in sorted(idle & previous):
     lock=s.reserve(uuid)
     if lock is not None:
      os.environ['GEOPHY_UNET_UUID_LOCKED']=uuid
      return uuid,lock
    previous=idle;time.sleep(15)
   raise RuntimeError('No safely reservable GPU in bounded 60s; no preemption')
  s.wait_gpu=wait;return s
 controller.c.scheduler=restricted;controller.ROOT=ROOT
 controller.run('pilot_'+arm,[sys.executable,str(PROJECT/'scripts/unet_baseline_v1/worker.py'),arm])
if __name__=='__main__':main()
