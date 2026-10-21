"""Wait for original siblings, preserve terminal evidence, exec same-budget recovery."""
import sys,os,time,pathlib,psutil,fcntl
P=pathlib.Path('/home/djc/projects/GeoPhy-Set3D')
sys.path.insert(0,str(P/'scripts/oari_targeted_v1'))
from stage import *
lock=(BASE/'recovery/supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
original_pid=319779
while psutil.pid_exists(original_pid):
 try:
  proc=psutil.Process(original_pid)
  if proc.status()==psutil.STATUS_ZOMBIE:break
  assert proc.username()=='djc' and str(HERE/'controller.py') in proc.cmdline()
 except psutil.NoSuchProcess:break
 time.sleep(15)
terminal=read(BASE/'status.json')
assert terminal['state']=='failed' and 'One or more jobs failed' in terminal['traceback']
for arm,seed in TASKS[:2]:
 summary=read(BASE/'training'/f'{arm}_seed{seed}'/'summary.json')
 assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900
resume=verified(BASE/'recovery/allow_original_epoch_commit_resume.json');assert resume['allow_resume']
assert sha(BASE/'training'/resume['task']/'checkpoints/last.pt')==resume['checkpoint_sha256']
verify_hashes(verified(BASE/'protocol.json')['sources'])
freeze(BASE/'recovery/original_controller_terminal_status.json',terminal)
freeze(BASE/'recovery/controller_relaunch.json',dict(at=now(),pid=os.getpid(),original_controller_pid=original_pid,
 source=str(HERE/'controller.py'),source_sha256=sha(HERE/'controller.py'),
 resumed_epoch=4,resumed_updates=332,skip_completed_siblings=True,resource_exclusions_sha256=sha(BASE/'scheduling/resource_exclusions.json'),
 full_budget_unchanged=True,original_training_engine_unchanged=True))
print('Original siblings complete; exec recovery controller',flush=True)
os.execv(sys.executable,[sys.executable,str(HERE/'controller.py')])
