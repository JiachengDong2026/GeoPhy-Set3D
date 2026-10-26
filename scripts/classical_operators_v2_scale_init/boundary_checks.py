"""Negative access checks use synthetic paths: no real GT is opened."""
import sys,tempfile
from scale_context_base import *
sys.path.insert(0,str(PROJECT/'src'))
from geophy_set3d.training.gt_read_guard import GTReadGuard

def main():
 with tempfile.TemporaryDirectory(prefix='geophy-v2-boundary-',dir=ROOT) as d:
  root=Path(d);row=dict(split='train',gt_volume_64_path='gt64.npy',source_target_path=str(root/'gt128.npy'),train_projection_path='allowed.npy')
  (root/'source_index.jsonl').write_text(json.dumps(row)+'\n');guard=GTReadGuard(root,root/'own-checkpoints')
  # Audit callback rejects before Python can perform a file open.
  guard.active=True;rejected=[]
  for name in ['gt64.npy','gt128.npy','val_projection.npy','ood_projection.npy','foreign.pt']:
   try:guard._audit('open',(str(root/name),'r',0))
   except PermissionError:rejected.append(name)
  assert len(rejected)==5
  guard.active=False
  write(ART/'boundary_tests.json',dict(passed=True,synthetic_paths_only=True,rejected=rejected,actual_GT_opened=False))
if __name__=='__main__':main()
