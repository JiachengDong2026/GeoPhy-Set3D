"""Read-only original objective directional derivatives, not network training."""
import sys
from unet_paths import *
sys.path.insert(0,str(PROJECT/'src'))
import torch
from geophy_set3d.training.classical_operators import load_config
from geophy_set3d.training.selfsup import make_dataset,objective,projection_scores
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard

class ConstantBranches(torch.nn.Module):
 def __init__(self,a,b):
  super().__init__();self.a=torch.nn.Parameter(torch.tensor(a));self.b=torch.nn.Parameter(torch.tensor(b));self.calls=0
 def forward(self,batch):
  c=self.a if self.calls==0 else self.b;self.calls+=1
  return c.expand(len(batch['case_id']),1,64,64,64)

def main():
 torch.set_num_threads(2);c=load_config('fno');guard=GTReadGuard(DATA,ROOT);rows=[];c0=read(ART/'reused_scale.json')['c0']
 with guard:
  ds=make_dataset(c)
  for indices in [[0,1],[2,3]]:
   batch=dual_measurement_collate([ds[i] for i in indices])
   for ca,cb in [(v,v) for v in [0.,1e-10,1e-8,1e-6,c0]]+[(c0,.5*c0)]:
    model=ConstantBranches(ca,cb);total,parts,volumes,projections=objective(model,batch)
    grads={}
    for name,loss in dict(total=total,**parts).items():
     g=torch.autograd.grad(loss,(model.a,model.b),retain_graph=True,allow_unused=True)
     grads[name]=[float(v) if v is not None else None for v in g]
    scores=projection_scores(projections,batch)
    rows.append(dict(indices=indices,A=ca,B=cb,total=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},dL_dc_AB=grads,
     H_relative_l1=sum(v['relative_l1'] for v in scores['heldout'])/len(scores['heldout'])))
  access=guard.summary()
 write(ART/'objective_direction_audit.json',dict(original_objective_sha256=sha(PROJECT/'src/geophy_set3d/training/selfsup.py'),optimizer_updates=0,rows=rows,access=access,
  caveat='Constant-subspace directional AD only. At zero abs/clamp returns a subgradient. Positive derivative implies decreasing c in that direction; not proof of all voxel directions or actual network trajectory. No scale or loss selected.'))
 print(json.dumps(rows,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
