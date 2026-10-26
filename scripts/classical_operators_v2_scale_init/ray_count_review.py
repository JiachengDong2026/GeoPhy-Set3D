"""Post-run CPU verification of unchanged input paths retaining every valid ray."""
import sys
from scale_context_base import *
sys.path.insert(0,str(PROJECT/'src'))
import torch
from geophy_set3d.training.classical_operators_v2_scale_init import load_config
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.models.baseline_deeponet_v1 import ray_maps
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator

def main():
 torch.set_num_threads(2);c=load_config('deeponet');guard=GTReadGuard(c['dataset']['root'],ROOT);checks=[]
 with guard,torch.no_grad():
  ds=make_dataset(c);batch=dual_measurement_collate([ds[0],ds[1]])
  for branch in ['A','B']:
   b=batch[branch];valid=b['input_valid_ray'].bool() & b['view_mask'][:,:,None,None,None]
   maps=ray_maps(b);assert torch.equal(maps[:,:,1:2].bool(),valid)
   op=ExplicitRayOperator();stencil=op.prepare(b,64)
   assert torch.equal(stencil.valid_ray,valid)
   assert torch.equal(stencil.valid_counts,valid.flatten(1).sum(1))
   checks.append(dict(branch=branch,original_valid_counts=valid.flatten(1).sum(1).tolist(),deeponet_map_count=int(maps[:,:,1:2].sum()),FNO_stencil_counts=stencil.valid_counts.tolist(),all_rays_retained=True));del op
  access=guard.summary()
 write(ART/'ray_count_review.json',dict(passed=True,checks=checks,access=access,model_parameter_updates=0))
if __name__=='__main__':main()
