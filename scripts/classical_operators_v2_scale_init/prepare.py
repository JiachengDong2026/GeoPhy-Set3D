"""No-update forward calibration of one registered residual rule plus physics check."""
import sys,math
from common import *
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch
from geophy_set3d.training.classical_operators_v2_scale_init import load_config,REGISTRY,initialize_scale
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.utils import seed_everything
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.rvtc_matched import state_hash

def visible(ds,i):
 row,p=ds._projections(i);a,b,h=ds.sampler.sample(ds.station_count,epoch=0,sample_index=i,case_id=row['case_id'])
 return {label:projection_pool_collate([ds._measurement_sample(row,p,ids,np.empty(0,dtype=np.int64))]) for label,ids in [('A',a),('B',b)]}

def stats(p,z):
 p=p.double();z=z.double()
 return dict(mean=float(p.mean()),std=float(p.std(correction=0)),min=float(p.min()),max=float(p.max()),logit_mean=float(z.mean()),
  logit_std=float(z.std(correction=0)),logit_quantiles=torch.quantile(z.flatten(),torch.tensor([0,.01,.5,.99,1],dtype=torch.float64)).tolist(),
  nearzero=float((p<1e-10).double().mean()),saturated=float((z.abs()>10).double().mean()))

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 scale=read(ART/'scale_calibration.json');assert scale['necessary_center_condition']
 assert not (ART/'initialization_calibration.json').exists()
 c0=scale['c0'];guard=GTReadGuard(load_config('deeponet')['dataset']['root'],ROOT)
 # Strict calibration allowlist: train files only; retain original audit-counter indices.
 root=Path(load_config('deeponet')['dataset']['root'])
 trainpaths={str((root/r['train_projection_path']).absolute()) for r in [json.loads(s) for s in (root/'source_index.jsonl').read_text().splitlines()] if r['split']=='train'}
 guard.allowed={p:i for p,i in guard.allowed.items() if p in trainpaths}
 result=dict(c0=c0,z0=scale['z0'],arms={})
 with guard,torch.no_grad():
  ds=make_dataset(load_config('deeponet'));batches=[visible(ds,i) for i in range(4)]
  batch=batches[0]['A'];v=torch.full((1,1,64,64,64),c0)
  valid=batch['input_valid_ray'] & batch['view_mask'][:,:,None,None,None]
  expected=1e6*batch['input_path_length_m']*scale['q'];reference=project_input(v,batch)
  operator=ExplicitRayOperator();explicit=operator(v,batch)
  result['physics']=dict(original_max_absolute=float((reference[valid]-expected[valid]).abs().max()),explicit_max_absolute=float((explicit[valid]-expected[valid]).abs().max()),
   original_relative_max=float(((reference[valid]-expected[valid]).abs()/expected[valid].clamp_min(1e-12)).max()),valid_rays=int(valid.sum()))
  assert torch.allclose(reference[valid],expected[valid],rtol=1e-5,atol=1e-5)
  assert torch.allclose(explicit[valid],expected[valid],rtol=1e-5,atol=1e-5)
  del operator
  for arm in ['deeponet','fno']:
   seed_everything(20261011);c=load_config(arm);model=REGISTRY[arm](**c['model']['settings']);model.eval()
   original=state_hash(model.state_dict());M=0.;before=[]
   bias=model.bias if arm=='deeponet' else model.readout[-1].bias
   for i,branches in enumerate(batches):
    for key,batch in branches.items():
     p,z=model(batch,return_logits=True);M=max(M,float((z-bias).abs().max()));before.append(dict(index=i,branch=key,**stats(p,z)))
   alpha=min(1.,.1/M) if M else 1.
   initialize_scale(model,arm,c0,alpha);after=[]
   target=ROOT/arm/'initialization';target.mkdir(parents=True,exist_ok=False)
   for i,branches in enumerate(batches):
    for key,batch in branches.items():
     p,z=model(batch,return_logits=True);s=stats(p,z);after.append(dict(index=i,branch=key,**s))
     assert s['mean']>=1e-8 and s['nearzero']<.95 and s['saturated']<.95 and torch.isfinite(p).all() and (p>0).all() and (p<1).all()
     np.savez_compressed(target/f'train_{i}_{key}.npz',volume=p.numpy(),logits=z.numpy())
   result['arms'][arm]=dict(alpha=alpha,max_abs_fresh_residual=M,parameters=sum(p.numel() for p in model.parameters()),
    fresh_state_sha256=original,initialized_state_sha256=state_hash(model.state_dict()),before=before,after=after)
   print(arm,'alpha',alpha,'mean',after[0]['mean'],flush=True)
   del model
  result['access']=guard.summary()
 write(ART/'initialization_calibration.json',result)
if __name__=='__main__':main()
