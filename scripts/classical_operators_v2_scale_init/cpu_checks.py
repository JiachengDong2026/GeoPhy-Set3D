"""Full-resolution CPU initialization tests; stop before GPU on any failure."""
import sys,io,copy,math,traceback
from common import *
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
import numpy as np
import torch
from geophy_set3d.training.classical_operators_v2_scale_init import load_config,build_model,REGISTRY
from geophy_set3d.training.selfsup import make_dataset,objective
from geophy_set3d.training.utils import seed_everything
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from prepare import stats

def verify():
 r=read(ART/'registration.json')
 for p,h in r['source_sha256'].items():assert sha(PROJECT/p)==h,p
 return r

def run(arm):
 reg=verify();c=load_config(arm);scale=read(ART/'initialization_calibration.json');c0=scale['c0'];alpha=scale['arms'][arm]['alpha']
 target=ROOT/arm/'cpu';target.mkdir(parents=True,exist_ok=False)
 guard=GTReadGuard(c['dataset']['root'],ROOT)
 with guard:
  ds=make_dataset(c);batch=dual_measurement_collate([ds[0],ds[1]])
  seed_everything(20261011);model=build_model(c,c0,alpha);model.eval()
  initial=state_hash(model.state_dict());assert initial==scale['arms'][arm]['initialized_state_sha256']
  seed_everything(20261011);twin=build_model(c,c0,alpha);twin.eval();assert initial==state_hash(twin.state_dict())
  # Audit exact modified parameter whitelist against an independently fresh v1 graph.
  seed_everything(20261011);fresh=REGISTRY[arm](**c['model']['settings'])
  changed=[k for k,v in model.state_dict().items() if not torch.equal(v,fresh.state_dict()[k])]
  allowed=['bias','branch.2.weight','branch.2.bias'] if arm=='deeponet' else ['readout.2.weight','readout.2.bias']
  assert set(changed)<=set(allowed),changed
  assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in fresh.parameters())
  del fresh
  with torch.no_grad():
   p,z=model(batch['A'],return_logits=True);q=twin(batch['A']);assert torch.equal(p,q);del twin
   s=stats(p,z);assert torch.isfinite(p).all() and (p>0).all() and (p<1).all()
   assert s['mean']>=1e-8 and s['nearzero']<.95 and s['saturated']<.95
   buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);model.load_state_dict(torch.load(buffer,weights_only=True));assert torch.equal(p,model(batch['A']))
   padded=dict(batch['A'])
   for k,v in batch['A'].items():
    if isinstance(v,torch.Tensor) and (k.startswith('input_') or k=='view_mask'):
     extra=v[:,:1].clone()
     if k=='view_mask':extra.fill_(False)
     elif extra.is_floating_point():extra.fill_(float('nan'))
     padded[k]=torch.cat([v,extra],1)
   pad=model(padded);assert torch.equal(p,pad)
   invalid=dict(batch['A']);invalid['input_valid_ray']=torch.zeros_like(invalid['input_valid_ray'])
   for k in ['input_projection','input_projection_raw','input_path_length_m','input_t_near','input_ray_direction']:invalid[k]=torch.full_like(invalid[k],float('nan'))
   assert torch.isfinite(model(invalid)).all()
   zero=dict(batch['A'])
   for k in ['input_projection','input_projection_raw']:zero[k]=torch.zeros_like(zero[k])
   zp=model(zero);assert torch.isfinite(zp).all()
   donor_row,donor=ds._projections(1);sample=ds[0]['A'];aligned=dict(sample);aligned.update(ds._views(donor,sample['input_station_indices'].numpy(),'input'))
   db=projection_pool_collate([aligned]);wrong=dict(batch['A'])
   for k in ['input_projection','input_projection_raw']:
    wrong[k]=batch['A'][k].clone();wrong[k][0]=db[k][0]
   assert all(wrong[k] is v for k,v in batch['A'].items() if k not in ['input_projection','input_projection_raw'])
   assert torch.isfinite(model(wrong)).all()
  seen=[];hook=model.register_forward_pre_hook(lambda m,a:seen.append(id(a[0])))
  model.train();loss,parts,volumes,projections=objective(model,batch);hook.remove()
  assert seen==[id(batch['A']),id(batch['B'])] and id(batch['H']) not in seen
  loss.backward()
  gradients={k:dict(norm=float(p.grad.double().norm()),nonzero=int(p.grad.count_nonzero()),finite=bool(torch.isfinite(p.grad).all())) for k,p in model.named_parameters() if p.grad is not None}
  assert len(gradients)==len(list(model.parameters())) and all(g['finite'] for g in gradients.values())
  gn=math.sqrt(sum(g['norm']**2 for g in gradients.values()));assert math.isfinite(gn) and gn>0
  before={k:p.detach().clone() for k,p in model.named_parameters()}
  opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  delta=math.sqrt(sum(float((p.detach()-before[k]).double().square().sum()) for k,p in model.named_parameters()));assert math.isfinite(delta) and delta>0
  assert all(torch.isfinite(p).all() for p in model.parameters())
  access=guard.summary();assert not any(access[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
 result=dict(passed=True,arm=arm,initial_state_sha256=initial,changed_parameters=changed,parameters=sum(p.numel() for p in model.parameters()),
  initial_distribution=s,full_grid=[64,64,64],batch_size=2,initial_params_bitwise=True,initial_output_bitwise=True,serialization_bitwise=True,
  padding_bitwise=True,invalid_nan_safe=True,H_excluded=True,interventions_finite=True,loss=float(loss.detach()),gradients=gradients,
  preclip_gradient_norm=gn,adamw_update_l2=delta,actual_cpu_test_updates=1,pilot_updates=0,access=access)
 write(ART/f'{arm}_cpu_tests.json',result);print(arm,'CPU PASS grad',gn,'update',delta,flush=True)

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 arm=sys.argv[1]
 try:run(arm)
 except BaseException:
  write(ART/f'{arm}_cpu_failure.json',dict(passed=False,traceback=traceback.format_exc(),GPU_pilot_started=False));raise
if __name__=='__main__':main()
