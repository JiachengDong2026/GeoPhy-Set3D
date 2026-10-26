"""Recheck selected-value ledger and stationary physical root; never select a scale."""
import sys,hashlib
from scale_context_base import *
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
from geophy_set3d.training.gt_read_guard import GTReadGuard

def main():
 root=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1');scale=read(ART/'scale_calibration.json');ledger=read(ART/'calibration_access_ledger.json')
 records={r['case_id']:r for r in [json.loads(s) for s in (root/'source_index.jsonl').read_text().splitlines()] if r['split']=='train'}
 guard=GTReadGuard(root,ROOT);allowed={str((root/r['train_projection_path']).absolute()) for r in records.values()};guard.allowed={p:i for p,i in guard.allowed.items() if p in allowed}
 xs=[];ys=[]
 with guard:
  with np.load(root/'geometry/train_geometry.npz') as g:valid=g['valid_ray'];length=g['path_length_m'].astype('float32')
  # Model-visible geometry is FP32, matching original frozen Dataset._views.
  for entry in ledger:
   r=records[entry['case_id']];ids=np.asarray(entry['stations']);mask=valid[ids]
   p=np.load(root/r['train_projection_path'],mmap_mode='r',allow_pickle=False)
   y=np.array(p[ids],copy=True)[mask].astype('float64')/scale['Y_SCALE'];ell=length[ids][mask].astype('float64')
   assert len(y)==entry['valid_rays']
   assert hashlib.sha256(y.tobytes()+ell.tobytes()).hexdigest()==entry['selected_values_sha256']
   xs.append(1e6*ell/scale['Y_SCALE']);ys.append(y)
  access=guard.summary()
 x=np.concatenate(xs);y=np.concatenate(ys);q=scale['q'];c=scale['c0'];a=1/16.043;b=1/28.965
 assert len(y)==scale['valid_rays'];assert np.isclose(a*c/(b+(a-b)*c),q,rtol=1e-14)
 def derivative(t):return float(np.sum(x*np.maximum(-1,np.minimum(1,(x*t-y)/.01))))
 left=derivative(q-1e-10);right=derivative(q+1e-10);assert left<0<right
 residual=x*q-y;curvature=float(np.sum(x[np.abs(residual)<.01]**2)/.01);assert curvature>0
 write(ART/'scale_independent_review.json',dict(passed=True,ledger_entries=len(ledger),valid_rays=len(y),gradient_at_registered_q=derivative(q),gradient_left=left,gradient_right=right,positive_local_curvature=curvature,inverse_physics_verified=True,c0_sigmoid_derivative=c*(1-c),access=access,no_alternative_scale_selected=True,geometry_precision='Model-visible FP32, then independent FP64 fit check'))
 print('independent ledger/physics/convex-root verification passed',len(y),'rays')
if __name__=='__main__':main()
