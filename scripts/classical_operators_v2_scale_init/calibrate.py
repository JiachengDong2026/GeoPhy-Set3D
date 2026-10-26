"""One predeclared train-A/B-only Huber physical constant fit, on CPU."""
import math,sys
from common import *
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch
from geophy_set3d.training.classical_operators import load_config
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.gt_read_guard import GTReadGuard

def main():
 torch.set_num_threads(2)
 assert not (ART/'scale_calibration.json').exists(), 'Never silently recalibrate'
 c=load_config('deeponet');root=Path(c['dataset']['root'])
 guard=GTReadGuard(root,ROOT)
 rows=read_lines(root/'source_index.jsonl')
 trainpaths={str((root/r['train_projection_path']).absolute()) for r in rows if r['split']=='train'}
 guard.allowed={p:i for p,i in guard.allowed.items() if p in trainpaths}
 sources={str(p):sha(p) for p in [root/'source_index.jsonl',root/'train_statistics.json',root/'geometry/train_geometry.npz']}
 xs=[];ys=[];ledger=[]
 with guard:
  ds=make_dataset(c);assert len(ds)==166
  for i in range(len(ds)):
   row,p=ds._projections(i)
   a,b,h=ds.sampler.sample(ds.station_count,epoch=0,sample_index=i,case_id=row['case_id'])
   assert not set(a)&set(b) and not (set(a)|set(b))&set(h)
   path=root/row['train_projection_path'];sources[str(path)]=sha(path)
   for label,ids in [('A',a),('B',b)]:
    v=ds._views(p,ids,'input');mask=v['input_valid_ray'].numpy()
    y=v['input_projection_raw'].numpy()[mask].astype('float64')/ds.y_ref
    ell=v['input_path_length_m'].numpy()[mask].astype('float64')
    assert np.isfinite(y).all() and (y>=0).all() and np.isfinite(ell).all() and (ell>0).all()
    xs.append(1e6*ell/ds.y_ref);ys.append(y)
    ledger.append(dict(index=i,case_id=row['case_id'],branch=label,stations=ids.tolist(),valid_rays=len(y),selected_values_sha256=hashlib.sha256(y.tobytes()+ell.tobytes()).hexdigest()))
  access=guard.summary()
 x=np.concatenate(xs);y=np.concatenate(ys)
 lo,hi=0.,1.
 for _ in range(80):
  q=(lo+hi)/2;g=float(np.sum(x*np.clip((x*q-y)/.01,-1,1),dtype=np.float64))
  if g>0:hi=q
  else:lo=q
 q=(lo+hi)/2;a=1/16.043;b=1/28.965;c0=b*q/(a-(a-b)*q)
 z0=math.log(c0/(1-c0)) if 0<c0<1 else None
 result=dict(method='Fixed FP64 80-step bisection of sum x*clip((x*q-y)/0.01,-1,1)=0',
  units=dict(Y='ppm m',L='m',c='CH4 mass fraction',q='CH4 mole fraction'),Y_SCALE=ds.y_ref,
  c0=c0,q=q,z0=z0,valid_rays=len(y),zero_measurement_rays=int((y==0).sum()),
  normalized_measurement_quantiles=np.quantile(y,[0,.25,.5,.75,.95,.99,1]).tolist(),
  length_m_quantiles=np.quantile(x*ds.y_ref/1e6,[0,.5,1]).tolist(),
  root_gradient=float(np.sum(x*np.clip((x*q-y)/.01,-1,1))),
  residual_rule='alpha=min(1,0.1/maxabs(fresh training0..3 A/B logits-existing output bias))',
  necessary_center_condition=bool(z0 is not None and abs(z0)+.1<=10),
  sources_sha256=sources,access=access,logical_value_access='Only sampled epoch0 train A/B; no H value indexed; full projection bytes hashed only for identity',
  protocol_sha256=sha(PROJECT/'docs/classical_operators_v2_scale_init_protocol_zh.md'),design_commit=git('rev-parse','HEAD'))
 write(ART/'calibration_access_ledger.json',ledger);result['ledger_sha256']=sha(ART/'calibration_access_ledger.json')
 write(ART/'scale_calibration.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('sources_sha256',)},ensure_ascii=False),flush=True)

def read_lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
if __name__=='__main__':main()
