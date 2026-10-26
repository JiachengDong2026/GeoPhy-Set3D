"""One deterministic final-layer scaling rule, train0..3 A/B only."""
from unet_context import *
from geophy_set3d.models.baseline_unet3d_v1 import UNet3DV1
from geophy_set3d.training.unet_baseline_v1 import initialize_scale

def stats(p,z):
 p=p.double();z=z.double()
 return dict(mean=float(p.mean()),std=float(p.std(correction=0)),min=float(p.min()),max=float(p.max()),nearzero=float((p<1e-10).double().mean()),saturated=float((z.abs()>10).double().mean()),
  logit_mean=float(z.mean()),logit_std=float(z.std(correction=0)),logit_quantiles=torch.quantile(z.flatten(),torch.tensor([0,.01,.5,.99,1],dtype=torch.float64)).tolist())
def main():
 assert not (ART/'initialization.json').exists();setup();scale=read(ART/'reused_scale.json');c=load_config()
 guard=GTReadGuard(DATA,ROOT)
 records=[json.loads(s) for s in (DATA/'source_index.jsonl').read_text().splitlines()]
 trainpaths={str((DATA/r['train_projection_path']).absolute()) for r in records if r['split']=='train'}
 guard.allowed={p:i for p,i in guard.allowed.items() if p in trainpaths}
 with guard,torch.no_grad():
  ds=make_dataset(c);batches=[];ledger=[]
  for i in range(4):
   row,p=ds._projections(i);a,b,h=ds.sampler.sample(ds.station_count,epoch=0,sample_index=i,case_id=row['case_id'])
   for label,ids in [('A',a),('B',b)]:
    assert not set(ids)&set(h)
    sample=ds._measurement_sample(row,p,ids,np.empty(0,dtype=np.int64));batches.append(projection_pool_collate([sample]));ledger.append(dict(index=i,branch=label,stations=ids.tolist()))
  model=UNet3DV1(**c['model']['settings']).eval();initial=state_hash(model.state_dict());M=0.;before=[]
  for b in batches:
   p,z=model(b,return_logits=True);M=max(M,float((z-model.readout.bias).abs().max()));before.append(stats(p,z))
  alpha=min(1.,.1/M) if M else 1.;initialize_scale(model,scale['c0'],alpha);after=[]
  directory=ROOT/'unet/initialization';directory.mkdir(parents=True,exist_ok=False);arrays=[]
  for i,b in enumerate(batches):
   p,z=model(b,return_logits=True);s=stats(p,z);assert s['mean']>=1e-8 and s['nearzero']<.95 and s['saturated']<.95 and torch.isfinite(p).all() and (p>0).all() and (p<1).all();after.append(s)
   path=directory/f'train{ledger[i]["index"]}_{ledger[i]["branch"]}.npz';np.savez_compressed(path,volume=p.numpy(),logits=z.numpy());arrays.append(dict(path=str(path),sha256=sha(path)))
  result=dict(c0=scale['c0'],z0=scale['z0'],alpha=alpha,maxabs_fresh_residual=M,parameters=sum(p.numel() for p in model.parameters()),
   fresh_state_sha256=initial,initialized_state_sha256=state_hash(model.state_dict()),before=before,after=after,ledger=ledger,arrays=arrays,access=guard.summary(),
   model_source_sha256=sha(PROJECT/'src/geophy_set3d/models/baseline_unet3d_v1.py'),adapter_source_sha256=sha(PROJECT/'src/geophy_set3d/models/baseline_fno_v1.py'),design_registration_sha256=sha(ART/'design_registration.json'))
 write(ART/'initialization.json',result);print('U-Net scale',result['c0'],'alpha',alpha,'parameters',result['parameters'],'mean',after[0]['mean'],flush=True)
if __name__=='__main__':main()
