"""Verify frozen evidence and summarize saved complete fields without training."""
import sys,time
from common import *
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch

def stats(v,z):
 v=v.astype('float64');z=z.astype('float64');m=float(v.mean())
 return dict(mean=m,std=float(v.std()),min=float(v.min()),max=float(v.max()),spatial_cv=float(v.std()/max(m,1e-30)),
  nearzero=float((v<1e-10).mean()),logit_mean=float(z.mean()),logit_std=float(z.std()),
  logit_quantiles=np.quantile(z,[0,.01,.25,.5,.75,.99,1]).tolist(),logit_lt_minus10=float((z< -10).mean()),logit_gt10=float((z>10).mean()))
def main():
 old=PROJECT/'docs/reports/artifacts/classical_operators_v1'
 publication=read(old/'publication_manifest.json')
 for p,h in publication['sha256'].items():assert sha(PROJECT/p)==h,p
 ck=read(old/'checkpoint_manifest.json')
 for r in ck.values():assert sha(r['path'])==r['sha256'],r['path']
 frozen={r['path']:r['sha256'] for r in ck.values()}
 frozen.update({
 '/data/djc/GeoPhy-Set3D/outputs/checkpoints/deepsets_selfsup_v1/best.pt':'949fb63be9068f676e7c6740fa8ce67b766d1f4d3ced86127e38b67a8eccdad0',
 '/scratch/djc/GeoPhy-Set3D/mrgr_oari_targeted_v1/training/uniform_plain_set_interaction_seed20261011/checkpoints/best.pt':'0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c'})
 for p,h in frozen.items():assert sha(p)==h,p
 tracked={p:sha(PROJECT/p) for p in git('ls-files').splitlines() if (PROJECT/p).is_file() and not p.startswith(('scripts/classical_operators_v2_scale_init/','docs/classical_operators_v2_scale_init'))}
 write(ROOT/'protected_tracked.json',tracked)
 result=dict(base_commit='279814b5152c3916913766e7a0fb28134e7d320d',publication_files_verified=len(publication['sha256']),frozen_sha256=frozen,protected_count=len(tracked),protected_manifest_sha256=sha(ROOT/'protected_tracked.json'),models={})
 rawroot=Path('/scratch/djc/GeoPhy-Set3D/classical_operators_v1')
 for arm in ['deeponet','fno']:
  pilot=rawroot/arm/'pilot';records=[]
  dirs=[pilot/'diagnostic_epoch0',rawroot/arm/'terminal_audit/interventions' if arm=='deeponet' else pilot/'diagnostic_epoch30']
  for d in dirs:
   if not d.exists():continue
   for p in sorted(d.glob('*.npz')):
    with np.load(p,allow_pickle=False) as a:
     for k in a.files:
      if k.endswith('_volume'):
       records.append(dict(source=str(p),sha256=sha(p),key=k,**stats(a[k],a[k.replace('_volume','_logits')])))
  checkpoints={}
  for kind in ['best','last']:
   c=torch.load(pilot/f'{kind}.pt',map_location='cpu',weights_only=False)
   checkpoints[kind]=dict(epoch=c['epoch'],updates=c['optimizer_steps'],optimizer_steps=sorted({int(v['step']) for v in c['optimizer']['state'].values()}),scheduler_epoch=c['scheduler']['last_epoch'])
  result['models'][arm]=dict(epoch0=read(old/f'{arm}_epoch0.json'),curve_csv=(old/f'{arm}_curve.csv').read_text(),fields=records,checkpoints=checkpoints,
    intervention_summary=read(old/f'{arm}_interventions.json'),independent_review=read(old/f'{arm}_independent_review.json'),
    update_count=read(old/'final_status.json')['models'][arm]['actual_updates'])
 result['gradient_evidence']=read(old/'deeponet_terminal_audit.json')['gradient_diagnostic']
 result['gradient_limitations']='epoch0 run_epoch gradient_norm_max=0 is evaluation, not a backward measurement; no missing initial module gradients invented. Terminal gradients cannot establish exact failure-instant mechanism.'
 write(ART/'v1_scale_audit.json',result)
 print('verified',result['publication_files_verified'],'sealed files;',len(frozen),'checkpoints;',sum(len(m['fields']) for m in result['models'].values()),'complete fields',flush=True)
if __name__=='__main__':main()
