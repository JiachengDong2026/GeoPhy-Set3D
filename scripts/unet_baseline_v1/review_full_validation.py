"""Independent NumPy full36 Val gate A from saved complete fields and geometry."""
from unet_paths import *
import numpy as np
from numpy_projector import project

def main():
 manifest=read(ART/'full_validation_manifest.json');rows={};stats=[];global_stats={}
 history=[json.loads(s) for s in (ROOT/'unet/pilot/metrics.jsonl').read_text().splitlines()];initial=read(ROOT/'unet/pilot/epoch0.json')
 for epoch in [0,29,30]:
  entry=manifest['epochs'][str(epoch)];scores=[]
  for file in entry['arrays']:
   assert sha(file['path'])==file['sha256']
   with np.load(file['path'],allow_pickle=False) as z:
    for branch in ['A','B']:
     for offset in [0,1]:
      mask=z['H_mask'][offset:offset+1];reference=z['H_input_projection_raw'][offset:offset+1].astype('float64');volume=z[branch+'_volume'][offset:offset+1].astype('float64')
      p=project(volume,*(z['H_'+k][offset:offset+1] for k in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m']),mask)
      saved=z[branch+'_H_projection'][offset:offset+1].astype('float64');np.testing.assert_allclose(p[mask],saved[mask],rtol=3e-4,atol=.02)
      score=float(np.abs(p[mask]-reference[mask]).sum()/max(np.abs(reference[mask]).sum(),1e-6));scores.append(score)
      lg=z[branch+'_logits'][offset:offset+1].astype('float64');np.testing.assert_allclose(volume,1/(1+np.exp(-lg)),rtol=2e-6,atol=1e-7)
      stats.append(dict(epoch=epoch,file=file['path'],branch=branch,offset=offset,H=score,mean=float(volume.mean()),std=float(volume.std()),nearzero=float((volume<1e-10).mean()),saturated=float((np.abs(lg)>10).mean())))
  assert len(scores)==72
  value=float(np.mean(scores));assert np.isclose(value,entry['GPU_logged_H'],rtol=1e-4,atol=1e-6)
  rows[str(epoch)]=value
  fields=[r for r in stats if r['epoch']==epoch];mean=float(np.mean([r['mean'] for r in fields]));std=float(np.sqrt(max(0.,np.mean([r['std']**2+r['mean']**2 for r in fields])-mean**2)))
  logged=initial['val']['output'] if epoch==0 else history[epoch-1]['val']['output']
  assert np.isclose(mean,logged['mean'],rtol=2e-4,atol=1e-10)
  assert np.isclose(std,logged['std'],rtol=1e-3,atol=1e-10)
  global_stats[str(epoch)]=dict(mean=mean,std=std)
 original=read(ART/'unet_pilot.json')['final_checks'];checks={}
 for epoch in [29,30]:
  score=rows[str(epoch)];before=rows['0'];c=dict(val_below_zero=score<=.99,val_initial_improvement=score<=.95*before and before-score>=.01)
  assert all(c[k]==original[str(epoch)][k] for k in c),(epoch,c,original[str(epoch)])
  checks[str(epoch)]=c
 write(ART/'full_validation_independent_review.json',dict(passed=True,full_fields=216,full_Val_case_count=36,independent_H=rows,gate_A_matches=True,checks=checks,field_statistics=stats,aggregate_field_statistics=global_stats,new_updates=0,GT_reads=0))
 print('NumPy full36 Val gate A verified',rows,flush=True)
if __name__=='__main__':main()
