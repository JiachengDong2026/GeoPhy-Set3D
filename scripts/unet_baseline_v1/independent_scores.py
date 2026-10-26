"""Recompute intervention H scores from NumPy ray integrals, then replay response gates."""
import sys
from unet_paths import *
import numpy as np
from numpy_projector import project

def main(arm):
 out=[];gates={}
 for epoch in [0,10,20,29,30]:
  d=read(ROOT/arm/f'pilot/diagnostic_epoch{epoch}/diagnostics.json');rows=[]
  for entry in d['arrays']:
   split,index=Path(entry['path']).stem.rsplit('_',1);index=int(index)
   with np.load(entry['path'],allow_pickle=False) as z:
    for branch in ['A','B']:
     scores={};volumes={}
     for mode in ['normal','zero','wrong_case']:
      volume=z[f'{branch}_{mode}_volume'].astype('float64');volumes[mode]=volume
      mask=z[branch+'_H_mask'];reference=z[branch+'_H_reference'].astype('float64')
      prediction=project(volume,*(z[branch+'_H_'+k] for k in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m']),mask)
      scores[mode]=float(np.abs(prediction[mask]-reference[mask]).sum()/max(np.abs(reference[mask]).sum(),1e-6))
     difference=np.abs(volumes['normal']-volumes['zero'])
     rows.append(dict(epoch=epoch,split=split,index=index,branch=branch,normal_H=scores['normal'],zero_H=scores['zero'],wrong_H=scores['wrong_case'],
      relative=float(difference.sum()/max(np.abs(volumes['normal']).sum(),1e-30)),absolute=float(difference.mean()),wrong_increase=scores['wrong_case']-scores['normal'],zero_benefit=scores['zero']-scores['normal']))
  def cases(split):
   answer=[]
   for index in sorted({r['index'] for r in rows if r['split']==split}):
    pair=[r for r in rows if r['split']==split and r['index']==index]
    answer.append({k:float(np.mean([r[k] for r in pair])) for k in ['relative','absolute','wrong_increase','zero_benefit']})
   return answer
  tr,va=cases('train'),cases('val')
  checks=dict(train_zero_2of3=sum(r['relative']>=1e-3 and r['absolute']>=1e-8 for r in tr)>=2,
   train_wrong_2of3=sum(r['wrong_increase']>=1e-6 for r in tr)>=2,
   val_zero_response_2of4=sum(r['relative']>=1e-3 and r['absolute']>=1e-8 for r in va)>=2,
   val_zero_benefit_2of4=sum(r['zero_benefit']>=.001 for r in va)>=2)
  original=read(ROOT/arm/f'pilot/diagnostic_epoch{epoch}/gates.json')
  assert all(checks[k]==original[k] for k in checks),(epoch,checks,original)
  gates[str(epoch)]=dict(checks=checks,cases=dict(train=tr,val=va));out.extend(rows)
 write(ART/f'{arm}_numpy_projected_interventions.json',dict(passed=True,all_intervention_gates_match=True,full_fields=210,model_forwards=0,new_updates=0,GT_reads=0,epochs=gates,rows=out))
 print(arm,'independent NumPy projection intervention gates match all epochs',flush=True)
if __name__=='__main__':main(sys.argv[1])
