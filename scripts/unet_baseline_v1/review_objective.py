"""Independent FP64 constant-field objective and analytic directional gradients."""
import sys
from unet_paths import *
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch
from geophy_set3d.training.classical_operators import load_config
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.gt_read_guard import GTReadGuard

def discrepancy(c,b):
 valid=b['input_valid_ray'].numpy();length=b['input_path_length_m'].numpy()[valid].astype('float64')
 target=b['input_projection_raw'].numpy()[valid].astype('float64')/32284.894580078173
 a=1/16.043;air=1/28.965;chi=a*c/(air+(a-air)*c);prime=a*air/(air+(a-air)*c)**2
 factor=1e6*length/32284.894580078173;d=factor*chi-target;absolute=np.abs(d)
 value=float(np.mean(np.where(absolute<.01,.5*d*d/.01,absolute-.005))+.1*absolute.sum()/max(np.abs(target).sum(),1e-6))
 derivative=float(np.mean(np.clip(d/.01,-1,1)*factor*prime)+.1*np.sum(np.sign(d)*factor*prime)/max(np.abs(target).sum(),1e-6))
 return value,derivative

def main():
 torch.set_num_threads(2);guard=GTReadGuard(DATA,ROOT);original=read(ART/'objective_direction_audit.json');result=[]
 with guard:
  ds=make_dataset(load_config('fno'))
  for row in original['rows']:
   a,b=row['A'],row['B'];parts={};grads={};samples=[ds[i] for i in row['indices']]
   for name,targets in [('hold',('H','H')),('cross',('B','A')),('observed',('A','B'))]:
    pair=[]
    for c,target in zip((a,b),targets):pair.append(np.mean([discrepancy(c,s[target]) for s in samples],axis=0))
    parts[name]=float(.5*(pair[0][0]+pair[1][0]));grads[name]=[float(.5*pair[0][1]),float(.5*pair[1][1])]
   difference=abs(a-b);den=.5*(abs(a)+abs(b))+1e-6;sign=np.sign(a-b)
   parts['consistency']=difference/den;grads['consistency']=[float(sign/den-.5*difference/den**2),float(-sign/den-.5*difference/den**2)]
   weights=dict(hold=1.,cross=.5,observed=.1,consistency=.05)
   value=sum(weights[k]*parts[k] for k in parts);gradient=[sum(weights[k]*grads[k][i] for k in grads) for i in range(2)]
   assert np.isclose(value,row['total'],rtol=2e-5,atol=1e-7),(value,row['total'])
   for k in grads:np.testing.assert_allclose(grads[k],row['dL_dc_AB'][k],rtol=2e-4,atol=.005)
   np.testing.assert_allclose(gradient,row['dL_dc_AB']['total'],rtol=2e-4,atol=.005)
   result.append(dict(indices=row['indices'],A=a,B=b,independent_total=value,independent_parts=parts,independent_total_gradient=gradient,independent_branch_gradients=grads))
  access=guard.summary()
 write(ART/'objective_independent_review.json',dict(passed=True,rows=result,method='NumPy FP64 closed-form constant physical projection, Huber/relative-L1/consistency and analytic gradients; no torch objective called',access=access,new_updates=0))
 print('Independent constant objective/gradient check passed',len(result),'conditions')
if __name__=='__main__':main()
