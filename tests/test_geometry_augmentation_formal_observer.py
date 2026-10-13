"""Exact CPU optimizer transition equivalence of the added scalar observer.
The full physics path is separately exercised by the registered real CUDA replay.
"""
import unittest
from unittest.mock import patch
import torch
from geophy_set3d.training import selfsup,geometry_augmentation_formal as observed

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.linspace(.1,.8,8).reshape(1,1,2,2,2))
    def forward(self,b):return self.w.expand(len(b['case_id']),-1,-1,-1,-1).sigmoid()

class Loader:
    batch_size=2
    dataset=range(4)
    def __len__(self):return 2
    def __iter__(self):
        for step in range(2):
            yield {k:dict(case_id=[f'case{2*step}',f'case{2*step+1}'],input_station_indices=torch.tensor([[i],[i]]),view_mask=torch.ones(2,1,dtype=torch.bool)) for i,k in enumerate(('A','B','H'))}

def objective(model,b):
    a=model(b['A']);bb=model(b['B']);parts={k:(a-target).square().mean() for k,target in [('hold',.2),('cross',.3),('observed',.4),('consistency',.5)]}
    loss=parts['hold']+.5*parts['cross']+.1*parts['observed']+.05*parts['consistency']
    return loss,parts,(a,bb),{}

def scores(projections,b):return {k:[dict(relative_l1=.1,relative_l2=.2)] for k in ('heldout','cross','observed')}

class ObserverTransitionTests(unittest.TestCase):
    def test_exact_two_adamw_steps_clip_state_and_outputs(self):
        cfg={'training':{'device':'cpu','gradient_accumulation':1,'gradient_clip_norm':.01}}
        states=[];losses=[]
        for module in (selfsup,observed):
            model=ToyModel();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            with patch.object(module,'objective',objective),patch.object(module,'projection_scores',scores):
                result=module.run_epoch(model,Loader(),cfg,opt)
            states.append((model.state_dict(),opt.state_dict()));losses.append(result['loss'])
        torch.testing.assert_close(states[0][0]['w'],states[1][0]['w'],rtol=0,atol=0)
        self.assertEqual(observed.tree_hash(states[0][1]),observed.tree_hash(states[1][1]));self.assertEqual(losses[0],losses[1])
        self.assertEqual(len(result['gradient_norms']),2)
        self.assertTrue(all(r['preclip']>.01 and r['postclip']<=.010001 for r in result['gradient_norms']))
if __name__=='__main__':unittest.main()
