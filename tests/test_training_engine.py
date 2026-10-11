"""Exercise partial accumulation groups and checkpoint/RNG replay independently of B0."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from geophy_set3d.training.engine import run_epoch
from geophy_set3d.training.utils import atomic_checkpoint, rng_state, restore_rng


class ToyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.2))

    def forward(self, batch):
        return self.weight*batch['x']


class TrainingEngineTests(unittest.TestCase):
    def test_accumulation_partial_group_matches_large_batches(self):
        samples=[dict(x=torch.tensor([[[[v]]]]),gt_volume_64=torch.tensor([[[[v*.4]]]]),case_id=str(i))
                 for i,v in enumerate((.3,.6,.9,.5,.8))]
        config=dict(training=dict(device='cpu',gradient_accumulation=2,gradient_clip_norm=1e6),
                    loss=dict(smooth_l1=1.,relative_l1=.1,relative_epsilon=1e-6))
        a,b=ToyRegressor(),ToyRegressor()
        oa,ob=torch.optim.SGD(a.parameters(),lr=.01),torch.optim.SGD(b.parameters(),lr=.01)
        run_epoch(a,DataLoader(samples,batch_size=2),config,oa)
        other=copy.deepcopy(config); other['training']['gradient_accumulation']=1
        run_epoch(b,DataLoader(samples,batch_size=4),other,ob)
        torch.testing.assert_close(a.weight,b.weight,rtol=0,atol=1e-7)

    def test_checkpoint_and_rng_restore(self):
        generator=torch.Generator().manual_seed(72)
        state=rng_state(generator)
        expected=(random.random(),np.random.rand(),torch.rand(2),torch.randperm(7,generator=generator))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'; atomic_checkpoint(path,dict(rng=state,epoch=2))
            loaded=torch.load(path,weights_only=False)
            restore_rng(loaded['rng'],generator)
            actual=(random.random(),np.random.rand(),torch.rand(2),torch.randperm(7,generator=generator))
            self.assertEqual(loaded['epoch'],2); self.assertEqual(actual[:2],expected[:2])
            torch.testing.assert_close(actual[2],expected[2]); torch.testing.assert_close(actual[3],expected[3])
            self.assertFalse(path.with_suffix('.tmp').exists())
