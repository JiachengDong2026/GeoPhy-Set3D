import unittest
from unittest.mock import patch
import torch
from geophy_set3d.models import GeoLiftOARIMRGRRes3D, GeoLiftOARIMRGRZeroResidualControl
from geophy_set3d.training.utils import load_config, permute_views
from geophy_set3d.training.comparison import assert_fair_mrgr_zero_control
from test_geolift_gca_res3d import gca_batch


class ZeroResidualControlTests(unittest.TestCase):
    def test_exact_parameters_and_same_seed_initialization(self):
        torch.manual_seed(20261011); full=GeoLiftOARIMRGRRes3D()
        torch.manual_seed(20261011); control=GeoLiftOARIMRGRZeroResidualControl()
        self.assertEqual(sum(p.numel() for p in control.parameters()),332070)
        self.assertEqual(list(dict(full.named_parameters())),list(dict(control.named_parameters())))
        for k,v in full.state_dict().items():
            torch.testing.assert_close(v,control.state_dict()[k],atol=0,rtol=0)
        self.assertTrue(all(p.requires_grad for p in control.parameters()))
        assert_fair_mrgr_zero_control(load_config('configs/train_supervised_oari_mrgr_300.yaml'),load_config('configs/train_supervised_oari_mrgr_zero_residual_control_300.yaml'))

    def test_zero_channel_keeps_valid_path_and_never_calls_projector(self):
        torch.set_num_threads(2); batch=gca_batch(); model=GeoLiftOARIMRGRZeroResidualControl()
        with patch('geophy_set3d.models.geolift_oari_mrgr_res3d.project_input',side_effect=AssertionError('Control must not project')):
            s=model(batch,return_stages=True)
            s['C1'].square().mean().backward()
        self.assertTrue((s['residual_input'][:,:,0]==0).all())
        torch.testing.assert_close(s['residual_input'][:,:,1:2],batch['input_valid_ray'].float())
        self.assertTrue(torch.isfinite(s['C0']).all() and torch.isfinite(s['C1']).all())
        torch.testing.assert_close(s['C0'],s['C1'],atol=0,rtol=0)
        self.assertGreater(float(model.residual_adapter[-1].weight.grad.abs().sum()),0)

    def test_control_permutation_padding_safe(self):
        torch.set_num_threads(2); model=GeoLiftOARIMRGRZeroResidualControl().eval(); batch=gca_batch()
        with torch.no_grad():
            model.residual_adapter[-1].weight.normal_(0,.02)
            p=model(batch); q=model(permute_views(batch,torch.tensor([2,0,1])))
            padded=dict(batch)
            for k,v in batch.items():
                if k.startswith('input_') or k=='view_mask':
                    padded[k]=torch.cat((v,torch.zeros_like(v[:,:1])),1)
            padded['input_projection'][:,-1]=float('nan')
            pad=model(padded)
        torch.testing.assert_close(p,q,atol=3e-6,rtol=0)
        torch.testing.assert_close(p,pad,atol=3e-6,rtol=0)
