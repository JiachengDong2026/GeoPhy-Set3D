import copy
import unittest
import torch
from model_fixture import model_batch
from geophy_set3d.models.geolift_deepsets_sum import GeoLiftDeepSetsSum,masked_scaled_sum
from geophy_set3d.training.deep_comparison import load_comparison_config,validate_comparison_config
from geophy_set3d.training.selfsup import validate_selfsup_config
from geophy_set3d.training.utils import permute_views


class DeepSetsTests(unittest.TestCase):
    def test_sum_preserves_cardinality_and_excludes_inactive_nan(self):
        f=torch.ones(1,3,2,2,2,2,requires_grad=True)
        vis=torch.ones(1,3,2,2,2,dtype=torch.bool);view=torch.tensor([[True,True,False]])
        with torch.no_grad():f[:,2]=float('nan')
        out=masked_scaled_sum(f,vis,view);self.assertTrue(torch.equal(out,torch.full_like(out,.5)))
        out.sum().backward();self.assertTrue(torch.isfinite(f.grad).all());self.assertEqual(float(f.grad[:,2].sum()),0)
        self.assertTrue(torch.equal(masked_scaled_sum(f,torch.zeros_like(vis),view),torch.zeros_like(out)))
        self.assertTrue(torch.equal(masked_scaled_sum(f[:,:1],vis[:,:1],view[:,:1]),torch.full_like(out,.25)))

    def test_model_permutation_padding_invalid_and_backward(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        m=GeoLiftDeepSetsSum();b=model_batch(1,3);b.pop('gt_volume_64');m.eval()
        with torch.no_grad():
            y=m(b);p=m(permute_views(b,[2,0,1]));self.assertTrue(torch.allclose(y,p,atol=2e-6,rtol=2e-5))
            padded=dict(b)
            for k,v in b.items():
                if k.startswith('input_') or k=='view_mask':
                    extra=v[:,:1].clone()
                    if k=='view_mask':extra.fill_(False)
                    elif extra.is_floating_point():extra.fill_(float('nan'))
                    padded[k]=torch.cat([v,extra],1)
            self.assertTrue(torch.allclose(y,m(padded),atol=2e-6,rtol=2e-5))
        b['view_mask'].zero_();b['input_projection'].fill_(float('nan'));b['input_path_length_m'].fill_(float('nan'))
        output=m(b);self.assertEqual(output.shape,(1,1,64,64,64));self.assertTrue(torch.isfinite(output).all())
        output.mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_independent_policy_preserves_v1_lock(self):
        c=load_comparison_config('configs/train_deepsets_selfsup_v1.yaml')
        with self.assertRaises(ValueError):validate_selfsup_config(c)
        for section,key,value in [('loss','consistency',.1),('model','v_ref',5),('training','max_epochs',301),('experiment','name','selfsup_oari_mrgr_v1')]:
            changed=copy.deepcopy(c);changed[section][key]=value
            with self.assertRaises(ValueError):validate_comparison_config(changed)
