import unittest
import torch
from geophy_set3d.models import GeoLiftGCARes3D,GeoLiftGCSIRes3D
from geophy_set3d.training.utils import permute_views
from test_geolift_gca_res3d import gca_batch


class GCSIModelTests(unittest.TestCase):
    def test_shared_b1_modules_and_initial_parameters(self):
        torch.manual_seed(20261011);b1=GeoLiftGCARes3D()
        torch.manual_seed(20261011);b2=GeoLiftGCSIRes3D()
        for name in ('encoder','lifting','decoder','fusion'):
            a,b=getattr(b1,name),getattr(b2,name)
            self.assertEqual(repr(a),repr(b))
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        self.assertLess(sum(p.numel() for p in b2.parameters()),400000)
        self.assertEqual(sum(p.numel() for p in b2.parameters())-sum(p.numel() for p in b1.parameters()),
                         sum(p.numel() for p in b2.interaction.parameters()))

    def test_b2_prediction_permutation_and_padding_invariance(self):
        torch.set_num_threads(2);torch.manual_seed(25)
        model=GeoLiftGCSIRes3D().eval();batch=gca_batch()
        with torch.no_grad():
            prediction,diag=model(batch,True)
            self.assertEqual(prediction.shape,(1,1,64,64,64))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue((prediction>=0).all() and (prediction<=1).all())
            other=model(permute_views(batch,torch.tensor([2,0,1])))
            torch.testing.assert_close(prediction,other,atol=3e-6,rtol=0)
            padded=dict(batch)
            for key,value in batch.items():
                if key.startswith('input_') or key=='view_mask':
                    padding=torch.full_like(value[:,:1],900) if value.dtype!=torch.bool else torch.zeros_like(value[:,:1])
                    padded[key]=torch.cat((value,padding),1)
            padded['input_projection'][:,-1]=float('nan')
            other,diagnostics=model(padded,True)
            torch.testing.assert_close(prediction,other,atol=3e-6,rtol=0)
            self.assertTrue((diagnostics['attention_weights'][:,-1]==0).all())
            self.assertTrue((diagnostics['interaction_attention'][:,:,-1]==0).all())
            self.assertTrue((diagnostics['interaction_attention'][:,:,:,-1]==0).all())
