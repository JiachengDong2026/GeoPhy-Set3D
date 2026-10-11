import unittest
import torch
from geophy_set3d.models import GeoLiftGCSIRes3D, GeoLiftOARIRes3D, build_model
from geophy_set3d.training.utils import permute_views, load_config
from geophy_set3d.training.comparison import assert_fair_b2_oari
from test_geolift_gca_res3d import gca_batch


class OARIModelTests(unittest.TestCase):
    def test_shared_b2_initial_parameters_and_parameter_count(self):
        torch.manual_seed(20261011); b2 = GeoLiftGCSIRes3D()
        torch.manual_seed(20261011); oari = GeoLiftOARIRes3D()
        for name in ('encoder', 'lifting', 'decoder', 'fusion'):
            a, b = getattr(b2, name), getattr(oari, name)
            self.assertEqual(repr(a), repr(b))
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        for name in ('norm1', 'norm2', 'qkv', 'attention_output', 'ffn', 'geometry_embedding'):
            for key, value in getattr(b2.interaction, name).state_dict().items():
                torch.testing.assert_close(value, getattr(oari.interaction, name).state_dict()[key], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in b2.parameters()), 319042)
        self.assertLess(sum(p.numel() for p in oari.parameters()), 340000)

    def test_oari_prediction_permutation_invariance(self):
        torch.set_num_threads(2); torch.manual_seed(25)
        model = GeoLiftOARIRes3D().eval(); batch = gca_batch()
        with torch.no_grad():
            p, diag = model(batch, True)
            other = model(permute_views(batch, torch.tensor([2, 0, 1])))
        self.assertEqual(p.shape, (1, 1, 64, 64, 64))
        self.assertTrue(torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        torch.testing.assert_close(p, other, atol=3e-6, rtol=0)

    def test_oari_model_padding_invariance(self):
        torch.set_num_threads(2); torch.manual_seed(25)
        model = GeoLiftOARIRes3D().eval(); batch = gca_batch(); padded = dict(batch)
        for key, value in batch.items():
            if key.startswith('input_') or key == 'view_mask':
                pad = torch.zeros_like(value[:, :1])
                padded[key] = torch.cat((value, pad), 1)
        padded['input_projection'][:, -1] = float('nan')
        with torch.no_grad():
            p, a = model(batch, True); other, b = model(padded, True)
        torch.testing.assert_close(p, other, atol=3e-6, rtol=0)
        torch.testing.assert_close(a['observability_descriptor'], b['observability_descriptor'], atol=0, rtol=0)
        self.assertTrue((b['interaction_attention'][:, :, -1] == 0).all())
        self.assertTrue((b['interaction_attention'][:, :, :, -1] == 0).all())
        self.assertTrue((b['attention_weights'][:, -1] == 0).all())

    def test_oari_config_uses_frozen_supervised_protocol(self):
        b2 = load_config('configs/train_supervised_b2_300.yaml')
        oari = load_config('configs/train_supervised_oari_300.yaml')
        assert_fair_b2_oari(b2, oari)
        self.assertIsInstance(build_model(oari), GeoLiftOARIRes3D)

    def test_relation_shortcut_is_evaluation_only(self):
        model = GeoLiftOARIRes3D()
        with self.assertRaises(ValueError):
            model(gca_batch(), relation_permutation=torch.tensor([2, 0, 1]))
