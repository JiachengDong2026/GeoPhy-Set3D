import unittest
import torch
from geophy_set3d.models import GeoLiftMeanRes3D, GeoLiftGCARes3D
from geophy_set3d.training.utils import permute_views
from model_fixture import model_batch


def gca_batch():
    batch = model_batch(1, 3)
    batch['input_station_azimuth_deg'] = torch.tensor([[10., 80., 150.]])
    return batch


class GCAModelTests(unittest.TestCase):
    def test_shared_modules_are_identical(self):
        torch.manual_seed(20261011)
        b0 = GeoLiftMeanRes3D()
        torch.manual_seed(20261011)
        b1 = GeoLiftGCARes3D()
        for module in ('encoder', 'decoder'):
            a, b = getattr(b0, module), getattr(b1, module)
            self.assertEqual(repr(a), repr(b))
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(type(b0.lifting), type(b1.lifting))
        self.assertEqual(sum(p.numel() for p in b1.parameters()) - sum(p.numel() for p in b0.parameters()),
                         sum(p.numel() for p in b1.fusion.parameters()))

    def test_full_model_shape_range_permutation_padding(self):
        torch.set_num_threads(2)
        torch.manual_seed(6)
        model = GeoLiftGCARes3D().eval()
        batch = gca_batch()
        with torch.no_grad():
            output, diag = model(batch, return_attention=True)
            self.assertEqual(output.shape, (1, 1, 64, 64, 64))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output >= 0).all() and (output <= 1).all())
            self.assertEqual(diag['attention_weights'].shape, (1, 3, 32, 32, 32))
            other = model(permute_views(batch, torch.tensor([2, 0, 1])))
            torch.testing.assert_close(output, other, atol=2e-6, rtol=0)
            padded = dict(batch)
            for key, value in batch.items():
                if key.startswith('input_') or key == 'view_mask':
                    padding = torch.full_like(value[:, :1], 900) if value.dtype != torch.bool else torch.zeros_like(value[:, :1])
                    padded[key] = torch.cat((value, padding), 1)
            padded['input_projection'][:, -1] = float('nan')
            result, diag = model(padded, return_attention=True)
            torch.testing.assert_close(output, result, atol=2e-6, rtol=0)
            self.assertTrue((diag['attention_weights'][:, -1] == 0).all())

    def test_descriptor_uses_frozen_angle_endpoints(self):
        model = GeoLiftGCARes3D()
        batch = gca_batch()
        descriptor = model.geometry_descriptor(batch)
        grid, _ = model.lifting.angular_grid(batch)
        torch.testing.assert_close(descriptor[:, :, :2].flatten(3).transpose(2, 3), grid)
        self.assertAlmostEqual(descriptor[0, 1, 3, 0, 0, 0].item(), 80/180, places=6)
        expected = torch.linalg.vector_norm(model.lifting.voxel_xyz[0]-batch['input_station_origin_m'][0, 0])/6.5
        torch.testing.assert_close(descriptor[0, 0, 2, 0, 0, 0], expected)
