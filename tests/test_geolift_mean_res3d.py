import unittest
import torch
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.training.losses import supervised_loss
from model_fixture import model_batch


class GeoLiftTests(unittest.TestCase):
    def test_shape_range_finite_backward(self):
        torch.set_num_threads(2); torch.manual_seed(5)
        model = GeoLiftMeanRes3D()
        batch = model_batch(1, 2)
        output = model(batch)
        self.assertEqual(output.shape, (1, 1, 64, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        loss, _ = supervised_loss(output, batch['gt_volume_64'])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
