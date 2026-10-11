import unittest
import torch
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.metrics import VolumeMetrics


class SupervisedLossTests(unittest.TestCase):
    def test_hand_computed_loss_and_safe_floor(self):
        p = torch.tensor([[[1., 2.]], [[0., 1.]]], requires_grad=True)
        g = torch.tensor([[[0., 2.]], [[0., 0.]]])
        total, parts = supervised_loss(p, g)
        self.assertAlmostEqual(float(parts['smooth_l1'].detach()), .25)
        self.assertAlmostEqual(float(parts['relative_l1'].detach()), (0.5+1e6)/2)
        self.assertTrue(torch.isfinite(total)); total.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        zero, _ = supervised_loss(g, g); self.assertEqual(float(zero), 0)

    def test_metrics_mean_versus_pooled(self):
        p = torch.tensor([[[2., 0.]], [[0., 0.]]])
        g = torch.tensor([[[1., 0.]], [[3., 0.]]])
        metrics = VolumeMetrics(); metrics.update(p, g)
        result = metrics.compute()
        self.assertEqual(result['mean_per_case']['relative_l1'], 1)
        self.assertEqual(result['pooled']['relative_l1'], 1)
        self.assertAlmostEqual(result['pooled']['rmse'], (10/4)**.5)
        metrics2 = VolumeMetrics(); metrics2.update(torch.zeros_like(g), g)
        self.assertEqual(metrics2.compute()['pooled']['relative_l2'], 1)
