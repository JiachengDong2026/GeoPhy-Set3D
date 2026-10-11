import unittest
import math
import torch
from geophy_set3d.models.lifting import GeometryLifting
from geophy_set3d.models.fusion import masked_mean
from model_fixture import model_batch


class GeometryLiftingTests(unittest.TestCase):
    def test_xyz_endpoint_grid_and_tdlas_angles(self):
        lift = GeometryLifting(3)
        xyz = lift.voxel_xyz.reshape(3, 3, 3, 3)
        torch.testing.assert_close(xyz[2, 0, 1], torch.tensor([.6, 0., .3]))
        batch = model_batch(1, 1)
        alpha, beta = math.radians(2.), math.radians(-1.)
        d = torch.tensor([math.cos(beta)*math.cos(alpha), math.cos(beta)*math.sin(alpha), math.sin(beta)])
        batch['input_station_origin_m'][0, 0] = xyz[1, 1, 1] - 6*d
        grid, visible = lift.angular_grid(batch)
        torch.testing.assert_close(grid[0, 0, 13], torch.tensor([.4, -.25]), atol=2e-6, rtol=0)
        self.assertTrue(visible[0, 0, 13])

    def test_angular_sampling_order_and_gradient(self):
        batch = model_batch(1, 1)
        lift = GeometryLifting(3)
        horizontal = torch.linspace(-1, 1, 64)[None, :].expand(64, -1)
        vertical = torch.linspace(-1, 1, 64)[:, None].expand(-1, 64)
        features = torch.stack([horizontal, vertical])[None, None].requires_grad_()
        output, visibility = lift(features, batch)
        grid, _ = lift.angular_grid(batch)
        torch.testing.assert_close(output.flatten(3).transpose(2, 3)[0, 0], grid[0, 0], atol=2e-6, rtol=0)
        self.assertEqual(output.shape, (1, 1, 2, 3, 3, 3))
        self.assertTrue(visibility.all()); self.assertTrue((grid.abs() <= 1).all())
        output.sum().backward(); self.assertTrue(torch.isfinite(features.grad).all())

    def test_behind_out_of_scan_and_masked_mean(self):
        lift = GeometryLifting(3); batch = model_batch(1, 3)
        batch['view_mask'][0, 1] = False
        batch['input_station_origin_m'][0, 2] = torch.tensor([6., .3, .3])
        f = torch.ones(1, 3, 2, 64, 64)
        values, visibility = lift(f, batch)
        self.assertFalse(visibility[:, 1:].any())
        torch.testing.assert_close(masked_mean(values, visibility), torch.ones(1, 2, 3, 3, 3))
        batch['input_station_origin_m'][0, 0, 1] = 100
        values, visibility = lift(f, batch)
        self.assertFalse(visibility.any())
        self.assertEqual(torch.count_nonzero(masked_mean(values, visibility)).item(), 0)
        values.fill_(float('nan'))
        self.assertTrue(torch.isfinite(masked_mean(values, visibility)).all())
