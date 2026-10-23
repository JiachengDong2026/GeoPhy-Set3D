"""CPU engineering checks for the independent ray-integral GNOT adaptation."""
import copy
import io
import unittest

import torch

from geophy_set3d.models.baseline_gnot_v1 import (
    NormalizedOperatorAttention, RayGNOTV1,
)


def ray_batch(batch_size=1, views=2, height=2, width=3):
    shape = (batch_size, views, 1, height, width)
    result = {
        'input_projection_raw': torch.rand(shape) * 1000,
        'input_valid_ray': torch.ones(shape, dtype=torch.bool),
        'view_mask': torch.ones(batch_size, views, dtype=torch.bool),
        'input_t_near': torch.full(shape, .2),
        'input_path_length_m': torch.full(shape, .6),
        'input_station_origin_m': torch.tensor([-.2, .3, .3]).expand(batch_size, views, 3).clone(),
        'input_ray_direction': torch.tensor([1., 0., 0.]).expand(batch_size, views, height, width, 3).clone(),
    }
    return result


class RayGNOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261011)

    def small_model(self, **kwargs):
        return RayGNOTV1(width=8, layers=2, heads=2, experts=2,
                         inner_multiplier=2, query_chunk_size=8192, **kwargs)

    def test_linear_statistics_matches_dense_normalized_kernel(self):
        attention = NormalizedOperatorAttention(8, 2, sources=2)
        queries = torch.randn(2, 7, 8)
        branches = [torch.randn(2, n, 8) for n in (9, 5)]
        masks = [torch.rand(2, n) > .6 for n in (9, 5)]
        statistics = [attention.statistics(x, i, mask)
                      for i, (x, mask) in enumerate(zip(branches, masks))]
        actual = attention.apply_statistics(queries, statistics)
        q = attention._heads(attention.query(queries)).softmax(-1)
        expected = q
        for i, (x, mask) in enumerate(zip(branches, masks)):
            k = attention._heads(attention.keys[i](x)).softmax(-1)
            v = attention._heads(attention.values[i](x))
            affinity = q @ k.transpose(-2, -1)
            affinity = affinity.masked_fill(mask[:, None, None, :], 0.)
            expected = expected + (affinity @ v) / affinity.sum(-1, keepdim=True).clamp_min(1.e-12)
        expected = attention.output(expected.transpose(1, 2).reshape(queries.shape))
        torch.testing.assert_close(actual, expected, atol=3.e-7, rtol=3.e-6)

    def test_permutation_padding_nan_empty_and_variable_batch(self):
        model = self.small_model().eval()
        coordinates = torch.rand(23, 3)*.6
        for batch_size, rays in ((1, 1), (2, 13)):
            tokens = torch.randn(batch_size, rays, 203)
            mask = torch.zeros(batch_size, rays, dtype=torch.bool)
            original = model.forward_tokens(tokens, mask, coordinates_m=coordinates)
            order = torch.randperm(rays)
            shuffled = model.forward_tokens(tokens[:, order], mask[:, order], coordinates_m=coordinates)
            padded = torch.cat([tokens, torch.full((batch_size, 3, 203), float('nan'))], 1)
            padded_mask = torch.cat([mask, torch.ones(batch_size, 3, dtype=torch.bool)], 1)
            padded_result = model.forward_tokens(padded, padded_mask, coordinates_m=coordinates)
            torch.testing.assert_close(original, shuffled, atol=2.e-6, rtol=2.e-5)
            torch.testing.assert_close(original, padded_result, atol=2.e-6, rtol=2.e-5)
            empty = model.forward_tokens(padded, torch.ones_like(padded_mask), coordinates_m=coordinates)
            self.assertTrue(torch.isfinite(empty).all())
        bad = torch.zeros(1, 1, 203);bad[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            model.forward_tokens(bad, torch.zeros(1, 1, dtype=torch.bool), coordinates_m=coordinates)

    def test_global_chunk_equivalence_forward_and_backward(self):
        a = self.small_model(checkpoint_chunks=False).train()
        b = copy.deepcopy(a);b.checkpoint_chunks = True
        source_a = torch.randn(2, 17, 203, requires_grad=True)
        source_b = source_a.detach().clone().requires_grad_()
        coords_a = (torch.rand(37, 3)*.6).requires_grad_()
        coords_b = coords_a.detach().clone().requires_grad_()
        excluded = torch.zeros(2, 17, dtype=torch.bool);excluded[1, 9:] = True
        first = a.forward_tokens(source_a, excluded, coordinates_m=coords_a, chunk_size=37)
        second = b.forward_tokens(source_b, excluded, coordinates_m=coords_b, chunk_size=7)
        torch.testing.assert_close(first, second, atol=2.e-6, rtol=2.e-5)
        weights = torch.linspace(.1, 1., 37).reshape(1, 37, 1)
        (first * weights).sum().backward();(second * weights).sum().backward()
        torch.testing.assert_close(source_a.grad, source_b.grad, atol=3.e-6, rtol=3.e-4)
        torch.testing.assert_close(coords_a.grad, coords_b.grad, atol=3.e-6, rtol=3.e-4)
        self.assertGreater(float(source_a.grad[..., 0].abs().sum()), 0.)
        self.assertGreater(float(source_a.grad[..., 1:].abs().sum()), 0.)
        self.assertGreater(float(coords_a.grad.abs().sum()), 0.)
        self.assertEqual(float(source_a.grad[1, 9:].abs().sum()), 0.)
        for (name, p), (name2, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name, name2)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p.grad, q.grad, atol=2.e-5, rtol=3.e-4, msg=name)

    def test_real_adapter_grid_mask_gt_and_checkpoint(self):
        model = self.small_model().eval()
        batch = ray_batch(batch_size=2, views=3)
        batch['view_mask'][1, 1:] = False
        batch['input_projection_raw'][1, 1:] = float('nan')
        batch['input_valid_ray'][0] = False
        with torch.no_grad():
            prediction = model(batch)
        self.assertEqual(prediction.shape, (2, 1, 64, 64, 64))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
        grid = model.query_coordinates_m.reshape(64, 64, 64, 3)
        torch.testing.assert_close(grid[0, 32, 0], torch.tensor([0., 32*.6/63, 0.]))
        checkpoint_file = io.BytesIO();torch.save(model.state_dict(), checkpoint_file)
        checkpoint_file.seek(0)
        restored = self.small_model().eval()
        restored.load_state_dict(torch.load(checkpoint_file, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(prediction, restored(batch), atol=0, rtol=0)
        batch['gt_volume_64'] = torch.empty(0)
        with self.assertRaises(ValueError):
            model(batch)


if __name__ == '__main__':
    unittest.main()
