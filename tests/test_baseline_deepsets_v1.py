"""CPU engineering checks; synthetic measurements only, no dataset/GT access."""
import copy
import io
import unittest

import torch

from geophy_set3d.models.baseline_deepsets_v1 import RayDeepSetsV1
from geophy_set3d.training.projection_losses import compute_observed_consistency


def synthetic_batch(batch_size=2, views=3):
    batch = {
        'input_station_origin_m': torch.tensor([-6., .2, .3]).expand(batch_size, views, 3).clone(),
        'input_ray_direction': torch.tensor([1., 0., 0.]).expand(batch_size, views, 2, 2, 3).clone(),
        'input_t_near': torch.full((batch_size, views, 1, 2, 2), 6.),
        'input_path_length_m': torch.full((batch_size, views, 1, 2, 2), .6),
        'input_valid_ray': torch.ones(batch_size, views, 1, 2, 2, dtype=torch.bool),
        'input_projection_raw': torch.linspace(100., 30000., batch_size * views * 4).reshape(batch_size, views, 1, 2, 2),
        'view_mask': torch.ones(batch_size, views, dtype=torch.bool),
    }
    return batch


class DeepSetsV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261011)
        self.model = RayDeepSetsV1(hidden_dim=32, query_chunk_size=7)
        self.queries = torch.rand(19, 3) * .6

    def predict(self, batch):
        return self.model.decode_queries(self.model.encode(batch), self.queries)

    def test_permutation_and_masked_nan(self):
        self.model.eval()
        batch = synthetic_batch()
        expected = self.predict(batch)
        permuted = {key: value[:, [2, 0, 1]] for key, value in batch.items()}
        torch.testing.assert_close(self.predict(permuted), expected, atol=1e-7, rtol=1e-6)
        # Permute individual rays too; all synthetic view origins coincide.
        ray_permutation = torch.randperm(12)
        shuffled = copy.deepcopy(batch)
        for key in ('input_t_near', 'input_path_length_m', 'input_valid_ray', 'input_projection_raw'):
            shuffled[key] = batch[key].reshape(2, 12)[:, ray_permutation].reshape_as(batch[key])
        shuffled['input_ray_direction'] = batch['input_ray_direction'].reshape(2, 12, 3)[:, ray_permutation].reshape_as(batch['input_ray_direction'])
        torch.testing.assert_close(self.predict(shuffled), expected, atol=1e-7, rtol=1e-6)
        padded = {key: torch.cat((value, value[:, :1]), dim=1) for key, value in batch.items()}
        padded['view_mask'][:, -1] = False
        for key, value in padded.items():
            if value.dtype.is_floating_point:
                value[:, -1] = float('nan')
        torch.testing.assert_close(self.predict(padded), expected, atol=0, rtol=0)
        masked = copy.deepcopy(batch)
        masked['input_valid_ray'][:, 0, :, 0, 0] = False
        reference = self.predict(masked)
        masked['input_projection_raw'][:, 0, :, 0, 0] = float('nan')
        masked['input_ray_direction'][:, 0, 0, 0] = float('nan')
        torch.testing.assert_close(self.predict(masked), reference, atol=0, rtol=0)

    def test_measurement_dependence_empty_and_batch(self):
        batch = synthetic_batch()
        baseline = self.predict(batch)
        changed = copy.deepcopy(batch)
        changed['input_projection_raw'] *= 4
        self.assertGreater(float((self.predict(changed) - baseline).detach().abs().max()), 1e-7)
        fewer = copy.deepcopy(batch)
        fewer['view_mask'][:, 1:] = False
        self.assertTrue(torch.isfinite(self.predict(fewer)).all())
        empty = copy.deepcopy(batch)
        empty['view_mask'][:] = False
        for key, value in empty.items():
            if value.dtype.is_floating_point:
                value[:] = float('nan')
        torch.testing.assert_close(self.model.encode(empty), torch.zeros(2, 32), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(self.predict(empty)).all())
        for i in range(2):
            single = {key: value[i:i+1] for key, value in batch.items()}
            torch.testing.assert_close(self.predict(single), baseline[i:i+1], atol=1e-7, rtol=1e-6)

    def test_gradient_checkpoint_equivalence(self):
        batch = synthetic_batch()
        latent = self.model.encode(batch)
        actual = self.model.decode_queries(latent, self.queries, checkpoint_chunks=True)
        plain = self.model.decode_queries(latent, self.queries, checkpoint_chunks=False)
        torch.testing.assert_close(actual, plain, atol=0, rtol=0)
        actual.square().mean().backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(self.model.phi[0].weight.grad.norm()), 0.)
        self.assertGreater(float(self.model.decoder[-1].weight.grad.norm()), 0.)

    def test_checkpoint_roundtrip_and_gt_guard(self):
        batch = synthetic_batch()
        expected = self.predict(batch)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = RayDeepSetsV1(hidden_dim=32, query_chunk_size=7)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        actual = restored.decode_queries(restored.encode(batch), self.queries)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        batch['gt_volume_64'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            self.model.encode(batch)

    def test_full_output_and_physics_backward(self):
        self.model.query_chunk_size = 16384
        batch = synthetic_batch(batch_size=1, views=1)
        output = self.model(batch)
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64, 64))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        loss = compute_observed_consistency(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(self.model.phi[0].weight.grad.norm()), 0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_validation_and_default_parameters(self):
        model = RayDeepSetsV1()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 104577)
        with self.assertRaises(ValueError):
            RayDeepSetsV1(query_chunk_size=0)
        with self.assertRaises(ValueError):
            RayDeepSetsV1(output_bias_logit=float('nan'))
        with self.assertRaises(ValueError):
            self.model.decode_queries(torch.zeros(1, 32), torch.zeros(4, 96))


if __name__ == '__main__':
    unittest.main()
