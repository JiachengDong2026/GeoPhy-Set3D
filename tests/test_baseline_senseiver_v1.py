"""CPU synthetic tests: no GT, data files, GPUs, or scientific optimisation."""
import copy
import io
import unittest

import torch

from geophy_set3d.models.baseline_senseiver_v1 import RaySenseiverV1
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS, ray_tokens


def synthetic_batch(batch_size=1, views=3):
    height, width = 2, 3
    return {
        'input_projection_raw': torch.linspace(0, 1000, batch_size * views * height * width).reshape(batch_size, views, 1, height, width),
        'input_valid_ray': torch.ones(batch_size, views, 1, height, width, dtype=torch.bool),
        'view_mask': torch.ones(batch_size, views, dtype=torch.bool),
        'input_station_origin_m': torch.tensor([-.1, .3, .3]).expand(batch_size, views, 3).clone(),
        'input_ray_direction': torch.tensor([1., 0., 0.]).expand(batch_size, views, height, width, 3).clone(),
        'input_t_near': torch.full((batch_size, views, 1, height, width), .1),
        'input_path_length_m': torch.full((batch_size, views, 1, height, width), .6),
    }


class BaselineSenseiverV1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261011)

    def test_original_state_identity_and_complete_output(self):
        original = RaySenseiverSS().eval()
        torch.manual_seed(20261011)
        model = RaySenseiverV1().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 254913)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        batch = synthetic_batch()
        with torch.no_grad():
            expected = original(batch)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = model(batch)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(actual.shape), (1, 1, 64, 64, 64))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_chunk_checkpoint_value_and_all_parameter_gradients(self):
        a = RaySenseiverV1(query_chunk_size=129, decoder_checkpoint=False)
        b = copy.deepcopy(a)
        batch = synthetic_batch(batch_size=2, views=2)
        batch['view_mask'][1, 1] = False
        za, zb = a.encode(batch), b.encode(batch)
        queries = a.query_encoding[:129].clone().requires_grad_()
        other_queries = queries.detach().clone().requires_grad_()
        x = a.decode_queries(za, queries, 129, False)
        y = b.decode_queries(zb, other_queries, 19, True)
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        weights = torch.linspace(.1, 1, 129).reshape(1, -1, 1)
        (x * weights).sum().backward()
        (y * weights).sum().backward()
        torch.testing.assert_close(queries.grad, other_queries.grad, atol=3e-6, rtol=3e-4)
        for (name, p), (other_name, other) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p.grad, other.grad, atol=5e-5, rtol=4e-4, msg=name)
        for group in (a.encoder.layer_1, a.encoder.layer_n, a.decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()), 0)
        self.assertGreater(float(a.encoder.latent.grad.abs().sum()), 0)

    def test_variable_rays_batch_empty_mask_and_valid_zero(self):
        model = RaySenseiverV1(query_chunk_size=32768).eval()
        batch = synthetic_batch(batch_size=2, views=3)
        batch['view_mask'][1].zero_()
        batch['input_valid_ray'][0, 1, :, 0].zero_()
        tokens, excluded, counts = ray_tokens(batch)
        self.assertEqual(counts.tolist(), [15, 0])
        self.assertEqual(tokens.shape, (2, 15, 203))
        self.assertFalse(excluded[1, 0])
        self.assertTrue(excluded[1, 1:].all())
        # A valid zero measurement is retained, rather than dropped as padding.
        self.assertEqual(float(tokens[0, 0, 0]), 0.)
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output.shape), (2, 1, 64, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        smaller = synthetic_batch(1, 1)
        with torch.no_grad():
            self.assertEqual(model.encode(smaller).shape, (1, 64, 64))

    def test_observation_permutation_and_masked_content(self):
        model = RaySenseiverV1().eval()
        batch = synthetic_batch(2, 3)
        batch['view_mask'][1, 2] = False
        perm = torch.tensor([2, 0, 1])
        shuffled = {key: value[:, perm] for key, value in batch.items()}
        with torch.no_grad():
            a, b = model.encode(batch), model.encode(shuffled)
        torch.testing.assert_close(a, b, atol=4e-6, rtol=3e-5)
        changed = copy.deepcopy(batch)
        changed['input_projection_raw'][1, 2] = 1e20
        with torch.no_grad():
            torch.testing.assert_close(a, model.encode(changed), atol=0, rtol=0)

    def test_checkpoint_roundtrip_and_gt_boundary(self):
        model = RaySenseiverV1().eval()
        archive = io.BytesIO()
        torch.save(model.state_dict(), archive)
        archive.seek(0)
        restored = RaySenseiverV1().eval()
        restored.load_state_dict(torch.load(archive, weights_only=True))
        batch = synthetic_batch()
        with torch.no_grad():
            a = model.decode_queries(model.encode(batch), model.query_encoding[:73])
            b = restored.decode_queries(restored.encode(batch), restored.query_encoding[:73])
        self.assertTrue(torch.equal(a, b))
        batch['gt_volume'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            model(batch)

    def test_invalid_memory_controls(self):
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                RaySenseiverV1(query_chunk_size=size)
        with self.assertRaises(ValueError):
            RaySenseiverV1(decoder_checkpoint='yes')


if __name__ == '__main__':
    unittest.main()
