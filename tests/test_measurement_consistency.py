import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from pool_fixture import make_pool
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.measurement_sampling import attach_hold, measurement_dataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.view_sampling import ViewSampler
from geophy_set3d.models import build_model
from geophy_set3d.training.utils import load_config
from geophy_set3d.training.projection_losses import (MaskedProjectionSmoothL1, Y_SCALE,
    compute_observed_consistency, compute_heldout_consistency, physics_weights)


def tiny_batch():
    batch = {}
    for prefix, origin_y in [('input', .2), ('hold', .4)]:
        batch[prefix+'_station_origin_m'] = torch.tensor([[[ -6., origin_y, .3]]], requires_grad=True)
        batch[prefix+'_ray_direction'] = torch.tensor([1., 0., 0.]).expand(1, 1, 2, 2, 3).clone().requires_grad_()
        batch[prefix+'_t_near'] = torch.full((1, 1, 1, 2, 2), 6., requires_grad=True)
        batch[prefix+'_path_length_m'] = torch.full((1, 1, 1, 2, 2), .6, requires_grad=True)
        batch[prefix+'_valid_ray'] = torch.ones(1, 1, 1, 2, 2, dtype=torch.bool)
        batch[prefix+'_projection_raw'] = torch.full((1, 1, 1, 2, 2), 2000., requires_grad=True)
        batch[prefix+'_projection'] = torch.zeros(1, 1, 1, 2, 2)
    batch['view_mask'] = batch['hold_view_mask'] = torch.ones(1, 1, dtype=torch.bool)
    return batch


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.root, _, _ = make_pool()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, split='train'):
        return attach_hold(ProjectionPoolDataset(self.root, split=split, hold_views=0))

    def test_hold_view_never_overlaps_input(self):
        for epoch in range(20):
            dataset = self.dataset(); dataset.set_epoch(epoch)
            legacy = ViewSampler(3, 6, 0)
            for i in range(len(dataset)):
                sample = dataset[i]
                inp, hold = sample['input_station_indices'], sample['hold_station_indices']
                expected, _ = legacy.sample(dataset.station_count, epoch=epoch, sample_index=i, case_id=sample['case_id'])
                np.testing.assert_array_equal(inp.numpy(), expected)
                self.assertEqual(len(hold), 1)
                self.assertFalse(set(inp.tolist()) & set(hold.tolist()))

    def test_hold_view_sampler_worker_invariant(self):
        signatures = []
        for workers in (0, 2, 4):
            dataset = self.dataset()
            loader = DataLoader(dataset, batch_size=2, num_workers=workers,
                                persistent_workers=workers > 0, collate_fn=projection_pool_collate)
            epochs = []
            for epoch in (0, 7):
                dataset.set_epoch(epoch)
                rows = []
                for batch in loader:
                    for i, case in enumerate(batch['case_id']):
                        rows.append((case, batch['input_station_indices'][i][batch['view_mask'][i]].tolist(),
                                     batch['hold_station_indices'][i][batch['hold_view_mask'][i]].tolist()))
                epochs.append(rows)
            signatures.append(epochs)
            del loader
        self.assertEqual(signatures[0], signatures[1]); self.assertEqual(signatures[0], signatures[2])

    def test_fixed_validation_preserves_b2_ids(self):
        config = load_config('configs/train_supervised_b3_phys_300.yaml'); config['dataset']['root'] = str(self.root)
        dataset = measurement_dataset(config, True)
        legacy = ProjectionPoolDataset(self.root, split='val', min_input_views=4, max_input_views=4, hold_views=0)
        for i in range(len(dataset)):
            torch.testing.assert_close(dataset[i]['input_station_indices'], legacy[i]['input_station_indices'])

    def test_projection_loss_ignores_invalid_rays(self):
        helper = MaskedProjectionSmoothL1()
        prediction = torch.tensor([100., float('nan'), 0.], requires_grad=True)
        reference = torch.tensor([0., float('nan'), 0.])
        valid = torch.tensor([True, False, True])
        actual = helper(prediction, reference, valid)
        self.assertAlmostEqual(float(actual), .5*float(helper(prediction[:1], reference[:1], valid[:1])), places=8)
        actual.backward(); self.assertEqual(float(prediction.grad[1]), 0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_projection_loss_all_invalid_safe(self):
        prediction = torch.tensor([float('nan'), 2.], requires_grad=True)
        loss = MaskedProjectionSmoothL1()(prediction, prediction.detach(), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(loss), 0); loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros(2))

    def test_projection_loss_uses_raw_ppm_m_reference(self):
        batch = tiny_batch(); pred = torch.full((1, 1, 8, 8, 8), .03)
        first = compute_heldout_consistency(pred, batch)
        batch['hold_projection'] += 1e6
        torch.testing.assert_close(first, compute_heldout_consistency(pred, batch))
        batch['hold_projection_raw'] = torch.full_like(batch['hold_projection_raw'], Y_SCALE)
        self.assertNotEqual(float(first), float(compute_heldout_consistency(pred, batch)))

    def test_projection_loss_gradient_to_volume(self):
        batch = tiny_batch(); pred = torch.full((1, 1, 8, 8, 8), .03, requires_grad=True)
        for function in (compute_observed_consistency, compute_heldout_consistency):
            gradient = torch.autograd.grad(function(pred, batch), pred)[0]
            self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(float(gradient.norm()), 0)
        self.assertTrue(all(v.grad is None for v in batch.values()))

    def test_projection_loss_gradient_to_model(self):
        batch = tiny_batch(); model = torch.nn.Conv3d(1, 1, 1)
        torch.nn.init.zeros_(model.weight); torch.nn.init.constant_(model.bias, -3.)
        for function in (compute_observed_consistency, compute_heldout_consistency):
            pred = model(torch.ones(1, 1, 8, 8, 8)).sigmoid()
            gradients = torch.autograd.grad(function(pred, batch), list(model.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients))

    def test_hold_projection_geometry_pairing(self):
        batch = tiny_batch(); pred = torch.full((1, 1, 8, 8, 8), .03)
        with patch('geophy_set3d.training.projection_losses.differentiable_project', return_value=torch.zeros(1, 2, 2)) as projector:
            compute_heldout_consistency(pred, batch)
            geometry = projector.call_args.args[1]
            torch.testing.assert_close(geometry['station_origin_m'], batch['hold_station_origin_m'][0])
            self.assertFalse(geometry['station_origin_m'].requires_grad)
            self.assertEqual(projector.call_args.kwargs['samples'], 96)

    def test_padding_does_not_contribute(self):
        batch = tiny_batch(); pred = torch.full((1, 1, 8, 8, 8), .03)
        before = compute_observed_consistency(pred, batch)
        for key in list(batch):
            if key.startswith('input_'):
                batch[key] = torch.cat([batch[key], batch[key]], dim=1)
        batch['view_mask'] = torch.tensor([[True, False]])
        torch.testing.assert_close(before, compute_observed_consistency(pred, batch))

    def test_hold_measurements_do_not_enter_network(self):
        config = load_config('configs/train_supervised_b3_phys_300.yaml')
        model = build_model(config).eval(); batch = projection_pool_collate([self.dataset()[0]])
        with torch.no_grad():
            before = model(batch)
            batch['hold_projection_raw'] = torch.randn(1, 1, 1, 64, 64)*1e8
            batch['hold_station_origin_m'] = torch.randn(1, 1, 3)*1e8
            torch.testing.assert_close(before, model(batch), rtol=0, atol=0)

    def test_pool_b_never_used_in_training(self):
        config = load_config('configs/train_supervised_b3_phys_300.yaml'); config['dataset']['root'] = str(self.root)
        config['train']['geometry_pool'] = 'ood'
        with self.assertRaises(ValueError):
            measurement_dataset(config)

    def test_b3_model_architecture_identical_to_b2(self):
        b2 = load_config('configs/train_supervised_b2_300.yaml'); b3 = load_config('configs/train_supervised_b3_phys_300.yaml')
        self.assertEqual(b2['model'], b3['model'])
        torch.manual_seed(20261011); a = build_model(b2)
        torch.manual_seed(20261011); b = build_model(b3)
        self.assertIs(type(a), type(b))
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_b3_parameter_count_identical_to_b2(self):
        counts = [sum(p.numel() for p in build_model(load_config(path)).parameters()) for path in
                  ('configs/train_supervised_b2_300.yaml', 'configs/train_supervised_b3_phys_300.yaml')]
        self.assertEqual(counts, [319042, 319042])

    def test_physics_warmup_schedule(self):
        p = dict(lambda_obs=.1, lambda_hold=.05)
        self.assertTrue(all(physics_weights(epoch, p) == (0., 0.) for epoch in range(1, 21)))
        self.assertEqual(physics_weights(41, p), (.1, .05))

    def test_physics_lambda_ramp(self):
        p = dict(lambda_obs=.1, lambda_hold=.05)
        self.assertEqual(physics_weights(21, p), (0., 0.)); self.assertEqual(physics_weights(40, p), (.1, .05))
        self.assertAlmostEqual(physics_weights(30, p)[0], .1*9/19)
        self.assertTrue(all(physics_weights(e, p)[0] < physics_weights(e+1, p)[0] for e in range(21, 40)))


if __name__ == '__main__':
    unittest.main()
