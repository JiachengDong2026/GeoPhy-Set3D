import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from geophy_set3d.data.build_utils import digest
from geophy_set3d.physics.projectors import differentiable_project
from geophy_set3d.physics.representation_capacity import (
    assert_projector_unchanged, load_audit_case, measurement_loss,
    measurement_metrics, optimize_volume, select_stations, select_train_cases,
    volume_diagnostics)
from pool_fixture import make_pool
from test_differentiable_projector_resolution import asymmetric_rays


class RepresentationCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.geometry = asymmetric_rays()
        initial = np.full((64,)*3, .002, np.float32)
        target = torch.full((64,)*3, .0028)
        reference = differentiable_project(target, cls.geometry, 96).detach().numpy()
        cls.optimized, cls.result = optimize_volume(initial, cls.geometry, reference, max_steps=5)

    def test_capacity_optimizer_gradient_finite(self):
        self.assertTrue(self.result['all_gradients_finite'])
        self.assertGreater(self.result['maximum_latent_gradient'], 0)
        self.assertTrue(np.isfinite(self.optimized).all())
        self.assertLess(self.result['minimum_observed_fit_relative_l1'],
                        self.result['initial_fit_relative_l1'])

    def test_capacity_volume_range(self):
        self.assertGreaterEqual(self.optimized.min(), 0.)
        self.assertLessEqual(self.optimized.max(), 1.)
        diagnostics = volume_diagnostics(self.optimized, np.full((64,)*3, .002))
        self.assertFalse(diagnostics['measurement_compatible_but_implausible'])
        self.assertGreater(diagnostics['relative_l1_vs_gt64'], 0)

    def test_fit_station_selection_disjoint_holdout(self):
        azimuths = np.random.default_rng(42).permutation(np.linspace(0, 180, 64))
        fit, heldout = select_stations(azimuths)
        fit2, heldout2 = select_stations(azimuths)
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(heldout, heldout2)
        self.assertEqual(len(set(fit)), 16)
        self.assertEqual(len(set(heldout)), 16)
        self.assertFalse(set(fit) & set(heldout))
        self.assertEqual(np.ptp(azimuths[fit]), 180)
        self.assertGreater(np.ptp(azimuths[heldout]), 160)

    def test_valid_ray_metrics_ignore_invalid(self):
        prediction = np.array([3., np.nan, 1e20])
        reference = np.array([1., np.nan, -1e20])
        valid = np.array([True, False, False])
        metrics = measurement_metrics(prediction, reference, valid)
        self.assertEqual(metrics['relative_l1'], 2.)
        self.assertEqual(metrics['relative_l2'], 2.)
        self.assertEqual(metrics['mean_signed_error_ppm_m'], 2.)
        p = torch.tensor(prediction, requires_grad=True)
        loss = measurement_loss(p, torch.tensor(reference), torch.tensor(valid), scale=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(p.grad[1:].abs().sum().item(), 0.)

    def test_saved_reference_projection_used(self):
        temporary, root, rows, _ = make_pool()
        try:
            row = rows[0]
            row['source_gt_sha256'] = digest(row['source_target_path'])
            projection = root/row['train_projection_path']
            gt64 = root/row['gt_volume_64_path']
            (projection.parent/'sample_meta.json').write_text(json.dumps(dict(
                split='train',reference_samples=256,
                hashes={projection.name:digest(projection),gt64.name:digest(gt64)})))
            saved = np.load(projection).copy()
            with patch('geophy_set3d.physics.projectors.ReferenceProjector',
                       side_effect=AssertionError('Reference regeneration forbidden')):
                reference, c64, c128 = load_audit_case(root, row)
            np.testing.assert_array_equal(reference, saved)
            self.assertEqual(float(reference.max()), float(saved.max()))
            self.assertEqual(c64.shape, (64,)*3)
            self.assertEqual(c128.shape, (128,)*3)
            # Reject modified saved reference rather than quietly regenerating it.
            np.save(projection, saved+1)
            with self.assertRaises(AssertionError):
                load_audit_case(root, row)
        finally:
            temporary.cleanup()

    def test_projector_source_unchanged(self):
        assert_projector_unchanged()

    def test_case_selection_train_only_and_evenly_spaced(self):
        rows = [dict(case_id=f'train-{i}',split='train') for i in range(166)]
        rows += [dict(case_id='val-0',split='val'),dict(case_id='test-0',split='test')]
        indices, selected = select_train_cases(rows)
        self.assertEqual(indices, list(range(0,166,15)))
        self.assertTrue(all(row['split']=='train' for row in selected))
        for split in ('val','test'):
            with self.assertRaises(ValueError):
                load_audit_case('/missing', dict(split=split))


if __name__ == '__main__':
    unittest.main()
