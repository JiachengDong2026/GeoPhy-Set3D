"""Synthetic full-grid measurement-only integration; no data files or GPUs."""
import copy
import unittest

import torch

from geophy_set3d.models.baseline_deepsets_v1 import RayDeepSetsV1
from geophy_set3d.models.baseline_gnot_v1 import RayGNOTV1
from geophy_set3d.models.baseline_senseiver_v1 import RaySenseiverV1
from geophy_set3d.training.selfsup import (
    FROZEN_LOSS, objective, projection_scores, selection_metric,
)


def synthetic_dual():
    """Disjoint A/B/H measurements, only four valid rays per 64² raster.

    Scalars are synthetic measurements, not samples from a target 3-D field.
    There is no GT field, loss mock, projector mock, or reduced output grid.
    """
    result = {}
    for index, label in enumerate(('A', 'B', 'H')):
        valid = torch.zeros(1, 1, 1, 64, 64, dtype=torch.bool)
        valid[:, :, :, :2, :2] = True
        raw = torch.zeros(1, 1, 1, 64, 64)
        raw[:, :, :, :2, :2] = torch.tensor([[300., 600.], [900., 1200.]]) * (1 + .1 * index)
        result[label] = {
            'case_id': ['synthetic_case'],
            'input_station_indices': torch.tensor([[index]]),
            'input_projection_raw': raw,
            'input_valid_ray': valid,
            'view_mask': torch.ones(1, 1, dtype=torch.bool),
            'input_station_origin_m': torch.tensor([[[-.1, .3, .3]]]),
            'input_ray_direction': torch.tensor([1., 0., 0.]).expand(1, 1, 64, 64, 3).clone(),
            'input_t_near': torch.full((1, 1, 1, 64, 64), .1),
            'input_path_length_m': torch.full((1, 1, 1, 64, 64), .6),
        }
    return result


class BaselineMeasurementContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261011)

    def test_full_objective_backward_and_heldout_isolation(self):
        factories = (
            ('deepsets', lambda: RayDeepSetsV1(hidden_dim=16)),
            ('senseiver', RaySenseiverV1),
            # Narrow CPU fixture retains heterogeneous operator attention,
            # global self operator and geometry-gated two-expert mixtures.
            ('gnot', lambda: RayGNOTV1(width=8, layers=1, heads=2,
                                     experts=2, inner_multiplier=2,
                                     query_chunk_size=16384)),
        )
        for name, factory in factories:
            with self.subTest(model=name):
                model = factory().train()
                batch = synthetic_dual()
                batch['H']['input_projection_raw'].requires_grad_()
                batch['H']['input_ray_direction'].requires_grad_()
                seen = []

                def check_input(module, args):
                    stations = args[0]['input_station_indices'].tolist()
                    self.assertNotEqual(stations, [[2]])
                    seen.append(stations)

                hook = model.register_forward_pre_hook(check_input)
                loss, parts, volumes, projections = objective(model, batch)
                hook.remove()
                self.assertEqual(seen, [[[0]], [[1]]])
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(loss, sum(FROZEN_LOSS[k] * v for k, v in parts.items()))
                for volume in volumes:
                    self.assertEqual(tuple(volume.shape), (1, 1, 64, 64, 64))
                    self.assertEqual(volume.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(volume).all())
                    self.assertTrue(((volume >= 0) & (volume <= 1)).all())
                loss.backward()
                grads = [p.grad for p in model.parameters()]
                self.assertTrue(all(g is not None for g in grads))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
                # H remains a detached measurement target/geometry only.
                self.assertIsNone(batch['H']['input_projection_raw'].grad)
                self.assertIsNone(batch['H']['input_ray_direction'].grad)
                with torch.no_grad():
                    scores = projection_scores(projections, batch)
                validation = {'measurement': {
                    key: {metric: sum(row[metric] for row in rows) / len(rows)
                          for metric in ('relative_l1', 'relative_l2')}
                    for key, rows in scores.items()}}
                self.assertEqual(selection_metric(validation),
                                 validation['measurement']['heldout']['relative_l1'])
                self.assertTrue(torch.isfinite(torch.tensor(selection_metric(validation))))

    def test_gt_and_overlap_rejected_before_network(self):
        class NoCall(torch.nn.Module):
            def forward(self, batch):
                raise AssertionError('Forbidden batch reached network')
        batch = synthetic_dual()
        invalid = copy.deepcopy(batch)
        invalid['H']['gt_volume'] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, 'ground truth'):
            objective(NoCall(), invalid)
        invalid = copy.deepcopy(batch)
        invalid['H']['input_station_indices'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'pairwise disjoint'):
            objective(NoCall(), invalid)
        invalid = copy.deepcopy(batch)
        invalid['B']['case_id'] = ['another_case']
        with self.assertRaisesRegex(ValueError, 'same plume'):
            objective(NoCall(), invalid)

    def test_selector_ignores_nonselection_diagnostics(self):
        first = {'measurement': {'heldout': {'relative_l1': .25}},
                 'loss': {'total': 1.}, 'volume_diagnostic': {'relative_l1': .01}}
        second = copy.deepcopy(first)
        second['loss']['total'] = 1e9
        second['volume_diagnostic']['relative_l1'] = 1000.
        self.assertEqual(selection_metric(first), .25)
        self.assertEqual(selection_metric(second), .25)


if __name__ == '__main__':
    unittest.main()
