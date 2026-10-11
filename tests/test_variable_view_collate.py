import unittest
import torch

from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from pool_fixture import make_pool


class CollateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary, cls.root, _, _ = make_pool()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_variable_input_and_hold_padding(self):
        a = ProjectionPoolDataset(self.root, min_input_views=3, max_input_views=3, hold_views=1)[0]
        b = ProjectionPoolDataset(self.root, min_input_views=6, max_input_views=6, hold_views=2)[1]
        # Real ZERO measurement views remain actual views.
        a['input_projection'].zero_()
        batch = projection_pool_collate([a, b])
        self.assertEqual(batch['input_projection'].shape, (2, 6, 1, 64, 64))
        self.assertEqual(batch['gt_volume_64'].shape, (2, 1, 64, 64, 64))
        self.assertEqual(batch['hold_projection'].shape, (2, 2, 1, 64, 64))
        torch.testing.assert_close(batch['view_mask'], torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1]], dtype=torch.bool))
        torch.testing.assert_close(batch['hold_view_mask'], torch.tensor([[1,0],[1,1]], dtype=torch.bool))
        for prefix, count in [('input', 3), ('hold', 1)]:
            for key in a:
                if key.startswith(prefix + '_'):
                    torch.testing.assert_close(batch[key][0, :count], a[key])
                    expected = -1 if key.endswith('station_indices') else 0
                    self.assertTrue(torch.all(batch[key][0, count:] == expected))
        self.assertTrue(batch['view_mask'][0, :3].all())
        self.assertGreater(batch['input_valid_ray'][0, :3].sum().item(), 0)
        self.assertFalse(batch['input_valid_ray'][0, 3:].any())
        self.assertIsNone(batch['condition_meta'][0]['source_mode'])

    def test_empty_hold_and_invalid_batch(self):
        sample = ProjectionPoolDataset(self.root, hold_views=0)[0]
        batch = projection_pool_collate([sample, sample])
        self.assertEqual(batch['hold_projection'].shape[:2], (2, 0))
        self.assertEqual(batch['hold_view_mask'].shape, (2, 0))
        with self.assertRaises(ValueError): projection_pool_collate([])
        with self.assertRaises(ValueError): projection_pool_collate([sample, {'case_id': 'a'}])
