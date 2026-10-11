import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from pool_fixture import make_pool


def signatures(loader):
    output = []
    for batch in loader:
        for i, case_id in enumerate(batch['case_id']):
            output.append((case_id,
                           tuple(batch['input_station_indices'][i][batch['view_mask'][i]].tolist()),
                           tuple(batch['hold_station_indices'][i][batch['hold_view_mask'][i]].tolist())))
    return output


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary, cls.root, cls.rows, cls.geometries = make_pool()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_membership_and_pool_policy(self):
        for split, count in [('train', 3), ('val', 2), ('test', 2)]:
            pools = ['train'] if split == 'train' else ['train', 'ood']
            for pool in pools:
                dataset = ProjectionPoolDataset(self.root, split, pool)
                self.assertEqual(len(dataset), count)
                self.assertTrue(all(r['split'] == split for r in dataset.records))
                self.assertEqual(dataset[0]['geometry_pool'], pool)
        with self.assertRaises(ValueError): ProjectionPoolDataset(self.root, 'train', 'ood')

    def test_train_never_reads_pool_b_or_gt128_by_default(self):
        original_load = np.load
        def guarded(path, *args, **kwargs):
            self.assertNotIn('ood', str(path))
            self.assertNotIn('source_', str(path))
            return original_load(path, *args, **kwargs)
        with patch('numpy.load', side_effect=guarded):
            dataset = ProjectionPoolDataset(self.root)
            for i in range(len(dataset)): self.assertNotIn('gt_volume_128', dataset[i])

    def test_raw_normalization_validity_and_geometry(self):
        dataset = ProjectionPoolDataset(self.root)
        sample = dataset[0]
        raw = np.load(self.root / self.rows[0]['train_projection_path'])
        for prefix in ('input', 'hold'):
            ids = sample[f'{prefix}_station_indices'].numpy()
            np.testing.assert_array_equal(sample[f'{prefix}_projection_raw'][:, 0].numpy(), raw[ids])
            torch.testing.assert_close(sample[f'{prefix}_projection'], sample[f'{prefix}_projection_raw']/1000)
            np.testing.assert_array_equal(sample[f'{prefix}_valid_ray'][:, 0].numpy(),
                                          self.geometries['train']['valid_ray'][ids])
            np.testing.assert_allclose(sample[f'{prefix}_station_origin_m'].numpy(),
                                       self.geometries['train']['station_origin_m'][ids], rtol=1e-6)
            np.testing.assert_allclose(sample[f'{prefix}_ray_direction'].numpy(),
                                       self.geometries['train']['ray_direction'][ids], rtol=1e-6)
        self.assertGreater(sample['input_projection'].max().item(), 1.)  # No clipping.
        self.assertEqual(sample['projection_scale_ppm_m'].item(), 1000.)
        self.assertEqual(sample['gt_volume_64'].shape, (1, 64, 64, 64))
        self.assertTrue(torch.all((sample['gt_volume_64'] >= 0) & (sample['gt_volume_64'] <= 1)))
        self.assertIsNone(sample['condition_meta']['source_mode'])
        ids = np.r_[sample['input_station_indices'], sample['hold_station_indices']]
        self.assertEqual(len(ids), len(np.unique(ids)))

    def test_log_transform_and_optional_gt128(self):
        dataset = ProjectionPoolDataset(self.root, normalization='log1p_p99_5', load_gt128=True)
        sample = dataset[0]
        torch.testing.assert_close(sample['input_projection'], torch.log1p(sample['input_projection_raw']/1000))
        self.assertEqual(sample['gt_volume_128'].shape, (1, 128, 128, 128))
        np.testing.assert_array_equal(sample['gt_volume_128'][0].numpy(), np.load(self.rows[0]['source_target_path']))

    def test_valid_zero_measurement_keeps_its_validity(self):
        sample = ProjectionPoolDataset(self.root, min_input_views=12, max_input_views=12, hold_views=0)[0]
        self.assertTrue(torch.any((sample['input_projection_raw'] == 0) & sample['input_valid_ray']))
        self.assertTrue(torch.any((sample['input_projection_raw'] == 0) & ~sample['input_valid_ray']))

    def test_statistics_from_train_required(self):
        path = self.root / 'train_statistics.json'
        original = path.read_text()
        try:
            for change in ({'split': 'val'}, {'case_ids': ['case-4']},
                           {'geometry_pool': 'B'}, {'projection_valid_rays': {'p99.5': 0}}):
                path.write_text(json.dumps({**json.loads(original), **change}))
                with self.assertRaises(ValueError): ProjectionPoolDataset(self.root)
        finally:
            path.write_text(original)

    def test_epoch_reproducibility(self):
        a, b = ProjectionPoolDataset(self.root), ProjectionPoolDataset(self.root)
        a.set_epoch(3); b.set_epoch(3)
        torch.testing.assert_close(a[0]['input_station_indices'], b[0]['input_station_indices'])
        first = tuple(a[0]['input_station_indices'].tolist())
        a.set_epoch(4)
        self.assertNotEqual(first, tuple(a[0]['input_station_indices'].tolist()))
        a.set_epoch(3)
        self.assertEqual(first, tuple(a[0]['input_station_indices'].tolist()))

    def test_persistent_multiworker_shared_epoch(self):
        a, b = ProjectionPoolDataset(self.root), ProjectionPoolDataset(self.root)
        loaders = [DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True,
                              collate_fn=projection_pool_collate) for ds in (a, b)]
        try:
            epoch0 = signatures(loaders[0])
            self.assertEqual(epoch0, signatures(loaders[1]))
            a.set_epoch(1); b.set_epoch(1)
            epoch1 = signatures(loaders[0])
            self.assertEqual(epoch1, signatures(loaders[1]))
            self.assertNotEqual(epoch0, epoch1)
            a.set_epoch(0)
            self.assertEqual(epoch0, signatures(loaders[0]))
        finally:
            for loader in loaders:
                if loader._iterator is not None: loader._iterator._shutdown_workers()

    def test_spawn_worker_epoch(self):
        dataset = ProjectionPoolDataset(self.root)
        loader = DataLoader(dataset, batch_size=2, num_workers=1, persistent_workers=True,
                            multiprocessing_context='spawn', collate_fn=projection_pool_collate)
        try:
            first = signatures(loader)
            dataset.set_epoch(1)
            self.assertNotEqual(first, signatures(loader))
            dataset.set_epoch(0)
            self.assertEqual(first, signatures(loader))
        finally:
            if loader._iterator is not None: loader._iterator._shutdown_workers()

    def test_worker_count_invariant_station_ids_and_order(self):
        datasets = [ProjectionPoolDataset(self.root) for _ in range(3)]
        loaders = [DataLoader(ds, batch_size=2, num_workers=workers,
                              persistent_workers=workers > 0, collate_fn=projection_pool_collate)
                   for ds, workers in zip(datasets, (0, 2, 4))]
        try:
            epoch0 = [signatures(loader) for loader in loaders]
            self.assertEqual(epoch0[0], epoch0[1])
            self.assertEqual(epoch0[0], epoch0[2])
            self.assertEqual(epoch0[0], signatures(loaders[0]))
            for ds in datasets: ds.set_epoch(1)
            epoch1 = [signatures(loader) for loader in loaders]
            self.assertEqual(epoch1[0], epoch1[1])
            self.assertEqual(epoch1[0], epoch1[2])
            self.assertGreater(sum(a != b for a, b in zip(epoch0[0], epoch1[0])), len(epoch0[0]) / 2)
            for ds in datasets: ds.set_epoch(0)
            for loader in loaders: self.assertEqual(epoch0[0], signatures(loader))
        finally:
            for loader in loaders:
                if loader._iterator is not None: loader._iterator._shutdown_workers()

    def test_missing_index_does_not_fallback_to_scan(self):
        path = self.root / 'source_index.jsonl'
        backup = path.with_suffix('.backup')
        path.rename(backup)
        try:
            with self.assertRaises(FileNotFoundError): ProjectionPoolDataset(self.root)
        finally:
            backup.rename(path)

    def test_geometry_loaded_once(self):
        original_load = np.load
        calls = []
        def tracked(path, *args, **kwargs):
            if str(path).endswith('.npz'): calls.append(path)
            return original_load(path, *args, **kwargs)
        with patch('numpy.load', side_effect=tracked):
            dataset = ProjectionPoolDataset(self.root)
            for i in range(len(dataset)): dataset[i]
        self.assertEqual(len(calls), 1)
