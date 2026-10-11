import unittest
import json
import numpy as np

from geophy_set3d.data.projection_audit import ErrorTotals, extra_error, run_audit
from geophy_set3d.data.build_utils import digest
from geophy_set3d.physics.projectors import mass_to_mole
from pool_fixture import make_pool


class ProjectionAuditTests(unittest.TestCase):
    def test_mask_signed_metrics_and_global_ray_weighting(self):
        a, b = ErrorTotals(), ErrorTotals()
        a.update([3, 999], [1, 0], np.array([True, False]))
        b.update([1, 2], [2, 2], np.array([True, True]))
        a.merge(b)
        metrics = a.metrics()
        self.assertEqual(metrics['valid_ray_count'], 3)
        self.assertAlmostEqual(metrics['relative_l1'], 3/5)
        self.assertAlmostEqual(metrics['relative_l2'], np.sqrt(5/9))
        self.assertEqual(metrics['mae_ppm_m'], 1)
        self.assertAlmostEqual(metrics['rmse_ppm_m'], np.sqrt(5/3))
        self.assertEqual(metrics['maximum_absolute_error_ppm_m'], 2)
        self.assertAlmostEqual(metrics['mean_signed_error_ppm_m'], 1/3)
        self.assertEqual(extra_error({'relative_l1': .1, 'relative_l2': .2},
                                    {'relative_l1': .3, 'relative_l2': .4})['relative_l2_increase'], .2)

    def test_eval_split_rejected_before_any_read(self):
        for split in ('val', 'test'):
            with self.assertRaises(ValueError): run_audit('/missing', '/missing', split=split)

    def test_end_to_end_json_csv_figures_and_train_only_reads(self):
        from unittest.mock import patch
        temporary, root, rows, geometries = make_pool()
        try:
            for row in rows:
                row['source_gt_sha256'] = digest(row['source_target_path'])
            index = root / 'source_index.jsonl'
            index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            row = rows[0]
            projection = root / row['train_projection_path']
            value = float(np.load(row['source_target_path'])[0, 0, 0])
            np.save(projection, (mass_to_mole(value)*1e6*geometries['train']['path_length_m']).astype(np.float32))
            meta = dict(split='train', reference_samples=256,
                        hashes={projection.name: digest(projection),
                                'supervised_target.npy': digest(root / row['gt_volume_64_path'])})
            (projection.parent / 'sample_meta.json').write_text(json.dumps(meta))
            build = dict(status='frozen', reference_samples=256,
                         complete_effective_config={'dataset': {'source_root': str(root / 'source')}},
                         source_index_sha256=digest(index),
                         geometry_hashes={'train_geometry.npz': digest(root / 'geometry/train_geometry.npz')})
            (root / 'dataset_build_manifest.json').write_text(json.dumps(build))
            output = root.parent / 'audit'
            load = np.load
            def guarded(path, *args, **kwargs):
                self.assertNotIn('ood', str(path))
                return load(path, *args, **kwargs)
            with patch('numpy.load', side_effect=guarded):
                summary = run_audit(root, output, max_cases=1, view_chunk=2, threads=2)
            saved = json.loads((output / 'audit_summary.json').read_text())
            self.assertEqual(saved['case_count'], 1)
            self.assertFalse(saved['review']['global_needs_review'])
            self.assertLess(summary['global_metrics']['E_total64']['relative_l1'], 1e-6)
            self.assertEqual(len((output / 'per_case_metrics.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len((output / 'per_station_metrics.csv').read_text().splitlines()), 12*3+1)
            self.assertEqual(len(list((output / 'figures').glob('*.png'))), 4)
            with self.assertRaises(ValueError): run_audit(root, root / 'audits')
        finally:
            temporary.cleanup()
