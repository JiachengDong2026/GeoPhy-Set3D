"""Check the new cache whitelist does not loosen GT/checkpoint boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_ours_c0_adapter_matched import SavedOutputReadGuard
from report_ours_c0_adapter_matched import stat
from pool_fixture import make_pool


class EvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.pool, self.root, _, _ = make_pool()
        self.output = tempfile.TemporaryDirectory()
        self.out = Path(self.output.name)
        self.rows = [json.loads(x) for x in (Path(self.root) / 'source_index.jsonl').read_text().splitlines()]

    def tearDown(self):
        self.pool.cleanup()
        self.output.cleanup()

    def test_only_expected_prediction_npz_can_be_read(self):
        allowed = self.out / 'Control/predictions/case0_V4-G1-d0.npz'
        allowed.parent.mkdir(parents=True)
        np.savez(allowed, C1=np.zeros(1, np.float32))
        unknown = allowed.parent / 'unknown.npz'
        np.savez(unknown, C1=np.zeros(1, np.float32))
        with SavedOutputReadGuard(self.root, [], self.out, 'gt') as guard:
            with np.load(allowed) as file:
                self.assertEqual(file['C1'][0], 0)
            with self.assertRaises(PermissionError):
                np.load(unknown)
            self.assertEqual(guard.summary()['registered_saved_prediction_npz_opens'], 1)
            self.assertEqual(guard.summary()['forbidden_attempts'], 1)

    def test_inference_still_denies_gt64_and_foreign_weights(self):
        val = next(r for r in self.rows if r['split'] == 'val')
        with SavedOutputReadGuard(self.root, [], self.out, 'inference') as guard:
            with self.assertRaises(PermissionError):
                open(Path(self.root) / val['gt_volume_64_path'], 'rb')
            with self.assertRaises(PermissionError):
                open(self.out / 'foreign.pt', 'rb')
            self.assertEqual(guard.summary()['forbidden_attempts'], 2)

    def test_gt_phase_still_denies_measurements(self):
        val = next(r for r in self.rows if r['split'] == 'val')
        with SavedOutputReadGuard(self.root, [], self.out, 'gt'):
            with self.assertRaises(PermissionError):
                open(Path(self.root) / val['train_projection_path'], 'rb')

    def test_cache_whitelist_cannot_write_source_data(self):
        with SavedOutputReadGuard(self.root, [], self.out, 'inference'):
            with self.assertRaises(PermissionError):
                open(Path(self.root) / 'forbidden-write.npy', 'wb')

    def test_statistics_keep_treatment_minus_reference_direction(self):
        before = np.ones(36)
        indices = np.tile(np.arange(36), (20, 1))
        result = stat(before, before * .8, indices)
        self.assertAlmostEqual(result['delta'], -.2)
        self.assertAlmostEqual(result['reduction'], .2)
        self.assertEqual(result['wins'], 36)
        self.assertLess(result['ci_high'], 0)


if __name__ == '__main__':
    unittest.main()
