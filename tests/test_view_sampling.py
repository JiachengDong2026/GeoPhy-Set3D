import unittest
import numpy as np

from geophy_set3d.data.view_sampling import ViewSampler


class ViewSamplingTests(unittest.TestCase):
    def test_range_disjointness_order_and_candidate_coverage(self):
        sampler = ViewSampler()
        seen, counts, unsorted = set(), set(), False
        for epoch in range(100):
            inputs, holds = sampler.sample(64, epoch=epoch, case_id='plume')
            self.assertTrue(3 <= len(inputs) <= 6)
            self.assertEqual(len(holds), 1)
            self.assertEqual(len(set(inputs) | set(holds)), len(inputs) + len(holds))
            counts.add(len(inputs)); seen.update(inputs.tolist())
            unsorted |= not np.array_equal(inputs, np.sort(inputs))
        self.assertEqual(counts, {3, 4, 5, 6})
        self.assertGreater(len(seen), 50)
        self.assertTrue(unsorted)

    def test_seed_epoch_case_index_worker_identity(self):
        sampler = ViewSampler(base_seed=123)
        args = dict(epoch=2, sample_index=4, case_id='a', worker_id=1)
        first = sampler.sample(64, **args)
        replay = sampler.sample(64, **args)
        for a, b in zip(first, replay):
            np.testing.assert_array_equal(a, b)
        for key, value in [('epoch', 3), ('sample_index', 5), ('case_id', 'b'), ('worker_id', 2)]:
            changed = sampler.sample(64, **{**args, key: value})
            self.assertNotEqual(tuple(first[0]), tuple(changed[0]))

    def test_invalid_configuration(self):
        for kwargs in ({'min_input_views': 0}, {'max_input_views': 2}, {'hold_views': -1}, {'base_seed': -1}):
            with self.assertRaises(ValueError): ViewSampler(**kwargs)
        with self.assertRaises(ValueError): ViewSampler().sample(6)
        with self.assertRaises(ValueError): ViewSampler().sample(64, epoch=-1)
        self.assertEqual(len(ViewSampler(1, 1, 0).sample(1)[1]), 0)
