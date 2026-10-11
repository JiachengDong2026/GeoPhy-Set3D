import unittest
import numpy as np
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler, coverage_descriptor


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.az=np.linspace(0,180,64)

    def test_unique_count_replay_draw_pool_and_seed_identity(self):
        for pool in ('train','ood'):
            for views in (2,3,4,5,6,8):
                a=EvaluationViewSampler(self.az,views,0,pool)
                ids=a.select('case-a');self.assertEqual(len(ids),views);self.assertEqual(len(set(ids)),views)
                np.testing.assert_array_equal(ids,a.select('case-a'))
                np.testing.assert_array_equal(ids,a.sample(64,epoch=7,sample_index=99,case_id='case-a')[0])
        sets=[tuple(EvaluationViewSampler(self.az,4,i,'train').select('case-a')) for i in range(5)]
        self.assertEqual(len(set(sets)),5)
        self.assertNotEqual(sets[0],tuple(EvaluationViewSampler(self.az,4,0,'ood').select('case-a')))
        with self.assertRaises(ValueError): EvaluationViewSampler(self.az,4,0,'train').sample(32,epoch=0,sample_index=0,case_id='x')

    def test_spread_and_cluster_algorithms(self):
        for v in (4,6):
            span=lambda ids:coverage_descriptor(self.az[ids])['azimuth_span_deg']
            spread=EvaluationViewSampler(self.az,v,0,'train',protocol='spread').select('x')
            cluster=EvaluationViewSampler(self.az,v,0,'train',protocol='clustered').select('x')
            random=EvaluationViewSampler(self.az,v,0,'train').select('x')
            self.assertEqual(span(spread),180);self.assertGreaterEqual(span(spread),span(random))
            self.assertAlmostEqual(span(cluster),180*(v-1)/63)
            self.assertLess(span(cluster),span(random))

    def test_descriptors_and_invalid_arguments(self):
        d=coverage_descriptor([0,30,90]);self.assertEqual(d['azimuth_span_deg'],90)
        self.assertEqual(d['largest_azimuth_gap_deg'],60);self.assertEqual(d['mean_pairwise_azimuth_distance_deg'],60)
        for kwargs in (dict(views=65,draw_id=0,pool='train'),dict(views=4,draw_id=-1,pool='train'),dict(views=4,draw_id=0,pool='B')):
            with self.assertRaises(ValueError): EvaluationViewSampler(self.az,**kwargs)
