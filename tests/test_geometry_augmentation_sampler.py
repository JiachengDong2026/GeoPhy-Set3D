import unittest
import numpy as np
from geophy_set3d.evaluation.geometry_augmentation_sampler import GeometryAugmentationABHSampler,hard_subsets
from geophy_set3d.evaluation.geometry_sensitivity import circular_span
from geophy_set3d.data.measurement_only import DisjointABHSampler

class AugmentationSamplerTests(unittest.TestCase):
    def setUp(self):
        self.az=np.r_[np.linspace(0,4,8),np.linspace(90,94,8),np.linspace(175,179,8)]
        theta=np.deg2rad(self.az)
        self.origins=np.stack([.3+6*np.cos(theta),np.linspace(.26,.34,len(theta)),.3+6*np.sin(theta)],1)
        self.bounds={str(v):[0,.7] for v in [3,4,5]}
    def test_wrapped_hard_candidates(self):
        ids=hard_subsets(np.array([359,1,2,3,90]),4,5)
        self.assertEqual(ids.tolist(),[[0,1,2,3]])
    def test_all_modes_V_disjoint_and_reproducible(self):
        sampler=GeometryAugmentationABHSampler(self.origins,self.az,self.bounds)
        seen=set()
        for epoch in range(50):
            identity=dict(epoch=epoch,sample_index=0,case_id='fixture')
            sets,meta=sampler.sample_with_metadata(len(self.az),**identity)
            again,againmeta=sampler.sample_with_metadata(len(self.az),**identity)
            for a,b in zip(sets,again):np.testing.assert_array_equal(a,b)
            self.assertEqual(meta,againmeta)
            baseline=DisjointABHSampler().sample(len(self.az),**identity)
            self.assertEqual(tuple(map(len,sets)),tuple(map(len,baseline)))
            self.assertFalse(set(sets[0])&set(sets[1]));self.assertFalse(set(sets[2])&(set(sets[0])|set(sets[1])))
            for i,k in enumerate(['A_mode','B_mode']):
                if meta[k]=='hard':self.assertLessEqual(circular_span(self.az[sets[i]]),10+1e-10)
                if meta[k]=='moderate':self.assertTrue(0<=sampler.score(sets[i])<.7)
            seen.add((meta['A_mode'],meta['B_mode']))
        self.assertEqual(len(seen),9)
    def test_bad_probabilities_and_impossible_geometry_stop(self):
        with self.assertRaises(ValueError):GeometryAugmentationABHSampler(self.origins,self.az,self.bounds,probabilities=[1,1,1])
        with self.assertRaises(ValueError):GeometryAugmentationABHSampler(self.origins,np.arange(len(self.az))*10,self.bounds,hard_span_deg=1)

if __name__=='__main__':unittest.main()
