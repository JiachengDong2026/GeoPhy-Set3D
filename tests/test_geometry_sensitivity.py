import unittest
import numpy as np
from geophy_set3d.evaluation.geometry_sensitivity import circular_span, group_masks, spatial_descriptors, spectrum_from_gram, correlation_rows

class GeometrySensitivityTests(unittest.TestCase):
    def test_wrapping_azimuth_and_groups(self):
        np.testing.assert_allclose(circular_span([[359,1,2,0],[0,90,180,270]]),[3,270])
        score=np.array([.9,.7,.5,.2,.01,.01]);span=np.array([240,200,180,90,3,180])
        masks=group_masks(score,span,[.05,.4,.6,.8],5)
        np.testing.assert_array_equal(np.stack(list(masks.values())).sum(0),[1,1,1,1,1,0])
        self.assertTrue(masks['G5'][4]);self.assertFalse(masks['G5'][5])
    def test_hull_axes_and_distance_units(self):
        o=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]],float)
        d=spatial_descriptors(o,[0,90,180,270])
        self.assertAlmostEqual(d['horizontal_hull_area_m2'],1)
        self.assertEqual(d['station_hull_volume_m3'],0)
        self.assertAlmostEqual(d['min_pairwise_distance_m'],1)
    def test_spectrum_identity_rank_deficiency_and_units(self):
        d=spectrum_from_gram(np.eye(4)*9)
        self.assertAlmostEqual(d['coarse_sigma_min_raw_ppm_m'],3)
        self.assertAlmostEqual(d['coarse_effective_rank'],4)
        self.assertEqual(d['coarse_condition_number'],1)
        d=spectrum_from_gram(np.diag([0,1,4,9]))
        self.assertIsNone(d['coarse_condition_number']);self.assertEqual(d['coarse_nullity'],1)
        self.assertEqual(d['coarse_retained_condition_number'],3)
    def test_constant_correlations_are_undefined(self):
        self.assertTrue(np.isnan(correlation_rows(np.ones(4),np.arange(4))))
        self.assertAlmostEqual(float(correlation_rows(np.arange(4),np.arange(4))),1)
        self.assertAlmostEqual(float(correlation_rows(np.arange(4),-np.arange(4),True)),-1)

if __name__=='__main__':unittest.main()
