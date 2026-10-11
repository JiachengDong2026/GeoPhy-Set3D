import unittest
import numpy as np
import torch
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.evaluation.analysis import aggregate_draws, plume_descriptors, correlation, metric_degradation
from geophy_set3d.evaluation.stratification import stratify


class GeneralizationAnalysisTests(unittest.TestCase):
    def test_draw_variance_is_not_case_variance(self):
        summaries=[];rows=[]
        for draw,values in enumerate(([0.,2.],[1.,3.])):
            m=VolumeMetrics();m.update(torch.tensor(values)[:,None],torch.ones(2,1))
            s=m.compute();s['draw_id']=draw;summaries.append(s)
            for r in m.rows:r['draw_id']=draw
            rows.extend(m.rows)
        a=aggregate_draws(summaries,rows)
        self.assertEqual(a['draw_count'],2);self.assertEqual(a['cases_per_draw'],2)
        expected=np.std([s['mean_per_case']['relative_l1'] for s in summaries],ddof=1)
        self.assertAlmostEqual(a['std_across_draw_means']['relative_l1'],expected)
        self.assertEqual(a['mean_per_case']['relative_l1'],1)

    def test_physical_descriptors_fixed_threshold_and_center(self):
        g=np.zeros((3,3,3));g[0,1,2]=.5;g[2,0,0]=1e-6
        d=plume_descriptors(g)
        self.assertAlmostEqual(d['active_voxel_fraction'],1/27)
        self.assertAlmostEqual(d['nonzero_voxel_fraction'],2/27)
        self.assertAlmostEqual(d['concentration_center_y_m'],.15/(.5+1e-6))
        self.assertAlmostEqual(d['concentration_center_z_m'],.3/(.5+1e-6))
        self.assertAlmostEqual(d['gt_integrated_concentration_proxy'],g.mean()*.216)
        self.assertEqual(d['active_voxel_threshold'],1e-5)

    def test_missing_meta_small_n_and_spearman(self):
        rows=[dict(condition_meta=dict(source_mode=None,wind_speed_m_s=i,source_family='G1' if i<3 else 'G2'),
                   relative_l1=i+1.,relative_l2=i+2.,rmse=.1*i) for i in range(4)]
        groups,continuous=stratify(rows)
        null=next(g for g in groups if g['field']=='source_mode');self.assertEqual(null['value'],'null');self.assertEqual(null['case_count'],4)
        g2=next(g for g in groups if g['field']=='source_family' and g['value']=='"G2"');self.assertTrue(g2['small_n'])
        self.assertEqual(continuous[0]['rho'],1)
        self.assertIsNone(correlation([dict(x=1,y=i) for i in range(4)],'x','y')['rho'])
