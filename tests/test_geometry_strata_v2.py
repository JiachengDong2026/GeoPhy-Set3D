import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from geophy_set3d.evaluation.geometry_catalog import GeometryCatalog
from geophy_set3d.evaluation.geometry_strata import population,frozen_subsets,membership,main_strata,validate_shared_inputs
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler

DATA=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')

class GeometryStrataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g=dict(np.load(DATA/'geometry/train_geometry.npz',allow_pickle=False))
        cls.catalog=GeometryCatalog(cls.g,4,8)
        cls.d=cls.catalog.describe([0,1,2,3])
        rng=np.random.default_rng(1)
        cls.origins=rng.uniform(-2,2,(12,3))
    def test_geometry_catalog_station_height_axis(self):
        np.testing.assert_array_equal(self.d['station_height_m'],self.g['station_origin_m'][[0,1,2,3],1])
        y=self.g['station_origin_m'][:,1];self.assertTrue(np.all((y>=.25)&(y<=.35)))
        self.assertFalse(np.array_equal(y,self.g['station_origin_m'][:,2]))
    def test_geometry_catalog_eigenvalue_order(self):
        d=self.d;self.assertLessEqual(d['ray_tensor_lambda_min'],d['ray_tensor_lambda_mid']);self.assertLessEqual(d['ray_tensor_lambda_mid'],d['ray_tensor_lambda_max'])
        np.testing.assert_allclose(d['actual_ray_direction_tensor_eigenvalues'],[d['ray_tensor_lambda_min'],d['ray_tensor_lambda_mid'],d['ray_tensor_lambda_max']])
    def test_geometry_catalog_lambda_ratio_finite(self):
        for k in ['ray_tensor_lambda_min_over_max','ray_tensor_lambda_mid_over_max']:self.assertTrue(np.isfinite(self.d[k]));self.assertGreaterEqual(self.d[k],0);self.assertLessEqual(self.d[k],1)
    def guarded(self,kind):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'geometry').mkdir();out=root/'out';out.mkdir()
            records=[dict(split='val',train_projection_path='val.npy',gt_volume_64_path='gt.npy',source_target_path=str(root/'gt128.npy'))]
            (root/'source_index.jsonl').write_text(json.dumps(records[0])+'\n')
            np.save(root/'val.npy',np.zeros(1));np.save(root/'gt.npy',np.zeros(1));np.save(root/'test.npy',np.zeros(1));np.savez(root/'geometry/ood_geometry.npz',x=np.zeros(1))
            with ClusteredEvaluationReadGuard(root,[],out,geometry_only=True) as guard:
                population(self.origins,3,1)
                path={'measurement':root/'val.npy','gt':root/'gt.npy','pool_b':root/'geometry/ood_geometry.npz','test':root/'test.npy'}[kind]
                with self.assertRaises(PermissionError):np.load(path)
                self.assertEqual(guard.counts['val_pool_a_projection_opens'],0);self.assertEqual(guard.counts['val_gt64_opens'],0)
    def test_geometry_boundary_generation_no_measurement_access(self):self.guarded('measurement')
    def test_geometry_boundary_generation_no_gt_access(self):self.guarded('gt')
    def test_pool_b_not_accessed(self):self.guarded('pool_b')
    def test_final_test_not_accessed(self):self.guarded('test')
    def test_V_specific_boundaries(self):
        a=population(self.origins,3,1)[2];b=population(self.origins,4,1)[2]
        self.assertEqual(a['population_size'],220);self.assertEqual(b['population_size'],495);self.assertNotEqual(a['boundaries'],b['boundaries'])
    def test_frozen_subset_reproducibility(self):
        ids,scores,s=population(self.origins,3,1)
        self.assertEqual(frozen_subsets(ids,scores,s['boundaries'],2,3),frozen_subsets(ids,scores,s['boundaries'],2,3))
        a,sc,s=population(self.origins,6,1,100);b,bc,_=population(self.origins,6,1,100);np.testing.assert_array_equal(a,b);self.assertEqual(len(set(map(tuple,a))),100)
    def test_stratum_membership_matches_frozen_boundaries(self):
        ids,scores,s=population(self.origins,3,1)
        for r in frozen_subsets(ids,scores,s['boundaries'],2,3):self.assertEqual(membership(r['center_direction_score'],s['boundaries']),r['stratum'])
        self.assertEqual(membership(s['boundaries'][0],s['boundaries']),'middle')
    def test_evaluation_models_share_identical_station_ids(self):
        r=dict(section='controlled',V=3,stratum='low',case_id='x',draw=0,input_station_ids=[3,1,2],H_station_ids=[4,5])
        validate_shared_inputs([dict(model='A',**r),dict(model='B',**r)])
        with self.assertRaises(ValueError):validate_shared_inputs([r,dict(r,input_station_ids=[1,2,3])])
    def test_extreme_clustered_excluded_from_main_strata(self):
        rows=[dict(section=s) for s in ['controlled','natural','clustered']];self.assertEqual(main_strata(rows),[rows[0]])
    def test_natural_random_preserved(self):
        sampler=EvaluationViewSampler(self.g['station_azimuth_deg'],4,0,'train',20261011,'random')
        root=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1/selfsup_oari_mrgr_v1/V4/draw0/C1/per_case_metrics.jsonl')
        for r in map(json.loads,root.read_text().splitlines()):self.assertEqual(sampler.select(r['case_id']).tolist(),r['selected_station_ids'])

if __name__=='__main__':unittest.main()
