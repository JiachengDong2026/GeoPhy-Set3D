"""Targeted geometry, paired statistics and the independent GT64 boundary."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from geophy_set3d.evaluation.clustered_one_swap import (
    directions, geometry_description, targeted_replacement, paired_station_sets,
    case_aggregate, paired_statistics, ClusteredEvaluationReadGuard)
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.rvtc_matched import state_hash
from pool_fixture import make_pool


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.origins=np.random.default_rng(13).normal(size=(16,3))+3
        self.original=[3,0,2,1]

    def test_exhaustive_score_optimum_and_three_preserved(self):
        t=targeted_replacement(self.original,self.origins)
        expected=min((-geometry_description([a if i==d else i for i in self.original],self.origins)['score'],d,a)
            for d in self.original for a in range(4,16))
        self.assertEqual((-t['score'],t['deleted'],t['added']),expected)
        self.assertEqual(len(set(t['station_ids'])),4)
        self.assertEqual(len(set(t['station_ids']) & set(self.original)),3)
        self.assertEqual(list(inspect.signature(targeted_replacement).parameters),['original','origins','center'])

    def test_exact_tie_break_and_antiparallel(self):
        origins=np.array([[2.,0,0],[-2.,0,0],[3.,0,0],[-3.,0,0],[4.,0,0],[-4.,0,0]])
        t=targeted_replacement([3,2,1,0],origins,center=(0,0,0))
        self.assertEqual((t['deleted'],t['added']),(0,4))
        desc=geometry_description([0,1,2,3],origins,center=(0,0,0))
        self.assertEqual(desc['score'],0)
        np.testing.assert_allclose(desc['direction_tensor_eigenvalues'],[0,0,1])
        self.assertEqual(desc['crossing_angle_max_deg'],0)
        with self.assertRaises(ValueError):directions([[.3,.3,.3]])

    def test_replay_random_shared_deletion_h_disjoint_and_coincidence_retained(self):
        fn=lambda seed:paired_station_sets(self.original,self.origins,'fixed',0,seed,101)
        before=np.random.get_state();a=fn(99);after=np.random.get_state()
        self.assertEqual(a,fn(99));np.testing.assert_array_equal(before[1],after[1])
        for cond in ('Original','Targeted','Random'):
            self.assertEqual(len(set(a[cond])),4)
            self.assertFalse(set(a['H']) & set(a[cond]))
        self.assertNotIn(a['deleted'],a['Random'])
        self.assertEqual(set(a['Random']) & set(self.original),set(self.original)-{a['deleted']})
        coincident=next(fn(seed) for seed in range(200) if fn(seed)['random_equals_targeted'])
        self.assertEqual(coincident['Random'],coincident['Targeted'])


class PairingTests(unittest.TestCase):
    def test_five_draws_then_36_case_bootstrap(self):
        rows=[]
        for case in range(36):
            for draw in range(5):
                for cond in ('Original','Targeted'):
                    rows.append(dict(model='m',case_id=f'c{case:02}',draw=draw,condition=cond,
                        error=10+draw+(case/100 if cond=='Targeted' else 0)))
        s=paired_statistics(rows,'error','m','Targeted','Original',10000,123)
        delta=np.arange(36)/100
        expected=np.quantile(delta[np.random.default_rng(123).integers(0,36,(10000,36))].mean(1),[.025,.975])
        np.testing.assert_allclose(s['case_paired_95ci'],expected,atol=1e-14)
        self.assertEqual(s['case_count'],36);self.assertAlmostEqual(s['mean_delta'],delta.mean())
        self.assertEqual((s['wins'],s['losses'],s['ties']),(0,35,1))
        with self.assertRaises(ValueError):case_aggregate(rows[1:],'error','Original','m')
        with self.assertRaises(ValueError):case_aggregate(rows+[rows[0]],'error','Original','m')


class AccessAndAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp,self.root,self.rows,self.geom=make_pool()
        self.output=Path(self.tmp.name)/'output';self.output.mkdir()
        self.ck=Path(self.tmp.name)/'frozen.pt';self.ck.write_bytes(b'checkpoint-test-only')
    def tearDown(self):self.tmp.cleanup()

    def test_registration_blocks_gt_and_measurements(self):
        with ClusteredEvaluationReadGuard(self.root,[self.ck],self.output,True) as guard:
            ds=MeasurementOnlyPoolDataset(self.root,split='val',min_input_views=4,max_input_views=4,hold_views=0,normalization='linear_p99_5')
            for key in ('gt_volume_64_path','train_projection_path'):
                with self.assertRaises(PermissionError):(self.root/self.rows[3][key]).open('rb')
            with self.ck.open('rb') as f:self.assertTrue(f.read())
        self.assertEqual(guard.counts['val_gt64_opens'],0)
        self.assertEqual(guard.counts['val_pool_a_projection_opens'],0)

    def test_all_station_fields_and_mask_indexed_together(self):
        with ClusteredEvaluationReadGuard(self.root,[self.ck],self.output) as guard:
            ds=MeasurementOnlyPoolDataset(self.root,split='val',min_input_views=4,max_input_views=4,hold_views=0,normalization='linear_p99_5')
            row,p=ds._projections(0);ids=np.array([9,2,7,0]);out=ds._measurement_sample(row,p,ids,np.array([],dtype=np.int64))
            np.testing.assert_array_equal(out['input_station_indices'],ids)
            np.testing.assert_array_equal(out['input_projection_raw'][:,0],p[ids])
            for k,v in ds.geometry.items():
                if k in ('horizontal_angle_deg','vertical_angle_deg'):continue
                actual=out['input_'+k].numpy()
                if k in ('valid_ray','path_length_m','t_near','t_far'):actual=actual[:,0]
                np.testing.assert_allclose(actual,v[ids],rtol=1e-6,atol=1e-6)
            np.load(self.root/row['gt_volume_64_path'])
            self.assertEqual(guard.summary()['unique_val_gt64_files'],1)

    def test_forbidden_reads_and_writes(self):
        with ClusteredEvaluationReadGuard(self.root,[self.ck],self.output) as guard:
            forbidden=[self.root/self.rows[0]['gt_volume_64_path'],self.root/self.rows[0]['train_projection_path'],
                self.root/self.rows[5]['train_projection_path'],Path(self.rows[3]['source_target_path']),
                self.root/self.rows[3]['ood_projection_path'],self.root/'geometry/ood_geometry.npz',Path(self.tmp.name)/'foreign.pt']
            for p in forbidden:
                with self.assertRaises(PermissionError):p.open('rb')
            for p in [self.root/'write.json',self.ck]:
                with self.assertRaises(PermissionError):p.open('wb')
        self.assertEqual(guard.counts['forbidden_attempts'],9)

    def test_parameter_buffer_hash_frozen_inference(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)).eval().requires_grad_(False)
        before=state_hash(model.state_dict())
        with torch.inference_mode():model(torch.ones(2,3))
        self.assertEqual(before,state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
