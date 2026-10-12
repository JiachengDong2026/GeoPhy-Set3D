"""Targeted diagnostic tests use explicit fixtures, not fabricated experiment results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pool_fixture import make_pool
from geophy_set3d.data.rich_measurement import RichMeasurementDataset
from geophy_set3d.evaluation import rvtc_failure_diagnostics as diag
from geophy_set3d.training import selfsup_v2


class FailureDiagnosticTests(unittest.TestCase):
    def test_history_window_and_fixed_evaluation_are_distinct(self):
        root=Path('docs/reports/artifacts/selfsup_mrgr_v2')
        review,rows=diag.history_review(root)
        self.assertAlmostEqual(review['first10_mean'],.03606435666442849)
        self.assertAlmostEqual(review['last10_mean'],.09986418401822447)
        self.assertEqual(review['minimum_training_discrepancy']['epoch'],9)
        self.assertAlmostEqual(review['minimum_training_discrepancy']['sparse_H_l1'],9.566079106850388,places=5)
        self.assertAlmostEqual(review['fixed_parameter_evaluations']['initial']['teacher_discrepancy'],.10301753599196672)
        self.assertAlmostEqual(review['fixed_parameter_evaluations']['final']['teacher_discrepancy'],.09283932950347662)
        self.assertFalse(review['original_protocol_passed'])
        self.assertEqual(len(rows),80)

    def test_new_stations_deterministic_disjoint_and_sufficient(self):
        protocol=dict(A=[0,1],B=[2,3],H=[4],T=[0,1,2,3])
        a=diag.unseen_station_ids(12,protocol,'case-a',2026101201,4)
        np.random.seed(47)
        self.assertEqual(a,diag.unseen_station_ids(12,protocol,'case-a',2026101201,4))
        self.assertFalse(set(a)&set(range(5)))
        with self.assertRaises(ValueError):diag.unseen_station_ids(6,protocol,'case-a',1,4)

    def test_case_paired_summary_and_no_ray_resampling(self):
        rows=[dict(case_id=str(i),student_relative_l1=1.,student_relative_l2=2.,T_relative_l1=1.+d,T_relative_l2=2.+d,
                   teacher_minus_student_relative_l1=d,teacher_minus_student_relative_l2=d) for i,d in enumerate((-.2,-.1,.3))]
        result=diag.paired_summary(rows,1000,3)
        self.assertAlmostEqual(result['relative_l1']['teacher_minus_student_mean'],0.)
        self.assertAlmostEqual(result['relative_l1']['teacher_minus_student_median'],-.1)
        self.assertAlmostEqual(result['relative_l1']['teacher_win_rate'],2/3)
        self.assertEqual(result['relative_l1']['teacher_worse_case_count'],1)
        self.assertEqual(result,diag.paired_summary(rows,1000,3))
        with self.assertRaises(ValueError):diag.paired_summary(rows+[rows[0]],1000,3)

    def test_weighted_losses_exact_frozen_objective(self):
        parts={key:torch.tensor(float(i+1)) for i,key in enumerate(('hold','cross','observed','hold_teacher','observed_teacher','teacher'))}
        losses=diag.weighted_losses(parts)
        torch.testing.assert_close(sum(losses.values()),sum(selfsup_v2.FROZEN_LOSS[k]*v for k,v in parts.items()))
        self.assertEqual(float(losses['consistency']),float(.1*parts['teacher']))

    def test_gradient_conflict_and_zero_norm_defined(self):
        vectors=dict(student_measurement=torch.tensor([1.,0.]),teacher_measurement=torch.tensor([1.,1.]),consistency=torch.tensor([-2.,0.]))
        result=diag.vector_geometry(vectors)
        self.assertEqual(result['cosine_student_consistency'],-1.)
        self.assertAlmostEqual(result['cosine_teacher_consistency'],-2**-.5)
        self.assertAlmostEqual(result['consistency_to_measurement_sum_norm'],2/5**.5)
        vectors['consistency']=torch.zeros(2)
        self.assertIsNone(diag.vector_geometry(vectors)['cosine_measurement_consistency'])

    def test_guard_denies_gt_pool_b_other_checkpoint_and_writes(self):
        temporary,root,records,_=make_pool()
        try:
            own=root.parent/'gate.pt';own.write_bytes(b'fixture')
            with diag.DiagnosticReadGuard(root,own) as guard:
                self.assertEqual(own.read_bytes(),b'fixture')
                for path,mode in ((root/records[0]['gt_volume_64_path'],'rb'),
                                  (Path(records[0]['source_target_path']),'rb'),
                                  (root/records[0]['ood_projection_path'],'rb'),
                                  (root/'geometry/ood_geometry.npz','rb'),
                                  (own.parent/'supervised.pt','rb'),(own.parent/'other_gate.pt','rb'),(own,'wb')):
                    with self.assertRaises(PermissionError):path.open(mode)
            self.assertEqual(guard.summary()['gt_open_attempts'],2)
            self.assertEqual(guard.summary()['Pool_B_access_attempts'],2)
            self.assertEqual(own.read_bytes(),b'fixture')
        finally:temporary.cleanup()

    def test_read_only_gradient_branch_and_shared_conflict(self):
        # Fixture deliberately has correct detach yet anti-aligned shared gradients.
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.3))
            def forward(self,batch):return (self.p*batch['factor']).expand(1,1,2,2,2)
        model=Model();before=model.p.detach().clone()
        def objective(m,b):
            a=m(dict(factor=1.));bb=m(dict(factor=2.));t=m(dict(factor=4.))
            teacher=selfsup_v2.teacher_discrepancy(a,bb,t)
            parts=dict(hold=a.mean(),cross=bb.mean(),observed=a.mean(),hold_teacher=t.mean(),observed_teacher=t.mean(),teacher=teacher)
            total=sum(selfsup_v2.FROZEN_LOSS[k]*v for k,v in parts.items())
            return total,parts,(a,bb,t),{}
        with patch.object(diag,'assert_rich_union'),patch.object(diag,'objective',side_effect=objective),patch('torch.optim.AdamW.step',side_effect=AssertionError('Update forbidden')):
            result,_=diag.gradient_diagnostic(model,{})
        self.assertTrue(result['teacher_consistency_output_gradient_zero'])
        self.assertGreater(result['student_A_gradient_dot_error'],0)
        self.assertLess(result['cosine_measurement_consistency'],0)
        torch.testing.assert_close(model.p,before)
        self.assertIsNone(model.p.grad)

    def test_actual_loss_no_gt_and_U_not_network_input(self):
        temporary,root,_,_=make_pool()
        try:
            ds=RichMeasurementDataset(root,split='train',a_range=(3,3),b_range=(3,3),hold_views=2)
            sample=ds[0];protocol={key:sample[key]['input_station_indices'].tolist() for key in ('A','B','H','T')}
            u=diag.unseen_station_ids(12,protocol,sample['A']['case_id'],7,2)
            fixed=diag.fixed_sample(ds,0,protocol,u);batch=diag.collate_sample(fixed,'cpu')
            seen=[]
            class Model(torch.nn.Module):
                def forward(self,b):
                    seen.append(id(b));return torch.full((1,1,2,2,2),.3)
            with patch.object(selfsup_v2,'project_input',side_effect=lambda c,b:torch.zeros_like(b['input_projection_raw'])):
                selfsup_v2.objective(Model(),batch)
            self.assertEqual(seen,[id(batch[k]) for k in ('A','B','T')])
            self.assertNotIn(id(batch['U']),seen);self.assertNotIn(id(batch['H']),seen)
        finally:temporary.cleanup()

    def test_mean_ratio_times_mean_denominator_not_absolute_error(self):
        ratios=np.array([1.,3.]);denominators=np.array([3.,1.])
        self.assertNotEqual(float(ratios.mean()*denominators.mean()),float((ratios*denominators).mean()))
