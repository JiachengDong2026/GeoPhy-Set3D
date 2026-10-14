import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pool_fixture import make_pool
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import GeoLiftOARIMRGRRes3D
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import (
    DiagnosticReadGuard, capture_context, masked_moments, safe_ratio, stage_amplitudes,
    aggregate_cases, paired_summary)


class FrozenC0C1Tests(unittest.TestCase):
    def test_real_return_stages_hook_and_padding(self):
        torch.set_num_threads(2)
        temporary,root,_,_=make_pool()
        try:
            ds=MeasurementOnlyPoolDataset(root,split='val',min_input_views=4,max_input_views=4,hold_views=0)
            b=projection_pool_collate([ds[0]])
            model=GeoLiftOARIMRGRRes3D().eval()
            with torch.no_grad():
                model.residual_adapter[-1].weight.normal_(0,.02)
                plain=model(b)
                with capture_context(model) as result:stages=model(b,return_stages=True)
            torch.testing.assert_close(plain,stages['C1'],atol=0,rtol=0)
            stats=stage_amplitudes(stages,result['context0'],b,1e-12)
            self.assertTrue(stats['context_contract_exact'])
            self.assertGreater(stats['valid_context_scalar_count'],0)
            self.assertEqual(len(model.fusion._forward_pre_hooks),0)
            self.assertEqual(result['local_observability'].shape[-1],5)
        finally:temporary.cleanup()

    def test_invalid_padding_nan_excluded_and_zero_denominator(self):
        v=torch.tensor([[[2.,float('nan')]],[[float('nan'),float('nan')]]])
        mask=torch.tensor([[[True,False]],[[False,False]]])
        stats=masked_moments(v,mask)
        self.assertEqual(stats['count'],1);self.assertEqual(stats['rms'],2)
        self.assertIsNone(masked_moments(v,torch.zeros_like(mask))['rms'])
        self.assertIsNone(safe_ratio(1,0,1e-12))
        self.assertIsNone(safe_ratio(1,1e-14,1e-12))
        self.assertIsNone(safe_ratio(None,1,1e-12))
        self.assertEqual(safe_ratio(2,4,1e-12),.5)
        with self.assertRaises(FloatingPointError):masked_moments(v,torch.ones_like(mask))

    def test_case_aggregation_requires_unique_five_draws(self):
        rows=[dict(case_id=case,case_index=i,group='G1',draw=d,geometry_id=f'G1-d{d}',value=(i+1)*(d+1)) for i,case in enumerate(['a','b']) for d in range(5)]
        table=aggregate_cases(rows)
        np.testing.assert_allclose(table.value,[3,6])
        with self.assertRaises(ValueError):aggregate_cases(rows[:-1])
        with self.assertRaises(ValueError):aggregate_cases(rows+[rows[0]])
        indices=np.array([[0,1],[1,0],[0,0],[1,1]])
        s=paired_summary([3,6],[2,8],indices,1e-8)
        self.assertEqual(s['case_count'],2);self.assertEqual(s['delta_mean'],.5)
        self.assertEqual(s['improved'],1);self.assertEqual(s['worsened'],1)

    def test_inference_gt_and_posthoc_measurement_boundaries(self):
        temporary,root,records,_=make_pool()
        try:
            out=root.parent/'diagnostic';out.mkdir(exist_ok=True)
            val=next(r for r in records if r['split']=='val')
            gt=root/val['gt_volume_64_path'];projection=root/val['train_projection_path']
            with DiagnosticReadGuard(root,[],out,'inference') as guard:
                with projection.open('rb') as f:self.assertTrue(f.read(4))
                for path in [gt,Path(val['source_target_path']),root/val['ood_projection_path']]:
                    with self.assertRaises(PermissionError):path.open('rb')
            self.assertEqual(guard.summary()['val_gt64_opens'],0)
            with DiagnosticReadGuard(root,[],out,'gt') as guard:
                with gt.open('rb') as f:self.assertTrue(f.read(4))
                with self.assertRaises(PermissionError):projection.open('rb')
                with self.assertRaises(PermissionError):Path('/tmp/foreign.pt').open('rb')
            self.assertEqual(guard.summary()['checkpoint_opens'],0)
        finally:temporary.cleanup()

if __name__=='__main__':unittest.main()
