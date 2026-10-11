import tempfile
import unittest
from pathlib import Path
import torch
from geophy_set3d.models import GeoLiftOARIMRGRRes3D
from geophy_set3d.evaluation.mrgr import evaluate_stages
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.measurement_sampling import attach_hold
from geophy_set3d.training.utils import load_config
from pool_fixture import make_pool


class MRGRPairedEvaluationTests(unittest.TestCase):
    def test_stage_metrics_and_disjoint_hold_retain_input_pairing(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        temporary, root, _, _=make_pool()
        try:
            ds=ProjectionPoolDataset(root, split='val', min_input_views=4, max_input_views=4)
            ids=[ds[i]['input_station_indices'].tolist() for i in range(len(ds))]
            ds=attach_hold(ds)
            config=load_config('configs/train_supervised_oari_mrgr_300.yaml'); config['training']['device']='cpu'
            with tempfile.TemporaryDirectory() as out:
                summary,rows=evaluate_stages(GeoLiftOARIMRGRRes3D().eval(), ds, config, out, condition='Val-A-V4',heldout=True)
                self.assertEqual(summary['C0']['mean_per_case'],summary['C1']['mean_per_case'])
                self.assertEqual(summary['C0']['observed'],summary['C1']['observed'])
                self.assertEqual(summary['C0']['heldout'],summary['C1']['heldout'])
                self.assertTrue((Path(out)/'C1/per_case_metrics.jsonl').exists())
                for i,row in enumerate(rows['C1']):
                    self.assertEqual(row['selected_station_ids'],ids[i])
                    self.assertFalse(set(ids[i]) & set(row['hold_station_ids']))
        finally:
            temporary.cleanup()
