"""Independent workflow policy tests; deliberately denied opens never read data."""
import unittest,tempfile,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_classical_v4 import ReadGuard,verify_selection

class ClassicalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text())
        self.records=[json.loads(l) for l in (Path(self.cfg['dataset'])/'source_index.jsonl').read_text().splitlines() if l.strip()]
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_train_calibration_does_not_admit_val_test_or_GT128(self):
        selected=next(r for r in self.records if r['split']=='train')
        with ReadGuard(self.cfg,self.out,'calibrate',[selected]) as guard:
            for split in ['val','test']:
                row=next(r for r in self.records if r['split']==split)
                for key in ['gt_volume_64_path','train_projection_path']:
                    with self.assertRaises(PermissionError):open(Path(self.cfg['dataset'])/row[key],'rb')
            with self.assertRaises(PermissionError):open(selected['source_target_path'],'rb')
            self.assertEqual(guard.forbidden,5)
    def test_evaluation_does_not_admit_train_pool_b_or_weights(self):
        train=next(r for r in self.records if r['split']=='train')
        with ReadGuard(self.cfg,self.out,'evaluate') as guard:
            for p in [Path(self.cfg['dataset'])/train['gt_volume_64_path'],Path(self.cfg['dataset'])/train['ood_projection_path'],Path(self.cfg['dataset'])/'geometry/ood_geometry.npz']:
                with self.assertRaises(PermissionError):open(p,'rb')
            self.assertEqual(guard.forbidden,3)
    def test_frozen_dataset_write_denied(self):
        with ReadGuard(self.cfg,self.out,'register') as guard:
            with self.assertRaises(PermissionError):open(Path(self.cfg['dataset'])/'forbidden.tmp','w')
            self.assertEqual(guard.forbidden,1)
    def test_geometry_only_registration_no_gt(self):
        row=next(r for r in self.records if r['split']=='train')
        with ReadGuard(self.cfg,self.out,'register'):
            with self.assertRaises(PermissionError):open(Path(self.cfg['dataset'])/row['gt_volume_64_path'],'rb')

if __name__=='__main__':unittest.main()
