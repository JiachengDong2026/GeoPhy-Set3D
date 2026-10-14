import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path('scripts').resolve()))
import evaluate_ray_senseiver as evaluation
from geophy_set3d.training.ray_senseiver import load_comparison_config


class SavedOutputBoundaryTests(unittest.TestCase):
    def test_gt_metadata_verification_never_hashes_checkpoint(self):
        c=load_comparison_config('configs/train_ray_senseiver_selfsup_v1.yaml')
        with tempfile.TemporaryDirectory() as root:
            c=copy.deepcopy(c);c['comparison']['output']=root;p=Path(root)
            (p/'frozen_best.json').write_text(json.dumps(dict(checkpoint='/forbidden/weights.pt',sha256='frozen',completed_epochs=300,optimizer_steps=24900,volumetric_gt_used=False)))
            (p/'training_complete.json').write_text(json.dumps(dict(completed=True,epochs=300,optimizer_steps=24900)))
            (p/'evaluation_source_manifest.json').write_text(json.dumps(dict(sha256={})))
            def hash_allowed(path):
                self.assertFalse(str(path).endswith(('.pt','.pth','.ckpt','.npy')))
                return 'any'
            with patch.object(evaluation,'verify_registration',return_value={}),patch.object(evaluation,'sha256',side_effect=hash_allowed),patch.object(evaluation.subprocess,'check_output',return_value=(p/'evaluation_source_manifest.json').read_bytes()):
                evaluation.verify(c,check_checkpoint=False)

    def test_gt_guard_is_active_before_saved_output_entry(self):
        c=load_comparison_config('configs/train_ray_senseiver_selfsup_v1.yaml')
        with tempfile.TemporaryDirectory() as root:
            c=copy.deepcopy(c);c['comparison']['output']=root
            def forbidden_probe(config,guard):
                self.assertTrue(guard.active)
                with self.assertRaises(PermissionError):Path('/forbidden/weights.pt').read_bytes()
                projection=next(iter(guard.projections))
                with self.assertRaises(PermissionError):Path(projection).read_bytes()
                self.assertEqual(guard.counts['checkpoint_opens'],0)
                self.assertEqual(guard.counts['val_pool_a_projection_opens'],0)
            with patch.object(evaluation,'gt_saved',side_effect=forbidden_probe):evaluation.gt(c)

if __name__=='__main__':unittest.main()
