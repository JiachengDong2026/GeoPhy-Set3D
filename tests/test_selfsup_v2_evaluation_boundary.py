import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.selfsup_v2 import SELECTION
from scripts import evaluate_selfsup_v2_with_gt as evaluation


class RVTCBoundaryTests(unittest.TestCase):
    def state(self,root,epochs=300):
        ckpt=root/'best.pt';ckpt.write_bytes(b'frozen-v2-fixture')
        (root/'frozen_checkpoint.json').write_text(json.dumps(dict(checkpoint=str(ckpt),sha256=digest(ckpt),
            completed_epochs=300,volumetric_gt_used=False,selection_metric=SELECTION)))
        logs=root/'logs/v2';logs.mkdir(parents=True)
        training=dict(completed_epochs=epochs,completed=True,selection_metric=SELECTION,
            gt_access=dict(gt_open_attempts=0,supervised_checkpoint_open_attempts=0,forbidden_projection_attempts=0,Pool_B_access_attempts=0))
        (logs/'training_summary.json').write_text(json.dumps(training))
        return dict(experiment=dict(output_root=str(root),name='v2')),ckpt,logs

    def test_reject_gt_before_full_300(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_,_=self.state(root,299)
            with patch.object(evaluation,'ROOT',root),patch.object(evaluation,'evaluation_dataset',side_effect=AssertionError('GT read')):
                with self.assertRaises(RuntimeError):evaluation.require_frozen(config)

    def test_reject_hash_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,ckpt,_=self.state(root)
            with patch.object(evaluation,'ROOT',root):
                evaluation.require_frozen(config);ckpt.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):evaluation.require_frozen(config)

    def test_reject_teacher_or_voxel_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_,logs=self.state(root)
            training=json.loads((logs/'training_summary.json').read_text());training['selection_metric']='teacher heldout'
            (logs/'training_summary.json').write_text(json.dumps(training))
            with patch.object(evaluation,'ROOT',root):
                with self.assertRaises(RuntimeError):evaluation.require_frozen(config)

    def test_reject_any_forbidden_training_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_,logs=self.state(root)
            original=json.loads((logs/'training_summary.json').read_text())
            for key in original['gt_access']:
                training=json.loads(json.dumps(original));training['gt_access'][key]=1
                (logs/'training_summary.json').write_text(json.dumps(training))
                with patch.object(evaluation,'ROOT',root):
                    with self.assertRaises(RuntimeError):evaluation.require_frozen(config)
