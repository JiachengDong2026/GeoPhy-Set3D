import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from geophy_set3d.data.build_utils import digest
from scripts import evaluate_selfsup_with_gt as evaluation


class SelfsupEvaluationBoundaryTests(unittest.TestCase):
    def setup_state(self,root,epochs=300):
        checkpoint=root/'selfsup.pt';checkpoint.write_bytes(b'boundary-fixture')
        freeze=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),completed_epochs=300,volumetric_gt_used=False)
        (root/'frozen_checkpoint.json').write_text(json.dumps(freeze))
        logs=root/'logs/experiment';logs.mkdir(parents=True)
        summary=dict(completed_epochs=epochs,completed=True,selection_metric='mean held-out measurement relative L1',
                     gt_access=dict(gt_open_attempts=0,supervised_checkpoint_open_attempts=0))
        (logs/'training_summary.json').write_text(json.dumps(summary))
        return dict(experiment=dict(output_root=str(root),name='experiment')),checkpoint

    def test_gt_evaluation_rejected_before_complete_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config,_=self.setup_state(root,299)
            with patch.object(evaluation,'ROOT',root),patch.object(evaluation,'evaluation_dataset',side_effect=AssertionError('GT accessed')):
                with self.assertRaises(RuntimeError):evaluation.require_frozen(config)

    def test_gt_evaluation_requires_frozen_hash_and_no_gt_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config,checkpoint=self.setup_state(root)
            with patch.object(evaluation,'ROOT',root):
                evaluation.require_frozen(config)
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):evaluation.require_frozen(config)
