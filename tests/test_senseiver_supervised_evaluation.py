"""Supervised evaluation must retain frozen identity and the saved-output GT boundary."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('scripts').resolve()))
import evaluate_senseiver_supervised as ev

class SupervisedEvaluationTests(unittest.TestCase):
    def test_gt_verify_does_not_hash_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=dict(models=dict(Senseiver=dict(checkpoint='/forbidden.pt',sha256='hash')),sources={},references={},manifest_sha256='manifest')
            (out/'protocol.json').write_text(json.dumps(p));(out/'frozen_best.json').write_text(json.dumps(dict(effective_learning=True)));(out/'station_subsets.json').write_text('[]')
            def digest(path):
                self.assertFalse(str(path).endswith(('.pt','.npy','.npz')))
                return 'manifest'
            with patch.object(ev,'EV',out),patch.object(ev,'sha256',side_effect=digest),patch.object(ev.subprocess,'check_output',return_value=(out/'protocol.json').read_bytes()):
                ev.verify({},False)
    def test_existing_guard_rejects_train_test_GT_and_weights_in_gt_phase(self):
        # Use a tiny metadata-only dataset: never open real prohibited data.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';root.mkdir();out=Path(tmp)/'out';out.mkdir()
            records=[dict(split=s,train_projection_path=f'{s}/p.npy',gt_volume_64_path=f'{s}/g.npy',source_target_path=f'{root}/{s}/g128.npy') for s in ['train','val','test']]
            (root/'source_index.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            with ev.DiagnosticReadGuard(root,[],out,'gt') as guard:
                for path in [root/'train/g.npy',root/'test/g.npy',root/'val/g128.npy',root/'val/p.npy',out/'best.pt']:
                    with self.assertRaises(PermissionError):path.read_bytes()
                self.assertEqual(guard.counts['val_pool_a_projection_opens'],0)
                self.assertEqual(guard.counts['checkpoint_opens'],0)
if __name__=='__main__':unittest.main()
