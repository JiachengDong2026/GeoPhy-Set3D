#!/usr/bin/env python3
"""Evaluation entry point with an exact whitelist for this experiment's NPZs.

The historical diagnostic guard only permits caches in its old case_images
directory. Extend that guard for the two independently saved arms, without
changing historical access rules or the registered training implementation.
"""
import json
import os
from pathlib import Path
import sys

import run_ours_c0_adapter_matched as experiment


class SavedOutputReadGuard(experiment.DiagnosticReadGuard):
    def __init__(self, dataset_root, checkpoints, output, phase):
        super().__init__(dataset_root, checkpoints, output, phase)
        c, _, _ = experiment.configuration()
        records = json.loads((experiment.REPO / c['layouts']).read_text())
        self.saved_predictions = {
            (Path(output) / arm / 'predictions' / f'case{r["case_index"]}_{r["geometry_id"]}.npz').resolve()
            for arm in experiment.ARMS for r in records
        }
        assert len(self.saved_predictions) == 1800
        self.saved_npz_opens = 0

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, Path)):
            p = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1:3]
            writing = any(k in mode for k in 'wa+') if isinstance(mode, str) else bool(flags & 3)
            if not writing and p in self.saved_predictions:
                self.saved_npz_opens += 1
                return
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), registered_saved_prediction_npz_opens=self.saved_npz_opens)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('infer', 'gt'):
        raise SystemExit('Usage: evaluate_ours_c0_adapter_matched.py infer|gt')
    c, base, out = experiment.configuration()
    experiment.verify(c, out)
    source = Path(__file__).resolve()
    manifest = dict(source=str(source.relative_to(experiment.REPO)), sha256=experiment.digest(source),
                    original_executor_sha256=experiment.digest(experiment.REPO / 'scripts/run_ours_c0_adapter_matched.py'),
                    scope='Exact 1,800 saved NPZ paths only; original geometry/GT/checkpoint restrictions retained',
                    prediction_values_or_scientific_settings_changed=False)
    path = out / 'evaluation_executor_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        experiment.write_json(path, manifest)
        experiment.archive(out, ['evaluation_executor_manifest.json'])
    experiment.DiagnosticReadGuard = SavedOutputReadGuard
    getattr(experiment, sys.argv[1])(c, base, out)


if __name__ == '__main__':
    main()
