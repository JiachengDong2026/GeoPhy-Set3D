# Ours C0 adapter matched v1

Decision: **NO_GO**. Under identical observations, the added adapter did not
produce a meaningful G5 improvement over matched continuation. The original
frozen SelfSup-MRGR v1 remains the baseline.

See [Chinese report](../../ours_c0_adapter_matched_v1_zh.md) and
[protocol](../../../ours_c0_adapter_matched_v1_protocol_zh.md).

- Each arm: same original GT-free v1 initialization, 20 continuation epochs,
  1,660 AdamW steps, same station plan and case order, unchanged objective.
- Control: 332,070 parameters; Treatment: 333,222 parameters.
- Each frozen arm: original 900 V4 layouts, full input hashes paired, C0/C1
  saved before the independent 36-case Val GT64 evaluation.
- No GT64/128 training, Pool B, final test, or source checkpoint overwrite.
- This is a single-seed continuation experiment, not an equal-budget fresh
  comparison to classical or other deep-learning methods.

Large checkpoints, station plan and 1,800 prediction NPZ files remain under:
`/data/djc/GeoPhy-Set3D/outputs/experiments/ours_c0_adapter_matched_v1/`.
This Git-side archive contains only reports, hashes, scalar results and plots.

The original registration remains unchanged. `preflight_amendment.json` records
the test-fixture repair before optimization. `analysis_source_manifest.json`
records independent evaluation/statistics sources before frozen inference and
new GT evaluation. `evaluation_executor_manifest.json` records the exact
saved-NPZ whitelist extension, which changes no prediction or metric values.

Read-only verification (no forwards, projector calls, or GT reads):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python scripts/verify_ours_c0_adapter_matched.py
```

Independent entry points:

```bash
python scripts/run_ours_c0_adapter_matched.py register
python scripts/run_ours_c0_adapter_matched.py smoke
python scripts/run_ours_c0_adapter_matched.py train --arm Control
python scripts/run_ours_c0_adapter_matched.py train --arm Treatment
python scripts/evaluate_ours_c0_adapter_matched.py infer
python scripts/evaluate_ours_c0_adapter_matched.py gt
python scripts/report_ours_c0_adapter_matched.py
```

Completed training is not repeated; only an incomplete arm's own `last.pt` may
resume. The reporting script regenerates scalar tables/figures and a draft;
the reviewed Chinese report additionally explains auxiliary metrics and limits.
`report_review.json` records the reviewed report hash. No performance inference
is needed to inspect `paired_statistics.csv`, `tails.csv`, or `decision.json`.

The independent model class is not enabled in the original model factory.
Treatment checkpoints are loaded by the experiment evaluator using their arm
and protocol identity; the shared base configuration retains the v1 name.
