# OARI mechanism v2

Frozen **Multi-scale MRGR + Uniform Fusion**, seed 20261011, epoch 285,
342,725 parameters. This diagnostic never reads GT64/GT128 or trains a model.
The official model factory and sealed historical predictions are verified by SHA256.

Run from the repository root using the verified environment:

```bash
/scratch/djc/envs/gas-torch/bin/python scripts/oari_mechanism_v2/checks.py
/scratch/djc/envs/gas-torch/bin/python scripts/oari_mechanism_v2/audit.py
/scratch/djc/envs/gas-torch/bin/python scripts/oari_mechanism_v2/controller.py
```

`audit.py` creates a new experiment directory and refuses to overwrite one.
It completes geometry-only audits and freezes the protocol/source identities.
`controller.py` uses the existing UUID lock and two idle snapshots 15 seconds apart,
runs two engineering inputs before formal inference, and uses at most one GPU.
It then runs `analyze.py`, `verify.py`, and `report_results.py` on CPU.
For a detached run, use a supervisor or `Popen(start_new_session=True)` with a
persistent log and closed standard input.

The 175 inputs cover all 35 cases and the five registered draw-0 layouts.
A0/P1/P2 reuse all 525 sealed C0/C1 predictions; only missing OARI intermediates
are replayed. Eight controlled spatial permutation fields and seven channel
interventions produce 2,625 new C0/C1 predictions. Permutations and voxels are
never statistical samples: average the eight fields within each case/layout;
for pooled results, also average layouts within each case before summarizing 35 cases.

Every committed group record has a SHA256 sidecar and hashes its prediction/probe.
Completed groups are reused on restart. Interrupted uncommitted artifacts are
moved into `interrupted_artifacts/`, recorded, and then deterministically replayed.
Failure files/logs are retained. Inspect failures before manually resuming the
controller; it never silently retries a failed worker. A source/hash discrepancy
stops execution; preserve the original protocol rather than editing it to pass.

The independent verifier reconstructs QKV, MLP bias, attention, and the complete
OARI context with NumPy float64 on small predetermined probes. It separately
recalculates every prediction response and regroups raw case records. Full-volume
mask, normalization and softmax checks run in the worker. Reports describe
sensitivity of the frozen model on previously used test cases, not independent
accuracy confirmation or a retrained module's performance contribution.

Large predictions remain under the registered scratch directory. The repository
contains manifests/hashes, raw group records compressed as JSONL, CSV tables
(compressed when large), per-layout probe archives, audits, and Chinese reports.

`audit_layouts.py --watch` separately records completion/hash checks as each layout
finishes, without model execution. `finalize_interpretation.py --register` records
a reporting-only supplement identity, and running it after the controller creates
`mrgr_oari_mechanism_v2_final_zh.md` and `acceptance_final.json`. The final report
uses the actual zero-contamination audit and supplies detailed layout/path tables;
it supersedes only the generated report's wording, leaving its original numerical
acceptance and all protocol/source hashes intact. These supplementary scripts are
hashed in their own registration receipts; the original numerical sources remain
unchanged.

Finally run `review_report.py` for the editorial review receipt. The authoritative
reading pointer is `artifacts/mrgr_oari_mechanism_v2/CURRENT_REPORT.json`, which
links `mrgr_oari_mechanism_v2_reviewed_zh.md` and `acceptance_reviewed.json`.
Older generated/final reports and their numerical acceptance remain preserved;
the review changes wording and the priority of proposed future work only.
