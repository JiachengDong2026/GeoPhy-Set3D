# SelfSup-MRGR v2 result snapshot

Decision: **SelfSup-MRGR v2 NO-GO at the engineering gate**.

[Full research report](../../selfsup_mrgr_v2.md)

One fresh-random, eight-case, 80-epoch gate used fixed epoch0 A/B/H/T station IDs.
Sparse held-out relative L1 improved from 377.356155 to 0.519349; rich-teacher
held-out relative L1 was 0.511356. Outputs did not collapse, and the permutation,
padding, all-invalid and branch-gradient checks passed. GT, foreign-checkpoint,
forbidden-projection and Pool B read attempts were all zero. All 179 tests passed.

The only failed criterion was the predeclared temporal student–teacher loss
decrease: first10 / last10 training means were 0.036064 / 0.099864. Failure
inspection found no union, geometry, measurement pairing, mask, stop-gradient or
gradient-direction bug. No gate retry, loss tuning, formal300 training, post-hoc
GT evaluation or final-test evaluation followed. SelfSup v1 remains the baseline.
This gate outcome does not establish that v2 voxel quality is worse than v1.

The files preserve the external run's recorded contents, including original
training/reporting commit IDs and local source paths. `artifact_manifest.json`
records SHA256 and size for every copied file. `gate_metrics.jsonl` contains all
80 epochs (zero-based epoch index); `teacher_training_diagnostics.csv` is its
compact diagnostic view. `terminal_report.txt` contains the full48-item checklist.
The voxel, teacher post-hoc and v1/v2 comparison CSVs explicitly record that those
evaluations were not executed. Model weights and datasets remain in the external
experiment storage.
