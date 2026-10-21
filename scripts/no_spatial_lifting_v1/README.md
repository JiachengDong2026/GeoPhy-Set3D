# GeometryLifting retraining ablation on final Uniform Fusion MRGR

This isolated experiment preserves the official default model sources. Both controls
and ablations use the registered final Multi-scale MRGR + Uniform Fusion constructor.
The parameter-free replacement resizes each detector plane to 32×32 and extrudes it
along X, retaining the original visibility and other geometry descriptors. It acts
on both main and residual feature lifting. It also changes sampling coverage and
depth structure; results are not a pure operator-only causal estimate.

Run with the repository's verified PyTorch environment:

```bash
python scripts/no_spatial_lifting_v1/audit.py
python scripts/no_spatial_lifting_v1/controller.py
```

`audit.py` refuses an existing experiment directory. The controller uses UUID locks,
real CUDA engineering and cross-process resume gates, two matched full-budget seeds,
then all-layout prediction, global seal, authorized GT64 scoring, independent metric
and statistics verification, Chinese report and acceptance. Failed tasks are preserved
and never silently retried. Do not rerun registration over existing artifacts.

Training sources and protocol are frozen before engineering/training. Evaluation
sources and metric definitions are separately frozen before test prediction and GT64.
GT128 is always forbidden. Controls' existing predictions are hashed and reused.
Seeds: 20261011, 20261019; 300 epochs and 24900 updates per seed. Ordinary Val-A-V4 C1
relative L1 is the sole primary endpoint; clustered-V4 is an important secondary.
This is supplementary analysis on a previously used test set.

Scratch root: `/scratch/djc/GeoPhy-Set3D/mrgr_no_spatial_lifting_v1`.
Final report: `docs/reports/mrgr_no_spatial_lifting_v1_zh.md`.

Completed execution used separately registered recovery entries after three
preserved execution anomalies: process exit after session interruption, duplicate
`seed` in the original controller's completed-status merge, and a same-name
`report` import collision after successful scoring/statistics. The original frozen
sources remain unchanged. `recovery_controller_v1.py` restored canonical training
state; `post_training_controller_v2.py` continued the locked evaluation;
`finish_recovery_v3.py` generated the report from existing verified statistics.
These entries require their immutable recovery registrations; do not rerun any
registration or completed inference/scoring. `final_integrity_review.py` additionally
checked saved evidence/exports and archived complete logs, without GT access.

Execution details: `docs/reports/mrgr_no_spatial_lifting_v1_execution_addendum_zh.md`.
Final supplemented acceptance: `docs/reports/artifacts/mrgr_no_spatial_lifting_v1/acceptance_recovery_v1.json`.
