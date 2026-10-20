# Supplementary35-case evaluation

Run `/scratch/djc/envs/gas-torch/bin/python scripts/mrgr_extension_test35_v1/controller.py`.
Requires the immutable historical paper-final and completed pruning/OARI development
artifacts at their registered server paths. The checkpoint/model/data/analysis lock
is frozen before new predictions. No training or checkpoint reselection is allowed.

Seven frozen checkpoints get595 predictions each, using the same35cases and17layouts.
Original MRGR two seeds and uniform fusion two seeds are hash-verified historical
references, not recomputed models. All new predictions are sealed before scoring.
GT128 is always forbidden. The guard permits testGT64 only after the complete seal.
The original predictor and scoring functions are reused without changing numerics.
Four GPUs are selected dynamically by UUID, shared flock and repeated no-process,
free>=20000MiB, used<100MiB/util<=5% checks15seconds apart plus launch recheck.

All11models/13090fields are independently recomputed with NumPy. Case-paired
bootstrap20000 seed20261020 includes clustered draws averaged within each case;
per-seed and descriptive two-seed comparisons are retained. Single-seed feedback
and reused35-case test data limit interpretation. This is supplementary evaluation,
not a new independent holdout confirmation. Frozen legacy sources/results are intact.
