# Residual-feedback OARI pilot

Use `/scratch/djc/envs/gas-torch/bin/python scripts/feedback_oari_pilot_v1/controller.py`
from this independent worktree. The controller reserves at most one dynamically
selected UUID through the same `/scratch/djc/GeoPhy-Set3D/gpu_locks` directory as
the ongoing C/D pruning experiments. A reservation gives our small pilot priority
within our queues; it never preempts, pauses or signals any running process.
Actual GPU work starts only after repeated process/memory/utilization checks show
an idle device. CPU checks start while waiting. All child tasks have their own
logs, state files and resource snapshots; failures stop dependent work.

Three fresh same-seed runs train for exactly20 epochs each on all166 training
cases and36 validation cases. The original cosine300 schedule is retained; this
is an early-training prefix, not a300-epoch result or final model comparison.
The variants are original MRGR, an independent OARI inserted before the output
convolution of the residual adapter, and the same augmented architecture with
the measurement residual input set to zero. Internal P(C0) is retained for all.
All retained initial tensors and RNG streams match the original same-seed model;
the added OARI is copied into independent parameters from its initial OARI.

CPU tests check real64³ stages, range, masks/permutation, physics, initialization
and bit-exact bypass equivalence with a nonzero adapter. GPU tests check the full
self-supervised objective, gradients and cross-process recovery. Only after both
pass does training start. Twenty epochs are matched across all arms. Checkpoint
selection uses H4 only. All17 development layouts are predicted and sealed before
development GT64 scoring; statistics aggregate draws within each case first and
independently recompute saved volume metrics. Test arrays and GT128 are forbidden.

Heavy artifacts are kept at
`/scratch/djc/GeoPhy-Set3D/mrgr_feedback_oari_pilot_v1`; the Chinese report is
`docs/reports/mrgr_feedback_oari_pilot_v1_zh.md` in this worktree. A favorable pilot
only supports allocating a longer matched-budget multi-seed experiment. An
unfavorable early prefix does not establish the converged architecture's value.
