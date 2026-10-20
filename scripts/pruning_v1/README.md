# Development-only structural pruning experiment

Use `/scratch/djc/envs/gas-torch/bin/python` from this worktree.

`python scripts/pruning_v1/pipeline.py` pre-registers the fixed design, audits
six reusable historical development runs, executes real CUDA correctness and
cross-process recovery checks, and schedules four full-budget trainings. GPU
assignments are discovered at runtime and bound by UUID. Every task has its own
console log and status record. A failed task stops dependent work and is never
silently restarted. Other users' processes are never signalled.

The pipeline then validates selected checkpoints, predicts all 36×17 development
inputs for each new run, seals all predictions before scoring development GT64,
benchmarks A/B/C/D using the same inference entry, computes paired case bootstrap
statistics and the factorial interaction, independently recomputes saved volume
metrics, and renders a Chinese report plus fixed-case figures. It never reads
test projection arrays, test GT64 or any GT128. A data access audit enforces this
boundary before file contents can be opened.

The original training, sampling, losses, physics, recovery and evaluation engines
are imported from the immutable historical checkout, with their full SHA256
mapping recorded in each protocol. Only process-local dispatch and output paths
are adapted. Retained initialization tensors are checked against a freshly built
same-seed full model, including exact column slices for the two reduced layers.

All heavy artifacts live at
`/scratch/djc/GeoPhy-Set3D/mrgr_gca_uniform_observability_pruning_v1`.
The report lives in this worktree at
`docs/reports/mrgr_gca_uniform_observability_pruning_v1_zh.md`.
The pipeline is deliberately development-only; the already-used final35 dataset
cannot serve as independent confirmation of these new structures.
