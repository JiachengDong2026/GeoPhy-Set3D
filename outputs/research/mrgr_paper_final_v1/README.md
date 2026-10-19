# MRGR final35: published experiment evidence

This Git snapshot contains scalar results, fixed-case figures, registration and
identity records, training histories, software recovery records, and final
acceptance evidence. The full local run additionally contains datasets, weights,
prediction arrays, recovery tensors, and resource snapshots; those large or
machine-specific artifacts are deliberately excluded from Git.

Start with [the Chinese final35 report](../../../docs/reports/mrgr_paper_final_test35_zh.md),
[final acceptance](final_acceptance.json), and
[the independent verification](evaluation/test/independent_verification.json).
The publication file list and SHA256 identities are in
[github_publication_manifest.json](github_publication_manifest.json).

The original final method lock remains unchanged. A seal-only recovery adds its
already registered checkpoints to the read allowlist; see
[the amendment](recovery/seal_checkpoint_allowlist_v2/amendment.json).
The failed original seal log remains available at
[stages/final_all_seal.log](stages/final_all_seal.log).

Execution records retain original absolute paths and checkpoint/array hashes as
provenance. Cloning this snapshot permits inspection and reanalysis of scalar
results; full array verification or rerunning models also requires the local
data and weights. Dependencies are described in
[requirements-research.txt](../../../requirements-research.txt), with workflow
entry points in [scripts/paper_final_v1](../../../scripts/paper_final_v1).

The primary endpoint is ordinary fixed V4 C1 3D relative L1 across 35 cases.
Bonferroni intervals cover all 20 registered comparators. Five draws are averaged
within each case; the two training seeds do not estimate population seed variance.
SART's linear nodal-mole-fraction operator and CPU runtime scope are explicitly
separate from the common nonlinear projector used for evaluation.
