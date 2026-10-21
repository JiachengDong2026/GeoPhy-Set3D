# Observability / reconstruction error pairing v1

This is an offline, CPU-only analysis of already frozen predictions. All four
checkpointed models use **Uniform Fusion**. There is no training or inference
entry point. GT128 is always denied by the existing AccessGuard.

Canonical results are in
`docs/reports/artifacts/mrgr_observability_error_pairing_v1/`; report:
`docs/reports/mrgr_observability_error_pairing_v1_zh.md`.

Execution order, using `/scratch/djc/envs/gas-torch/bin/python`:

1. `analyze.py audit`: strict checkpoint loading in isolated subprocesses;
   verify historical seals, 490 prediction hashes/shapes/ranges, and reconstruct
   all 70 measurement-only input hashes. Register immutable design and seal.
2. `analyze.py historical`: extract matched draw-0 scalar metrics already scored
   by the official evaluation. This does not read GT.
3. `score.py authorize`: record the explicit in-session user authorization
   received on 2026-10-21. **This command is not itself an authorization source**;
   it was executed only after the user's explicit reply. Do not run it in a new
   scope or infer permission from historical GT access.
4. `score.py score`: after the authorization/identity/seal checks, read exactly
   the 35 registered GT64 arrays once and call the original volume and morphology
   functions. Verify all 560 archived baseline/structural fields exactly.
5. `score.py summaries`: 20,000 case-paired bootstrap resamples, seed 20261025;
   per-layout, per-stage, per-seed summaries, prediction changes and correlations.
6. `verify.py`: independently reconstruct pairing, case-count weighted bootstrap
   and sorted-percentile interpolation, and recompute D by vector norms without
   reading GT. Historical scoring code identity is checked through the preserved
   snapshot and the narrowly scoped statistical export amendment.
7. `report.py`: Chinese report, two scatter panels and acceptance manifest.

Outputs refuse overwrite. To independently recompute in another directory,
import `analyze` first, set its `ART` to a fresh absolute directory, then import
`score` so that it inherits that directory; invoke the audit/historical functions
and the explicitly authorized score/summaries functions in that order. Bind any
new execution to an authorization record for precisely that scope. Preserve old
results. The verification script expects `statistics_export_amendment.json` only
in this original execution, where an export-only repair was necessary.

Two recoveries were retained. An initial audit looked for a sidecar in the repo
export; the authoritative scratch registry has the sidecar and was used instead.
The first statistics export encountered a constant peak-location delta, making
correlations undefined. The original source/outputs are preserved under
`statistics_attempt01/`; the amended code emits null with a reason. No metric,
hypothesis, input, seed, bootstrap rule, prediction, or target score changed.

`D` uses the requested additive epsilon 1e-6; `D_legacy` separately preserves the
old max-denominator definition. Their difference is negligible here, but the two
definitions are recorded explicitly. Centroid definitions and Dice thresholds
are also kept distinct rather than silently interchanged.
