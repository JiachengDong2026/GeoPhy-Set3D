# OARI directed experiments v1

Three independent fresh 300-epoch / 24,900-update trainings:

- `uniform_no_ray_relations_seed20261019`
- `uniform_plain_set_interaction_seed20261011`
- `uniform_plain_set_interaction_seed20261019`

The fourth experiment reuses the official seed-20261019 Uniform Fusion checkpoint
for 70 frozen `obs_token` predictions. It loads the **unchanged** OARI v2 hook.
Two fixed no-op forwards check ordinary/clustered historical prediction agreement.
No frozen-checkpoint training is performed.

The stage is isolated under `/scratch/djc/GeoPhy-Set3D/mrgr_oari_targeted_v1`.
The output root is derived from the audited canonical stage. Historical model,
training and evaluation sources are imported without file edits. The new model
is an opt-in variant; default builders and finalized models retain their behavior.

**Model construction must dispatch by the registered arm through
`variants.model_for`; do not construct from the inherited config model-name
alone.** The underlying config deliberately retains the formal training fields.
The plain variant has 342,209 parameters, uses the original MRGR forward through
C3 MRO, and supplies a compatible `context_features`/`visibility` dictionary.
Plain attention returns no invented OARI bias. Its 48-input projection is fresh;
compatible sublayers are initialized from a fresh same-seed control, never from
a trained checkpoint. All non-interaction initialized tensors match exactly.

Run with `/scratch/djc/envs/gas-torch/bin/python` from repository root:

1. `scripts/oari_targeted_v1/stage.py register`
2. `scripts/oari_targeted_v1/checks.py cpu`
3. `scripts/oari_targeted_v1/controller.py`

The controller obtains an exclusive UUID lock, verifies two idle snapshots 15s
apart and passes the lock FD to each worker. GPU smoke checks precede training.
At most two jobs launch initially; four are allowed only after both first jobs
have completed at least three epochs and the existing host-resource gate passes.
GPUs with other compute processes are never used, even when utilization is low.

Workers reuse the original numerical training engine, AdamW/cosine/FP32,
full train/validation datasets and H4-only checkpoint choice. `last.pt` is the
canonical resumable epoch commit (optimizer, scheduler, RNG, metrics included).
The controller does not automatically replay failed tasks; it retains evidence
and blocks the next phase. Recovery guards skip accepted completed outputs.

An audited recovery was needed for a GPU1 fold-transition race: another user's
already-started next-fold process created its CUDA context after both idle
snapshots. Only this stage's seed19 plain worker and its data-loader children
were interrupted/cleaned up. Its authoritative epoch4 / 332-update commit,
optimizer/scheduler/RNG and original logs were preserved under `recovery/`.
The recovery supervisor waits for the original two siblings and then execs the
controller; `scheduling/resource_exclusions.json` excludes GPU1/2 for the rest
of this stage to avoid context-free gaps in those foreign multi-fold workflows.
The canonical engine resumes the same run on GPU0/3. All four earlier committed
metric rows must match the pinned interrupted checkpoint, and a canonical resume
event is required before prediction. Budgets refer to committed updates; partial
uncommitted epoch work is replayed. No foreign process was signalled.

After all three trainings and the frozen probe finish, the controller generates
210 predictions per trained model (ordinary draw0, clustered draw0..4), benchmarks
all six trained/control methods on one GPU and three fixed validation inputs,
then calls `evaluate.py`. All 700 new predictions and 700 matched reused files
are identity/hash/shape/range checked and sealed **before current-phase GT64**.
GT128 and training-time GT64 are always denied. Permission for this bounded
evaluation is recorded from the user's current explicit instructions, not
inferred from an unrelated historical stage.

The analysis calls the original volume/morphology implementations. Statistics
use 35 paired cases, 20,000 bootstrap resamples, seed 20261026. Clustered five-draw
means and draw0 analyses are reported separately. Both training seeds are shown
individually; case intervals are never training-seed population intervals.
An independent case-count-weighted/bootstrap-percentile calculation verifies
all summary vectors. The report does not automatically promote any model or
claim equivalence from an interval crossing zero.

Completed audit/statistical artifacts are exported to
`docs/reports/artifacts/mrgr_oari_targeted_v1/`, with the Chinese report at
`docs/reports/mrgr_oari_targeted_v1_zh.md`. Prediction arrays and checkpoints stay
in scratch with SHA256 identities in the repository.
