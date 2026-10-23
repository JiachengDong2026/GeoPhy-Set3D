# Plain interaction final model: six paired core ablations

Run from repository root with `/scratch/djc/envs/gas-torch/bin/python`:

1. `scripts/plain_core_ablations_v1/register.py` — refuses existing stage.
2. `scripts/plain_core_ablations_v1/checks.py cpu`
3. `scripts/plain_core_ablations_v1/controller.py` — durable two-batch GPU queue,
   six batch2 gradient/save/reload smoke tests and cross-process epoch recovery,
   six formal300 runs, checkpoint validation, prediction/seal, authorized35 GT64
   scoring, independently verified paired stats, Chinese report and audit export.

Only six registered arms may train. Model dispatch MUST use `common_stage.model_for`
and the arm identity; inherited config model-name is not sufficient. Historical
numerical code, checkpoints and predictions are never edited. Geometry and single
scale definitions are imported directly, while PRF removal calls plain stage0.
All retained initial tensors and post-construction CPU RNG exactly match fresh
same-seed plain baselines. Engine precision/data/loss/budget/selection unchanged.

Scratch: `/scratch/djc/GeoPhy-Set3D/mrgr_plain_core_ablations_v1`.
Protocol: `docs/mrgr_plain_core_ablations_v1_protocol_zh.md`.
Tracker: `docs/reports/mrgr_plain_final_experiment_tracking_v1_zh.md`.
Report is generated only after six full runs and all checks pass.

UUID locks, 15s two-snapshot idle gates, host limits, pinned external workflow
parents and inherited worker lock FDs protect other tasks. Failed tasks stop future
launches; inspect evidence and register same-definition recovery before resuming.
Own worker/loader children alone may be interrupted on external CUDA collision.
Never signal other users' processes. No automatic publishing or remote messages.
