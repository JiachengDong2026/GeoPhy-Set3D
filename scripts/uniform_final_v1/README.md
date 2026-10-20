# MRGR-UniformFusion final candidate v1

Protocol: `docs/mrgr_uniform_final_v1_protocol_zh.md`.
Frozen model configuration: `configs/mrgr_uniform_fusion_v1.json`.
Use `/scratch/djc/envs/gas-torch/bin/python` in the main worktree.

```bash
python scripts/uniform_final_v1/register.py
python scripts/uniform_final_v1/controller.py
```

Registration is metadata-only and verifies historical identities before training.
The controller runs CPU/GPU engineering gates and full recovery checks, then the
currently included training jobs. The original14-run design is preserved. Two
append-only amendments first cancelled3 supplementary second seeds, then deferred
all7 unstarted ablation runs at the user's request. The active stage contains only
the4 uniform/OARI jobs (seeds20261020/20261021). Initial concurrency is two; promotion to four needs
host and real-run audits. GPU UUID locks and repeated idle checks are mandatory.
The OARI source is separately committed on `experiment/uniform-feedback-oari-v1`
in `/home/djc/projects/GeoPhy-Set3D-uniform-oari-v1`; it is never added to the main
model. Data/large artifacts stay on the original data roots and scratch.

No test GT is opened before all4 included training runs end and all their
predictions are sealed. Deferred ablations do not automatically restart. Five
unperformed ablation comparisons remain missing in the original14-comparison
Bonferroni family; its denominator is not reduced. Final OARI model adoption is
pending user review, even after stage acceptance and GitHub sync.
This reuses the previously used 35-case test set; it is supplementary evaluation,
not new independent confirmation. GT128 is always forbidden. Checkpoint choice
uses validation projection H4 only. The deployment candidate is fixed to the first
historical uniform seed, not the best test-ranked seed; all four seeds are reported.

After training, explicit steps (normally performed by the controller):

```bash
python scripts/uniform_final_v1/benchmark.py
python scripts/uniform_final_v1/evaluate.py register
python scripts/uniform_final_v1/evaluate.py predict --key uniform_seed20261020
# Repeat predict for every new method in final_lock.json, before score.
python scripts/uniform_final_v1/evaluate.py score
python scripts/uniform_final_v1/finish.py
```

Reproduction after the evaluation lock exists, without GT access:

```bash
python scripts/uniform_final_v1/infer.py --case-index 0 --layout-index 0 \
  --device cpu --output /scratch/djc/GeoPhy-Set3D/mrgr_uniform_final_v1/reproduce/case0.npz
```

Existing accepted outputs are immutable. On failure inspect preserved logs and
protocol hashes; create an explicit recovery approval for the affected task before
resuming from its complete epoch checkpoint. Do not change the frozen model,
precision, loss, data stream or total budget. File hashes include original server
paths and dependencies; porting to another machine requires a new registration.
