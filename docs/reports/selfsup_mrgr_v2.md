# SelfSup-MRGR v2: Rich-View Teacher Consistency

Decision: **SelfSup-MRGR v2 NO-GO**. Training source commit `adc45de7cdaa0697a56238b56689e921f57c9419`. Tests passed: 179.

## 1. Motivation

Test whether additional real TDLAS views provide a useful GT-free 3D structural target for sparse reconstruction.

## 2. Limitation of SelfSup v1

v1 shows feasible GT-free reconstruction, while measurement compatibility does not establish volumetric correctness. Symmetric sparse A↔B consistency may reinforce a common ambiguous solution.

## 3. Frozen MRGR architecture

GeoLift-OARI-MRGR-Res3D, exactly332,070 parameters. Encoder, lifting, OARI, residual branch, GCA, decoder, 64³ output and projector source remain unchanged.

## 4. Sparse subsets A/B

Each train case draws disjoint A3–5 and B3–5 from Pool A. The v1 stateless seed/case/epoch sampler is reused exactly; formal training varies epoch.

## 5. Rich subset T=A∪B

T6–10 is built before padding from the concatenated station IDs. All measurement/geometry/mask fields are indexed with those same IDs. View order does not change the set prediction. Validation uses T8.

## 6. Held-out subset H

Train H2, validation H4. A/B/H are pairwise disjoint and T excludes H. H is a projector target, never a reconstruction network input. Gate epoch0 stations and all36 Val-A station IDs are saved.

## 7. Rich-view teacher formulation

C_A=F(A), C_B=F(B), C_T=F(T) use one shared network and one optimizer. No EMA, second teacher network, prior or volume label. A reconstruction obtained from the union of two sparse measurement subsets is used as a richer-view pseudo-teacher for volumetric consistency.

## 8. Stop-gradient volumetric distillation

D(X,T)=mean_voxel|X−T|/(mean_voxel|T|+1e-6), computed per sample then batch mean. L_teacher=.5[D(C_A,C_T.detach())+D(C_B,C_T.detach())]. Teacher consistency cannot directly move the C_T branch; its measurement losses remain differentiable. Both numerator and denominator use the detached teacher. No absolute voxel L1 or symmetric A/B loss is optimized.

## 9. Measurement objectives

Frozen v1 M: per-case valid-ray SmoothL1(P_diff(C,G)/32284.894580078173,Y_raw/32284.894580078173,beta=.01)+.1 valid-ray relativeL1. K96, all rays, view_chunk2. L_hold=.5[M(C_A,H)+M(C_B,H)], L_cross=.5[M(C_A,B)+M(C_B,A)], L_obs=.5[M(C_A,A)+M(C_B,B)], L_hold_teacher=M(C_T,H), L_obs_teacher=M(C_T,T).

## 10. Full v2 loss

L_total=1.0 L_hold+.5 L_cross+.1 L_obs+1.0 L_hold_teacher+.1 L_obs_teacher+.1 L_teacher. No weight sweep, warmup, TV, sparsity, mass/PDE/smoothness prior, teacher dropout or architecture change.

## 11. Strict GT-free protocol

Runtime GT/foreign-checkpoint/projection-whitelist audits retain worker-inherited shared counters, with explicit Pool B access counts. Any forbidden read immediately fails. Training and model selection are volumetric-GT-free. Method development has inspected v1 volumetric GT post-hoc results; it is incorrect to claim otherwise.

```json
{
  "gt_file_opened": false,
  "gt_open_attempts": 0,
  "forbidden_npy_attempts": 0,
  "supervised_checkpoint_open_attempts": 0,
  "projection_open_count": 2235,
  "unique_pool_a_projection_files": 8,
  "policy": "Only Pool A train/val projection NPY; GT64/GT128 and foreign checkpoints denied",
  "audit_hook_inherited_by_workers": true,
  "Pool_B_access_attempts": 0,
  "forbidden_projection_attempts": 0,
  "external_checkpoint_attempts": 0
}
```

## 12. Engineering gate

Exactly one fresh-random 8-case / 80-epoch gate, fixed epoch0 A/B/H/T. Initial sparse held-out L1 377.35615548; final 0.51934925; teacher final 0.51135635; zero-volume baseline 1.00000000. Frozen v1 gate reference 0.520501; v1 was not retrained and this is not a formal v1/v2 comparison. First10 / last10 training teacher loss means 0.03606436 / 0.09986418. Output collapsed: False; passed: False.

```json
{
  "all_losses_finite": true,
  "teacher_better_than_zero": true,
  "sparse_A_improved": true,
  "sparse_B_improved": true,
  "student_teacher_loss_decreased": false,
  "output_not_collapsed": true,
  "set_properties": true,
  "teacher_gradients": true,
  "gt_free": true
}
```

The gate failed only the frozen temporal student–teacher loss-decrease condition. It does not establish that v2 voxel quality is worse than v1: the formal comparison was not executed. Failure diagnostics checked all8 cases and120 view fields; A/B union geometry, measurement pairing and masks matched exactly (max difference0). Both student output gradients point toward the detached rich teacher, and teacher measurement gradients remain nonzero. No implementation bug was found; the gate was not rerun and no weights were tuned.

## 13. Measurement-only model selection

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

The only selector is .5[heldout(C_A,H)+heldout(C_B,H)] relativeL1. Teacher measurements, volumetric consistency and GT metrics never select a checkpoint.

## 14. Val-A measurement results

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

## 15. Pool-B measurement results

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

## 16. Post-hoc voxel evaluation

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

All case/station IDs are asserted against both frozen v1 and supervised MRGR, which already pair with supervised OARI. Full300 and checkpoint freeze must precede any GT read. Measurement evaluation finishes first.

## 17. V3/V4/V6

Five exact deterministic draws per condition, using the frozen reference case/station IDs. Repeated draws are not independent samples.

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

## 18. Clustered V4

Five paired draws. Engineering reference relativeL1≤.42 guards against materially losing v1’s clustered advantage.

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

## 19. Teacher-quality diagnostics

GT teacher-quality diagnostic not executed. Measurement-only gate diagnostics compare A/B/T held-out errors and log output min/max/mean/std, fractions near0/1, case differences and D(A,T)/D(B,T)/D(A,B) each epoch.

## 20. SelfSup v1 vs v2

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

The predeclared engineering GO requires Val-A absolute L1 gain≥.003, Val-B relative worsening≤3%, improvements in V3/V6, and clustered L1≤.42. The V3 .115 value is a supporting reference only. Measurement error alone cannot decide success.

## 21. Gap to supervised MRGR

Not executed: engineering gate failed. No formal training or GT post-hoc evaluation.

## 22. Limitations

Rich measurements do not guarantee unique or more accurate 3D reconstruction. Saved projections originate historically from CFD, while training never reads volumetric fields. One seed limits inference. No final test, full ablation or multi-seed was run. The working term RVTC is not a final paper name.

## 23. GO / NO-GO

**SelfSup-MRGR v2 NO-GO**

Retain SelfSup v1. Pause automatic experiments; do not create v3 or add priors/EMA/modules.

```json
{
  "dataset_unchanged": true,
  "all_historical_artifacts_unchanged": true,
  "source_gt128_metadata_unchanged": true,
  "projector_unchanged": true,
  "network_architecture_unchanged": true,
  "v1_training_unchanged": true,
  "numerical_training_sources_unchanged": true,
  "test_used": false,
  "pool_b_training_unused": true,
  "gt_hashes_checked": false,
  "passed": true
}
```
