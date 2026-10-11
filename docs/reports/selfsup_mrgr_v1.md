# Self-supervised MRGR v1

Decision: **SelfSup-MRGR v1 GO**.

This is an engineering viability GO under the predeclared evaluation references. SelfSup C1 voxel relative L1 is 0.096193/0.090946 on Val-A/Val-B, versus supervised MRGR 0.067980/0.060948 (41.50%/49.22% higher error). Clustered V4 improves from 0.495210 to 0.393402 (20.56% lower error); this single-seed result does not establish superiority or supervised parity. The architecture and v1 objective remain frozen.

Training numerical source commit: `3f76c3fe3cb17f75e4244bf67424b00355eca6bf`. Post-hoc evaluation source commit: `3f76c3fe3cb17f75e4244bf67424b00355eca6bf`. First permitted GT evaluation: `2026-10-12T04:41:49.294340+08:00` (Asia/Shanghai), after all 300 epochs and the checkpoint SHA256 freeze.

## 1. Objective

Self-supervised training without volumetric ground truth. The network is trained only from multi-view TDLAS measurements and acquisition geometry. Volumetric CFD fields are used exclusively for post-training evaluation.

## 2. Strict no-volumetric-GT protocol

Independent MeasurementOnlyPoolDataset/DualMeasurementDataset use the existing frozen geometry, statistics and view-field helpers while independently loading projection NPY only. The supervised dataset is unchanged. No GT64/GT128 key exists in any A/B/H sample or batch. Runtime open audit, inherited by forked workers with shared counters, denies GT64/GT128, foreign checkpoints, all non-Pool-A train/val projection NPY and Pool B geometry. Dataset works with absent GT files.

```json
{
  "gt_file_opened": false,
  "gt_open_attempts": 0,
  "forbidden_npy_attempts": 0,
  "supervised_checkpoint_open_attempts": 0,
  "projection_open_count": 182124,
  "unique_pool_a_projection_files": 202,
  "policy": "Only Pool A train/val projection NPY; GT64/GT128 and foreign checkpoints denied",
  "audit_hook_inherited_by_workers": true
}
```

## 3. Frozen MRGR architecture

Exactly GeoLift-OARI-MRGR-Res3D, 332,070 parameters. Encoder, lifting, OARI, residual branch, GCA, decoder and 64³ output remain unchanged. Signed input residual R=Y_input−P_diff(C0,input), fixed scale32284.894580078173, no clipping; C0 is not detached. No supervised initialization or third module.

## 4. Dual input subsets A/B

Each train case draws A and B independently sized3–5 with pairwise-disjoint station selections. Stateless blake2b/SeedSequence uses base seed, epoch, sample_index and case_id, without worker_id, Python hash or global RNG. Gate holds epoch0 fixed; formal train varies epoch.

## 5. Held-out measurements H

Two common Pool A held-out stations per training case, disjoint from A and B. H is used only as a target; the network receives A or B separately. Validation uses fixed disjoint A4/B4/H4.

## 6. Robust measurement discrepancy

M(C,G,Y)=mean per-case valid-ray SmoothL1(P_diff(C,G)/Y_scale,Y_raw/Y_scale,beta=.01)+0.1×valid-ray sample-relative-L1. K96, all rays, frozen projector, chunk2. Invalid/padded rays contribute exactly zero; valid Y=0 remains valid. Reference and geometry are detached. Soft measurement error does not assert exact data equality or a theoretical representation floor.

## 7. Cross-view prediction

L_cross=.5[M(C_A,B,Y_B)+M(C_B,A,Y_A)]. The reconstruction network never sees the opposite subset; only the projector uses its target geometry. Cross-view geometry and target pairing are unit-tested.

## 8. Cross-input 3D consistency

L_cons=mean_batch(mean|C_A−C_B|/(.5(mean|C_A|+mean|C_B|)+1e-6)). Both volumes are differentiable; there is no GT reference or detach. No TV, sparsity, mass prior or PDE term.

## 9. Full self-supervised objective

L_total=1.0 L_hold+0.5 L_cross+0.1 L_obs+0.05 L_cons; L_hold=.5[M(C_A,H)+M(C_B,H)], L_obs=.5[M(C_A,A)+M(C_B,B)]. These weights are frozen; no warmup, weight search or supervised auxiliary objective.

## 10. GT-free engineering gate

One fresh random 8-train-case, fixed epoch0 A/B/H, 80-epoch gate. Held-out relative L1/L2: initial 377.35615444/95.26291573, final 0.52050107/0.16415947. Zero-volume L1/L2: 1.00000000/1.00000000. Relative L1 reduction 99.862%. Passed True. Collapsed False. No volumetric metrics were computed.

Losses hold/cross/observed/consistency and total, output min/max/mean/std/saturation, case-to-case volume difference, gradient and set behavior are recorded in gate_summary.json and metrics.jsonl. Only measurement-domain and numerical behavior inform the gate.

## 11. Measurement-only validation and model selection

36 Val-A cases have one frozen epoch0 A4/B4/H4 station list, stored explicitly. The sole selection quantity is the mean of the two per-case held-out relative L1 predictions. L2/cross/observed/consistency are supporting logs. Pool B and GT never participate. This measurement-only validation partition differs from the legacy fixed-V4 station set used later for paired scientific voxel evaluation.

## 12. Formal training

Fresh random initialization (neither supervised nor gate weights), 166 Pool A train / 36 Val-A cases, all 300 epochs, seed20261011, FP32, effective batch2. AdamW lr2e-4/wd1e-4, clip1, cosine T_max300, no early stopping. Best epoch 278 (1-based); selected exclusively by mean held-out measurement relative L1. Mean epoch 32.993 s; peak allocated 7.668 GiB. Checkpoint `/data/djc/GeoPhy-Set3D/outputs/checkpoints/selfsup_oari_mrgr_v1/best.pt`, SHA256 `1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026`.

## 13. Post-hoc GT evaluation

After full300 and measurement-only selection, the best checkpoint hash is frozen in frozen_checkpoint.json. Independent evaluate_selfsup_with_gt.py verifies provenance before accessing a volume-loading dataset. posthoc_gt_boundary.json records the first permitted GT evaluation. No architecture, loss, checkpoint or sampler adjustment follows this GT access. All voxel metrics are evaluation only and never influence optimization or selection.

## 14. Val-A / Val-B

| Condition | Supervised OARI L1/L2 | Supervised MRGR L1/L2 | SelfSup C0 L1/L2 | SelfSup C1 L1/L2 |
|---|---:|---:|---:|---:|
| Val-A-V4 | 0.09369431/0.10225217 | 0.06798015/0.07770150 | 0.12247507/0.17234965 | 0.09619263/0.16484722 |
| Val-B-V4 | 0.09065965/0.10135402 | 0.06094801/0.07151314 | 0.11965872/0.17169781 | 0.09094554/0.15998378 |
| V3 | 0.13630230/0.17124648 | 0.09830877/0.12234692 | 0.15342572/0.21407507 | 0.11836432/0.18894760 |
| V4 | 0.09713875/0.11350930 | 0.06692271/0.07829088 | 0.12352862/0.17659003 | 0.09564931/0.16481803 |
| V6 | 0.08006464/0.08762266 | 0.05340104/0.06166384 | 0.11323223/0.16749432 | 0.08805847/0.15954573 |
| clustered-V4 | 0.56504478/0.72246884 | 0.49520997/0.66506829 | 0.52982818/0.60489396 | 0.39340239/0.52929375 |
| spread-V4 | 0.08994035/0.10480935 | 0.05891869/0.06964059 | 0.11951395/0.17101344 | 0.08764829/0.15746743 |


## 15. V3/V4/V6

Five exact deterministic draws each, identical case/station pairs to frozen OARI/MRGR. Per-case/draw stage data and pooled metrics are retained. Repeated draws of the same cases are not independent; spread V4 draw0 is supplemental.

## 16. Clustered V4

Five paired clustered draws are in the result tables and stage artifacts. Compare the scientific gap candidly; measurement generalization alone does not prove the correct 3D field.

## 17. Measurement-domain generalization

| Measurement evaluation | Held-out L1/L2 | Cross-view L1/L2 | Observed L1/L2 |
|---|---:|---:|---:|
| Val-A | 0.03275179/0.04244088 | 0.03239258/0.04190874 | 0.02557021/0.03379197 |
| Pool-B | 0.03322154/0.04387108 | 0.03255909/0.04260653 | 0.02548844/0.03431086 |


## 18. Gap to supervised MRGR

| Condition | SelfSup minus supervised MRGR L1 | SelfSup / supervised L1 |
|---|---:|---:|
| Val-A-V4 | 0.02821248 | 1.415 |
| Val-B-V4 | 0.02999754 | 1.492 |
| V3 | 0.02005555 | 1.204 |
| V4 | 0.02872660 | 1.429 |
| V6 | 0.03465743 | 1.649 |
| clustered-V4 | -0.10180758 | 0.794 |
| spread-V4 | 0.02872959 | 1.488 |


## 19. Limitations

Saved measurement targets were historically generated from CFD volumes, but the training process never reads those volumes. The finite 64³/K96 model and 128³/K256 references need not agree exactly. Measurement agreement and cross-input consistency do not establish a unique/correct volumetric field. One seed and fixed v1 losses limit generality; no tuning, third module, full ablation or final test evaluation was conducted.

## 20. GO / NO-GO

**SelfSup-MRGR v1 GO**

```json
{
  "strict_gt_free_training": true,
  "gate_passed": true,
  "val_heldout_better_than_zero": true,
  "unseen_geometry_heldout_better_than_zero": true,
  "posthoc_val_a_voxel_better_than_zero": true,
  "posthoc_val_b_voxel_better_than_zero": true,
  "output_not_collapsed": true,
  "set_and_feedback_checks": true
}
```

The operational decision uses measurement-domain improvement over zero, non-collapse, strict provenance, and post-hoc Val-A/Val-B voxel errors below the zero-volume relative-L1=1 reference. These evaluation references were frozen before GT access and are not checkpoint selectors or a claim of supervised parity.

```json
{
  "dataset_files": 1262,
  "dataset_unchanged": true,
  "all_historical_artifacts_unchanged": true,
  "protected_code_unchanged": true,
  "mrgr_v2_unchanged": true,
  "oari_unchanged": true,
  "b0_b1_b2_unchanged": true,
  "b3_unchanged": true,
  "projector_unchanged": true,
  "projection_files_unchanged": true,
  "gt_files_unchanged": true,
  "source_gt128_metadata_unchanged": true,
  "source_gt128_hash_unchanged": true,
  "source_gt128_check": "sha256 against frozen generation index",
  "gt64_check": "sha256",
  "pool_b_training_unused": true,
  "test_used": false,
  "hash_audit_outside_training_process": true,
  "numerical_training_sources_unchanged": true,
  "passed": true
}
```

Artifacts: `/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1`. Tests passed: 158.
