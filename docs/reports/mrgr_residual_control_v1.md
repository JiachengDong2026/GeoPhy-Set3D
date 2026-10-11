# MRGR matched residual control v1

**Stage A complete. Supervised architecture experiments are closed.**

Evaluation source commit: `3f76c3fe3cb17f75e4244bf67424b00355eca6bf`. Training source provenance is retained in the checkpoint, training logs and numerical source manifest; a later documentation commit does not change the experiment.

## Question and matched design

Does the signed measurement residual supply information beyond the added refinement capacity? The zero-residual control has exactly the same 332,070 parameters, parameter names/order, modules and same-seed initial tensors as full MRGR. The scalar channel is identically zero; valid-ray and path channels, residual encoder, adapter, shared GCA and decoder remain. Control training/inference does not invoke a projector. This is a separately trained from-scratch control, rather than a shortcut on full MRGR weights.

## Frozen training protocol

166 Pool A train cases, V3–6; 36 fixed-V4 Val-A cases. FP32, batch2, seed20261011; AdamW lr2e-4/wd1e-4, clip1, cosine T_max300; all 300 epochs, no early stopping. The objective remains SmoothL1(C1,GT64,beta=1)+0.1 sample-relative-L1. Best epoch 278 (1-based), selected only by Val-A supervised loss. Initial weights and data sampler match full MRGR exactly. All model-selection station IDs match. Pool B is evaluated only after checkpoint freeze; test is unused.

Control checkpoint `/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_oari_mrgr_zero_residual_control_300/best.pt`. SHA256 `f7db3472ede967856aa6e976ee25b323df3d4c661ad4c6e6f15fdd4692bb430d`.

## Paired results

| Condition | Frozen OARI L1 / L2 | ZR control L1 / L2 | Full MRGR L1 / L2 | Full gain vs control L1 |
|---|---:|---:|---:|---:|
| Val-A-V4 | 0.09369431 / 0.10225217 | 0.09462988 / 0.10254914 | 0.06798015 / 0.07770150 | 0.02664973 |
| Val-B-V4 | 0.09065965 / 0.10135402 | 0.09205316 / 0.10245256 | 0.06094801 / 0.07151314 | 0.03110515 |
| V3 | 0.13630230 / 0.17124648 | 0.13733413 / 0.17139810 | 0.09830877 / 0.12234692 | 0.03902536 |
| V4 | 0.09713875 / 0.11350930 | 0.09746792 / 0.11318981 | 0.06692271 / 0.07829088 | 0.03054521 |
| V6 | 0.08006464 / 0.08762266 | 0.08013981 / 0.08829062 | 0.05340104 / 0.06166384 | 0.02673877 |
| clustered-V4 | 0.56504478 / 0.72246884 | 0.58320254 / 0.73682240 | 0.49520997 / 0.66506829 | 0.08799257 |
| spread-V4 | 0.08994035 / 0.10480935 | 0.08851746 / 0.10078374 | 0.05891869 / 0.06964059 | 0.02959877 |

Means are per-case; V3/random V4/V6/clustered V4 aggregate five exact deterministic draws, spread V4 uses draw0. All station/case pairings are asserted against frozen full MRGR. MAE/RMSE and complete per-case/draw metrics are in the external artifacts. Frozen OARI/full MRGR were neither retrained nor modified.

## Attribution

signed measurement residual provides information beyond the added refinement capacity.

For a descriptive reference, full MRGR must improve relative L1 by at least 0.003 in each of Val-A, V3 and clustered V4. This reference is an engineering interpretation, not a statistical significance threshold. Supported: True. No extra control, encoder/adapter sweep or further supervised experiment is introduced regardless of this outcome.

## Engineering and integrity

Training mean epoch 10.698 s; peak allocated memory 5.439 GiB. Tests passed: 137. The GPU set checks are:

```json
{
  "permutation_max_diff": 2.980232238769531e-07,
  "padding_max_diff": 2.384185791015625e-07,
  "visibility_identical": true,
  "all_invalid_safe": true,
  "passed": true
}
```

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

Single training seed and repeated case/draw samples limit attribution strength. No uniqueness or cross-seed stability claim is made. Artifacts: `/data/djc/GeoPhy-Set3D/outputs/comparisons/mrgr_residual_control_v1`.
