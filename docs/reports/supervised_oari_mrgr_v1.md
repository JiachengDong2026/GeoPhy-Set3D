# Supervised OARI-MRGR v1

Decision: **MRGR GO**. Training source `2d6e22a5a129d1db860684054f72737ddd0e1a20`. Best checkpoint `/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_oari_mrgr_300/best.pt`, SHA256 `4c9a39fd39432988cf5971b9a657a4cf22180650d3490bf65f3a8a7aeb04905c`.

## 1. Motivation

The initial 3D estimate is reprojected into the measurement domain, and the signed measurement residual is geometrically lifted back into the volumetric feature space to refine the reconstruction. The question is whether explicit measurement feedback improves voxel reconstruction, especially with sparse or clustered views.

## 2. Why additive projection loss was insufficient

Frozen B3 is NO-GO. B3 converts discrepancy into a scalar loss for gradient supervision. MRGR converts the same discrepancy into signed spatial features, lifts them through actual acquisition geometry, and corrects context features. This experiment adds no measurement loss.

## 3. Frozen OARI backbone

OARI source, config, checkpoint, report and artifacts remain unchanged. Frozen OARI comparison summary SHA256: `b9366c6e2e1fd475cef119a02e434451333b688be507a7e963ef50af2dd2835a`; OARI checkpoint SHA256: `a91fcdf200b034cc0e25a2687f61bb5bf8bdfdacb18441b4c1c1ea60e766a680`. Integrity checks cover all historical outputs and protected code.

## 4. Initial reconstruction C0

The unchanged shared encoder → geometry lifting → OARI interaction → GCA → decoder computes C0. The MRGR formal model uses fresh random initialization. Shared same-seed state tensors match OARI exactly; C0 is an intermediate of the jointly trained MRGR and is not the frozen OARI prediction.

## 5. Differentiable measurement residual

Y0=P_diff(C0,G_input), K=96. The unchanged projector uses midpoint integration, XYZ→ZYX grid_sample coordinates, align_corners=True, mass-to-mole conversion and raw ppm·m units. R=Y_raw−Y0 and R_scaled=R/32284.894580078173. No clipping or per-view/per-case normalization. Invalid rays/padded views are zero; valid zero measurements remain valid. Measurements and geometry are detached; C0 remains connected. Signed scaled residual min/max/mean/std/p95/p99 are logged each epoch over all valid rays, with population std and signed quantiles.

## 6. Residual 2D encoder

Input [R_scaled, valid_ray, path_length/(√3×0.6)] has shape [B,V,3,64,64]. Conv3×3 3→16, GN4, SiLU, two 16-channel residual blocks (GN4), no downsampling. Output [B,V,16,64,64].

## 7. Geometry-aware residual lifting

The exact same GeometryLifting instance projects residual features into [B,V,16,32,32,32]. Both visibility tensors match exactly; no new coordinate convention or attention module.

## 8. Residual context refinement

Concatenate context0 (32) and residual lifted features (16); adapter Conv3d1×1 48→32, SiLU, Conv3d1×1 32→32. The final layer starts at zero. delta_context is masked by visibility AND view_mask; invisible context1 is exactly zero.

## 9. C0 → C1 formulation

context1=context0+masked_adapter([context0,R3D]); C1=shared_decoder(shared_GCA(context1)). Exactly one refinement iteration, one GCA module and one decoder. Initial C1=C0 exactly. The only objective is SmoothL1(beta=1)+0.1×sample-relative-L1 on C1 versus GT64; no C0 auxiliary loss, projection, held-out, TV, sparsity or PDE term.

## 10. Parameter / compute cost

OARI 319,558; MRGR 332,070; increase 12,512 (3.915%). MRGR peak allocated 4.342 GiB; mean epoch 16.277 s. OARI observed peak 2.658 GiB and mean epoch 8.906 s. These runs are not a controlled throughput benchmark. View chunks of 2 retain all rays/K96. FP32 on GPU 1.

## 11. Quick gate

Initial C0/C1 L1: 0.03683891 / 0.03683891. Final: 0.03683891 / 0.02946851. C1 supervised loss: 0.00368395 → 0.00294689. Passed: True. All shared weights unchanged.

One fixed-V4 4-train-case run, 60 epochs, AdamW at the frozen lr/wd, only encoder/adapter trained. Frozen OARI loaded only for this functional test. No gate weight was used for the formal run.

## 12. Val-A

Formal protocol: 166 Pool A train cases, V3–6; 36 fixed-V4 Val-A cases, seed20261011, FP32, batch2, AdamW lr2e-4/wd1e-4, clip1, cosine T_max300, all 300 epochs, no early stopping. Best epoch 295 (1-based) selected exclusively by Val-A C1 supervised loss. Training validation station IDs match OARI exactly.

| Condition | OARI MAE | RMSE | L1 | L2 | C0 MAE | RMSE | L1 | L2 | C1 MAE | RMSE | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Val-A-V4 | 0.00031895 | 0.00157575 | 0.09369431 | 0.10225217 | 0.00037354 | 0.00190128 | 0.11122778 | 0.12386717 | 0.00022352 | 0.00117953 | 0.06798015 | 0.07770150 |
| Val-B-V4 | 0.00030436 | 0.00154879 | 0.09065965 | 0.10135402 | 0.00035408 | 0.00177927 | 0.10638539 | 0.11664975 | 0.00019848 | 0.00108122 | 0.06094801 | 0.07151314 |
| V3 | 0.00050009 | 0.00280206 | 0.13630230 | 0.17124648 | 0.00058391 | 0.00320995 | 0.15912602 | 0.19594038 | 0.00035757 | 0.00198341 | 0.09830877 | 0.12234692 |
| V4 | 0.00032386 | 0.00174557 | 0.09713875 | 0.11350930 | 0.00037397 | 0.00198014 | 0.11139167 | 0.12805063 | 0.00022097 | 0.00119449 | 0.06692271 | 0.07829088 |
| V6 | 0.00027028 | 0.00134939 | 0.08006464 | 0.08762266 | 0.00031878 | 0.00160555 | 0.09416012 | 0.10392856 | 0.00017672 | 0.00093827 | 0.05340104 | 0.06166384 |
| clustered-V4 | 0.00195527 | 0.01153871 | 0.56504478 | 0.72246884 | 0.00265898 | 0.01246097 | 0.80439077 | 0.79192603 | 0.00170946 | 0.01059054 | 0.49520997 | 0.66506829 |
| spread-V4 | 0.00029659 | 0.00157676 | 0.08994035 | 0.10480935 | 0.00035131 | 0.00193291 | 0.10631253 | 0.12703848 | 0.00019135 | 0.00105583 | 0.05891869 | 0.06964059 |


## 13. Val-B

Pool B was evaluated only after all 300 epochs and the best checkpoint had been frozen. Exact OARI case/station pairs are asserted. No test split was used. See the paired metrics above.

## 14. V3/V4/V6

Five deterministic draws each, identical to frozen OARI. Tables report means over case/draw samples, not pooled errors; pooled metrics and draw standard deviations are retained in summary.json. Repeated cases across draws are not independent. Spread V4 draw0 is supplementary.

| Condition | OARI minus C1 L1 | C0 minus C1 L1 | Relative gain vs OARI % |
|---|---:|---:|---:|
| Val-A-V4 | 0.02571416 | 0.04324764 | 27.445 |
| Val-B-V4 | 0.02971164 | 0.04543738 | 32.773 |
| V3 | 0.03799353 | 0.06081725 | 27.874 |
| V4 | 0.03021603 | 0.04446896 | 31.106 |
| V6 | 0.02666361 | 0.04075908 | 33.303 |
| clustered-V4 | 0.06983481 | 0.30918079 | 12.359 |
| spread-V4 | 0.03102166 | 0.04739384 | 34.491 |


## 15. Clustered V4

| Draw | OARI L1 | MRGR C0 L1 | MRGR C1 L1 | OARI minus C1 |
|---|---:|---:|---:|---:|
| 0 | 0.71394546 | 0.92384384 | 0.57672886 | 0.13721660 |
| 1 | 0.52518876 | 0.76511030 | 0.46892625 | 0.05626251 |
| 2 | 0.55008861 | 0.81795704 | 0.49994765 | 0.05014096 |
| 3 | 0.56668564 | 0.83835346 | 0.51364090 | 0.05304474 |
| 4 | 0.46931545 | 0.67668920 | 0.41680620 | 0.05250924 |

Five exact deterministic clustered draws. Per-draw metrics are in clustered_v4.csv; all 180 case/draw rows per model are in clustered_v4_per_case.csv. Six compact diagnostics (indices0/18/35 × random/clustered draw0) record station IDs, raw measured/predicted maps, signed residual, stage voxel errors and feature/delta magnitudes. No full training volumes are exported.

## 16. C0 vs C1

The stage contribution must reach ≥0.003 absolute relative-L1 improvement in Val-A, V3 or clustered V4. A lower final error alone does not establish a useful refinement stage. See the stage gains above. C0 L1 exceeds frozen OARI in all seven conditions. Only C1 is supervised, so the initial stage adapts during joint training; the useful correction is demonstrated by the stage gains and residual shortcut probes.

```json
{
  "permutation_max_diff": 2.980232238769531e-07,
  "padding_max_diff": 2.384185791015625e-07,
  "visibility_identical": true,
  "all_invalid_safe": true,
  "prediction_std": 0.011344880796968937,
  "residual_gradient_finite": true,
  "residual_gradient_abs_sum": 0.9114412743074354,
  "projector_feedback_gradient_finite": true,
  "projector_feedback_gradient_abs_sum": 10.07624626159668,
  "passed": true
}
```

## 17. Projection residual before / after

| Condition | C0 observed L1 / L2 | C1 observed L1 / L2 |
|---|---:|---:|
| Val-A-V4 | 0.06702245 / 0.07594510 | 0.03221829 / 0.04402062 |
| Val-B-V4 | 0.06791929 / 0.07631344 | 0.03096208 / 0.04154847 |
| V3 | 0.09786606 / 0.11419107 | 0.04054965 / 0.05307048 |
| V4 | 0.06733235 / 0.07675563 | 0.03171405 / 0.04289987 |
| V6 | 0.05877310 / 0.06777085 | 0.02933769 / 0.04140545 |
| clustered-V4 | 0.56951707 / 0.69880836 | 0.25590852 / 0.32424906 |
| spread-V4 | 0.06484369 / 0.07425187 | 0.03112776 / 0.04278006 |

Observed projection L1 improves in 7/7 conditions: Val-A-V4, Val-B-V4, V3, V4, V6, clustered-V4, spread-V4.  Errors are means of per-case raw ppm·m relative errors. The 64³ prediction and frozen 128³/K256 references have different representation/operator resolutions. No absolute projection threshold or theoretical representation floor is asserted.

## 18. Held-out measurement evaluation

Val-A fixed input V4 plus one deterministic disjoint Pool A held-out station (the existing B3 hold sampler). C0 held-out L1/L2 0.07116949 / 0.07890756; C1 0.04034466 / 0.05024781. The network only receives input V4. Held-out errors never affect training, selection or stopping; they are supporting evidence, not proof of uniqueness.

## 19. Residual shortcut tests

| Residual input | C1 relative L1 | Relative L2 |
|---|---:|---:|
| normal | 0.06798015 | 0.07770150 |
| zero | 0.10938813 | 0.12244277 |
| wrong_pairing | 0.10625450 | 0.12095658 |
| signflip | 0.18026601 | 0.20074724 |

zero: delta L1 vs normal +0.04140799; wrong_pairing: delta L1 vs normal +0.03827435; signflip: delta L1 vs normal +0.11228586. Zero keeps valid/path channels; wrong pairing rolls only real residual maps while geometry remains fixed; signflip negates only residual. These are evaluation-only probes with no strict performance threshold.

## 20. GO / NO-GO

**MRGR GO. Final architecture candidate: v2.**

```json
{
  "val_a_not_materially_worse": true,
  "val_b_not_materially_worse": true,
  "v3_or_clustered_gain": true,
  "refinement_contribution": true,
  "mask_permutation_gradient_connectivity": true
}
```

Engineering criteria are 2% relative main-error tolerance, ≥0.005 absolute V3 or clustered gain versus frozen OARI, ≥0.003 absolute stage gain in a core condition, and correct masks/permutation/gradients. They are operational criteria, not statistical significance claims. The experiment uses one training seed, and clustered C1 L1 remains high at 0.49521. The results do not establish cross-seed stability or inverse-problem uniqueness. No third module, tuning, second seed or multi-step refinement was added.

Integrity:

```json
{
  "dataset_files": 1262,
  "dataset_unchanged": true,
  "all_historical_artifacts_unchanged": true,
  "protected_code_unchanged": true,
  "oari_source_checkpoint_report_unchanged": true,
  "b0_b1_b2_unchanged": true,
  "b3_artifacts_unchanged": true,
  "projector_unchanged": true,
  "numerical_training_sources_unchanged": true,
  "pool_b_evaluation_only": true,
  "test_used": false,
  "passed": true
}
```

Artifacts: `/data/djc/GeoPhy-Set3D/outputs/comparisons/oari_vs_mrgr_300`. See `terminal_report.txt` for all 42 requested terminal items.
