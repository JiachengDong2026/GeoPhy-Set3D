# B0 Generalization Evaluation Protocol v1

## 1. Frozen B0 reference

正式 baseline 仍为 GeoLift-Mean-Res3D / supervised_b0，网络、encoder/lifting/fusion/decoder、
loss、split、Pool A-only training 与归一化保持原样。Training source commit：
`e30bbcd5f2962a509bffe7a24ebdd1db709fc231`。
checkpoint：`/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b0/best.pt`；SHA256：
`7ae74df86022c823123c47668d02fe10ff42c1c6903c0ec06d94c1c3fcabf25e`。best epoch=148（1-based）。
完整 effective config、manifest/split/geometry/statistics hashes 和 model source hashes 在 reference.json。

```json
{
  "dataset_build_manifest.json": "c3e1f7107ae49042b91772101cb8763f85c5911e1520d47559c5058cfbe5f382",
  "train_statistics.json": "7511c5c9035f645838e433db058c4f4c2fc5a95f7dbf03986481da1af62808d2",
  "source_index.jsonl": "cfc482fc84847c90620f0115edafbb12b2ad3cbd2c2ca21b6445ad0025924707",
  "splits/split_summary.json": "364e9f3798f426d22fd4a30c06a7afc509e4e4085f9a78ec10e7c885325041dd",
  "splits/test_cases.txt": "c01cc52c75b5a4326a3faaad2eb1b5aa836ade851e436afc4e67ec4c15be1d61",
  "splits/train_cases.txt": "02bcbd11c86dfad9d369bd0bea4bb97a32ae07b8da46201b6c0823e79db3aa89",
  "splits/val_cases.txt": "7786d70ce7097fb2682aefffe88760cf7d2a6c19b2a5f586f5a92e4ac7c7561e",
  "geometry/canonical_fixed_geometry.npz": "86b2d17db9c105834fc4d96f1694360630881283b94f7e31093925058a8985f6",
  "geometry/ood_geometry.npz": "3c044b86aa39948e78ba4b99b2a68f8c2c7dd8c25a242cb4f7dd8a2e8096fdd9",
  "geometry/train_geometry.npz": "83f583dcc38370e5c66873585c724ab116076d9a9f29f80f674c5c03c96fe3f9"
}
```

reference 首次复现被默认 cuDNN TF32 设置触发误差门槛，后续实验当时停止。
修复为与原评估一致的 FP32、TF32 off、cudnn benchmark off 后，四项 mean/per-case 与四项 pooled
全部精确复现（absolute differences=0），未放宽原先 atol=1e−8、rtol=1e−5。
失败结果完整保留在 reference_reproduction_initial_tf32_mismatch.json 与 plume_generalization/initial_tf32_mismatch/。

## 2–6. Evaluation protocol、Pool定义、view count、draw与覆盖

本轮全部 inference 为 val，model.eval() + torch.no_grad()，GT64 直接在 CH4 mass fraction
物理空间计算，复用 training.metrics.VolumeMetrics，未创建第二套误差公式。
normalization 只使用 frozen train/Pool A p99.5=32284.8945801 ppm·m；
不使用 val/test normalization，不做 per-case 最大值归一化。
Mean per-case 对各 case 的指标平均；pooled 用全体 voxel 误差与GT sums计算。

**Pool B consists of independently sampled unseen station poses from the same defined TDLAS acquisition family.**
Pool A有64 stations，Pool B有32；B使用记录中的 ood_projection_path + ood_geometry.npz，
配对经明确 integration test 检查。准确称呼是 unseen station geometry / in-family geometry generalization，
不称 arbitrary OOD geometry。Pool B 从未用于训练、网络选择或 optimizer 选择。

两个 legacy reference 条件 Val-A-V4 / Val-B-V4 使用原 ViewSampler：seed20261011、epoch0、V4、hold0，
case在val Dataset中的原index参与seed。所有三个训练seed及horizon model selection均固定这同一组Val-A视图。

View-count 条件使用独立 EvaluationViewSampler，V∈[2,3,4,5,6,8]，draw∈[0,1,2,3,4]。
seed仅由 base evaluation seed=20261011、case_id、V、draw_id、pool 组成，
case/pool用稳定BLAKE2b摘要 + NumPy SeedSequence，sample index/worker/全局RNG不参与。
每个subset为V个唯一station；所有per-case/per-draw行记录IDs、azimuths、pool与draw，可精确重放。
原训练 ViewSampler 未修改。独立draw随机子集不要求不同V嵌套。

因此 view-count V4 的五draw均值与 legacy reference V4 属于不同预注册 subset 协议，不能混为同一指标。
训练只见过V=3–6；V=2和8是 cardinality extrapolation。

Angular coverage仅 val/Pool A，V=4/6：

- Spread：含最小/最大frozen azimuth，最大化span；之后 greedy max-min azimuth distance填充。
  每个V仅一个canonical set，全case共享，draw std=0表示只有一组，不是无不确定性。
- Clustered：对azimuth排序，枚举长度V的连续窗口，取span最小的五个窗口，按span、ID排序。
  draw0为全局最小span，其余draw为后续最小窗口，不硬编码station index。
- Random：与view-count完全相同的五个case-specific deterministic draws。

每行记录 min/max/span、largest consecutive sorted azimuth gap（不包括半环边界空隙）、
mean pairwise absolute azimuth distance；角度差是定义域0–180°内的线性距离。
Draw std是5个case-averaged means的 sample std(ddof=1)；case variability单独保存在case_std_per_draw；
seed std是三个独立训练seed的sample std。三者不混用。

## 7–8. Plume与unseen station geometry结果

固定 legacy V4，mean per-case：

| Condition | MAE | RMSE | relative L1 | relative L2 |
|---|---:|---:|---:|---:|
| Val-A-V4 | 0.000459300374 | 0.00238582859 | 0.132893722 | 0.151605587 |
| Val-B-V4 | 0.000429529728 | 0.00242581141 | 0.125969268 | 0.154995709 |

pooled：

| Condition | MAE | RMSE | relative L1 | relative L2 |
|---|---:|---:|---:|---:|
| Val-A-V4 | 0.000459300374 | 0.00263896663 | 0.125037837 | 0.158147457 |
| Val-B-V4 | 0.000429529728 | 0.00267013852 | 0.116933212 | 0.160015519 |


Val-B相较Val-A，relative L1绝对变化=-0.00692445471，
relative degradation=-5.21052%；
relative L2 degradation=2.23615%。
当前单reference的L1略改善、L2略恶化，不把某一有利metric当作全面更优结论。
所有case行在 plume_generalization/Val-A-V4/per_case_metrics.jsonl 与 geometry_generalization/Val-B-V4/per_case_metrics.jsonl。

## 9–10. Variable V、draw variability与cardinality extrapolation

| Condition | V | Mean L1 | Draw std L1 | Mean L2 | Draw std L2 | RMSE | Cardinality |
|---|---:|---:|---:|---:|---:|---:|---|
| Val-A | 2 | 0.3703531 | 0.0295020 | 0.5189983 | 0.0229983 | 0.0087684 | extrapolation |
| Val-A | 3 | 0.2031749 | 0.0222756 | 0.2614774 | 0.0279682 | 0.0043517 | trained range |
| Val-A | 4 | 0.1369043 | 0.0064636 | 0.1625321 | 0.0060683 | 0.0025695 | trained range |
| Val-A | 5 | 0.1181811 | 0.0057357 | 0.1339347 | 0.0086224 | 0.0021026 | trained range |
| Val-A | 6 | 0.1109225 | 0.0021309 | 0.1226630 | 0.0038849 | 0.0019237 | trained range |
| Val-A | 8 | 0.1024933 | 0.0029500 | 0.1137662 | 0.0037881 | 0.0017633 | extrapolation |
| Val-B | 2 | 0.3698618 | 0.0240286 | 0.5105307 | 0.0421650 | 0.0086112 | extrapolation |
| Val-B | 3 | 0.1758126 | 0.0127937 | 0.2280451 | 0.0203138 | 0.0037039 | trained range |
| Val-B | 4 | 0.1294845 | 0.0078913 | 0.1584430 | 0.0152506 | 0.0025613 | trained range |
| Val-B | 5 | 0.1205475 | 0.0056729 | 0.1398991 | 0.0079692 | 0.0021967 | trained range |
| Val-B | 6 | 0.1118213 | 0.0022232 | 0.1277725 | 0.0034583 | 0.0019896 | trained range |
| Val-B | 8 | 0.1048113 | 0.0017276 | 0.1180020 | 0.0023252 | 0.0018202 | extrapolation |


在当前两个pool和该evaluation seed下，case/draw均值随测试V增加而下降。
V4–5 之后收益明显递减，但两池 V6→V8 仍改善超过0.005，未达到预注册严格饱和阈值；不能宣称 V≥6 已饱和。
V2可以运行且输出finite，但relative L1约0.37；V8可以运行且误差低于V6。
这只支持测试过的V∈[2,3,4,5,6,8]，不能扩展到任意数量views。
标准差是draw-induced variability，非case std或training-seed std。
数字源文件 view_count_metrics.csv；对应三张PNG和PDF保留全部数据/errorbars。

## 11. 相同V的angular coverage

| V | Protocol | Draws | Mean span (deg) | Mean L1 | Draw std L1 | Mean L2 | RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 | spread | 1 | 173.5743 | 0.1206317 | 0.0000000 | 0.1486583 | 0.0022696 |
| 4 | random | 5 | 101.1562 | 0.1369043 | 0.0064636 | 0.1625321 | 0.0025695 |
| 4 | clustered | 5 | 2.0149 | 0.7442095 | 0.0397431 | 0.8742404 | 0.0141043 |
| 6 | spread | 1 | 173.5743 | 0.1009869 | 0.0000000 | 0.1167140 | 0.0018036 |
| 6 | random | 5 | 120.3137 | 0.1109225 | 0.0021309 | 0.1226630 | 0.0019237 |
| 6 | clustered | 5 | 4.0470 | 0.6555103 | 0.0324974 | 0.8228637 | 0.0132748 |


Spread优于random，clustered显著变差；增加数量无法抵消极窄coverage。
V4 span↔relative L1 Spearman rho=-0.826833，V6 rho=-0.803885。
每个V有36×(1+5+5)=396行，含重复case；相关系数用于描述，不将重复行当作独立样本做显著性推断。
nominal p-value仅原始输出，未做多重比较校正；不能把rho解释为纯azimuth span的因果效应。
station radius/height和选择的specific poses也可能影响误差。

## 12. Condition stratification

只使用 frozen condition_meta 的真实字段。source_mode=null被保留，未从filename推断；
所有groups均输出，n<3标small_n，不做强结论、不删除case。

| Field | Value | n | Mean L1 | Median L1 | Mean L2 | Mean RMSE | small_n |
|---|---|---:|---:|---:|---:|---:|---|
| geometry_id | "G1_1" | 4 | 0.102035 | 0.092109 | 0.094603 | 0.0010196 | False |
| geometry_id | "G2_12" | 6 | 0.120504 | 0.111652 | 0.130511 | 0.0018845 | False |
| geometry_id | "G2_crosswind" | 6 | 0.133011 | 0.102641 | 0.166600 | 0.0024355 | False |
| geometry_id | "G2_streamwise" | 6 | 0.129177 | 0.110256 | 0.150230 | 0.0021278 | False |
| geometry_id | "G3_streamwise" | 7 | 0.141097 | 0.132422 | 0.155935 | 0.0027519 | False |
| geometry_id | "case3_crosswind_strict_g3" | 7 | 0.156028 | 0.161093 | 0.186258 | 0.0034087 | False |
| source_family | "G1" | 4 | 0.102035 | 0.092109 | 0.094603 | 0.0010196 | False |
| source_family | "G2" | 18 | 0.127564 | 0.105740 | 0.149113 | 0.0021493 | False |
| source_family | "G3" | 14 | 0.148563 | 0.147349 | 0.171096 | 0.0030803 | False |
| wind_speed_m_s | 0.0 | 10 | 0.089857 | 0.078493 | 0.105553 | 0.0021378 | False |
| wind_speed_m_s | 1.0 | 11 | 0.123200 | 0.104918 | 0.133325 | 0.0024376 | False |
| wind_speed_m_s | 2.0 | 4 | 0.125616 | 0.124582 | 0.130002 | 0.0016495 | False |
| wind_speed_m_s | 3.0 | 11 | 0.184358 | 0.180999 | 0.219608 | 0.0028273 | False |
| pressure_pa | 100.0 | 4 | 0.137222 | 0.101450 | 0.152893 | 0.0020340 | False |
| pressure_pa | 1000.0 | 9 | 0.124941 | 0.097855 | 0.139278 | 0.0020351 | False |
| pressure_pa | 1500.0 | 9 | 0.124832 | 0.125043 | 0.133823 | 0.0024027 | False |
| pressure_pa | 300.0 | 10 | 0.145824 | 0.124582 | 0.168303 | 0.0024033 | False |
| pressure_pa | 600.0 | 4 | 0.132272 | 0.118896 | 0.176322 | 0.0034451 | False |
| active_hole_count | 1 | 4 | 0.102035 | 0.092109 | 0.094603 | 0.0010196 | False |
| active_hole_count | 2 | 18 | 0.127564 | 0.105740 | 0.149113 | 0.0021493 | False |
| active_hole_count | 3 | 14 | 0.148563 | 0.147349 | 0.171096 | 0.0030803 | False |
| condition_id | "p0100_w0_equal" | 2 | 0.084402 | 0.084402 | 0.080795 | 0.0016126 | True |
| condition_id | "p0100_w1_equal" | 1 | 0.104918 | 0.104918 | 0.120357 | 0.0015286 | True |
| condition_id | "p0100_w3_equal" | 1 | 0.275165 | 0.275165 | 0.329625 | 0.0033823 | True |
| condition_id | "p0300_w0_inlet2_half" | 1 | 0.112874 | 0.112874 | 0.108322 | 0.0023733 | True |
| condition_id | "p0300_w1_equal" | 1 | 0.147903 | 0.147903 | 0.160293 | 0.0033839 | True |
| condition_id | "p0300_w2_equal" | 3 | 0.112790 | 0.116742 | 0.120788 | 0.0016288 | False |
| condition_id | "p0300_w3_equal" | 1 | 0.180247 | 0.180247 | 0.218071 | 0.0031692 | True |
| condition_id | "p0300_w3_inlet1_half" | 2 | 0.171187 | 0.171187 | 0.206535 | 0.0024709 | True |
| condition_id | "p0300_w3_inlet2_half" | 1 | 0.106563 | 0.106563 | 0.159411 | 0.0016836 | True |
| condition_id | "p0300_w3_inlet3_half" | 1 | 0.229907 | 0.229907 | 0.261505 | 0.0035949 | True |
| condition_id | "p0600_w0_equal" | 3 | 0.104540 | 0.076700 | 0.136885 | 0.0028471 | False |
| condition_id | "p0600_w1_equal" | 1 | 0.215466 | 0.215466 | 0.294632 | 0.0052390 | True |
| condition_id | "p1000_w0_inlet1_half" | 1 | 0.096351 | 0.096351 | 0.088062 | 0.0015942 | True |
| condition_id | "p1000_w0_inlet2_half" | 1 | 0.051461 | 0.051461 | 0.080025 | 0.0014588 | True |
| condition_id | "p1000_w1_equal" | 1 | 0.095011 | 0.095011 | 0.080220 | 0.0010064 | True |
| condition_id | "p1000_w1_inlet1_half" | 2 | 0.096662 | 0.096662 | 0.109511 | 0.0021398 | True |
| condition_id | "p1000_w2_equal" | 1 | 0.164092 | 0.164092 | 0.157646 | 0.0017117 | True |
| condition_id | "p1000_w3_inlet1_half" | 2 | 0.188717 | 0.188717 | 0.228226 | 0.0028367 | True |
| condition_id | "p1000_w3_inlet2_half" | 1 | 0.146795 | 0.146795 | 0.172077 | 0.0025920 | True |
| condition_id | "p1500_w0_inlet2_half" | 1 | 0.080286 | 0.080286 | 0.105835 | 0.0019361 | True |
| condition_id | "p1500_w0_inlet3_half" | 1 | 0.075172 | 0.075172 | 0.101038 | 0.0022494 | True |
| condition_id | "p1500_w1_equal" | 1 | 0.173253 | 0.173253 | 0.179113 | 0.0040169 | True |
| condition_id | "p1500_w1_inlet1_half" | 2 | 0.099459 | 0.099459 | 0.087813 | 0.0013825 | True |
| condition_id | "p1500_w1_inlet2_half" | 1 | 0.125043 | 0.125043 | 0.131379 | 0.0022916 | True |
| condition_id | "p1500_w1_inlet3_half" | 1 | 0.101366 | 0.101366 | 0.105935 | 0.0023024 | True |
| condition_id | "p1500_w3_inlet1_half" | 1 | 0.188455 | 0.188455 | 0.203338 | 0.0027576 | True |
| condition_id | "p1500_w3_inlet3_half" | 1 | 0.180999 | 0.180999 | 0.202140 | 0.0033057 | True |
| source_mode | null | 36 | 0.132894 | 0.114808 | 0.151606 | 0.0023858 | False |


wind/pressure 的 exact-value groups之外，Spearman结果：

```json
[
  {
    "x": "wind_speed_m_s",
    "y": "relative_l1",
    "n": 36,
    "rho": 0.6747439339567655,
    "p_value": 6.3881663969116494e-06,
    "interpretation": "descriptive only; no multiple-comparison correction"
  },
  {
    "x": "wind_speed_m_s",
    "y": "relative_l2",
    "n": 36,
    "rho": 0.6767556928534761,
    "p_value": 5.8516319404075815e-06,
    "interpretation": "descriptive only; no multiple-comparison correction"
  },
  {
    "x": "pressure_pa",
    "y": "relative_l1",
    "n": 36,
    "rho": -0.09821944953281551,
    "p_value": 0.5687458067784168,
    "interpretation": "descriptive only; no multiple-comparison correction"
  },
  {
    "x": "pressure_pa",
    "y": "relative_l2",
    "n": 36,
    "rho": -0.09689573727496087,
    "p_value": 0.5739918095992947,
    "interpretation": "descriptive only; no multiple-comparison correction"
  }
]
```

这些条件存在共变（例如source_family与active_hole_count），分层均值不能解释为独立因果效应。
风速3m/s组较难；specific source geometry结果必须结合n解读。condition_id小组仅描述，不据此调参。

## 13. Best / worst case分析

Pool A和B分别按legacy V4 relative L1排序，保留best5与worst10，未删bad cases。
两个worst10中wind=3m/s分别有
7 / 7例。
许多worst case具有较小mean/active fraction的细窄plume，并涉及G2/G3多源；也存在coverage较窄的case，
但较大span仍有难例，不能只用coverage解释全部误差。
A最差case图中可见两条窄plume的形状/峰值偏差；整体GT mass较小会放大relative error。
worst case的GTmax、mean、active/nonzero fractions、center、integrated proxy、condition_meta、IDs、
azimuths和coverage描述全部存于 rankings.json / enriched.jsonl，三轴GT/pred/error图保存在figures/worst_cases/Pool_A和Pool_B。

| Pool | Rank | Case | L1 | Wind (m/s) | Source family | Active fraction | Angular span |
|---|---:|---|---:|---:|---|---:|---:|
| A | 1 | case2_streamwise_g2_transient_scheme_a_v1_2s_p0100_w3_equal | 0.275165 | 3.0 | G2 | 0.022911 | 120.011 |
| A | 2 | case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet1_half | 0.242010 | 3.0 | G3 | 0.056614 | 105.899 |
| A | 3 | case3_streamwise_g3_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet3_half | 0.229907 | 3.0 | G3 | 0.049385 | 120.156 |
| A | 4 | case2_crosswind_g2_transient_scheme_a_v1_2s_asymmetric_p1000_w3_inlet1_half | 0.226209 | 3.0 | G2 | 0.067795 | 33.271 |
| A | 5 | case2_crosswind_g2_transient_scheme_a_v1_2s_p0600_w1_equal | 0.215466 | 1.0 | G2 | 0.227699 | 23.794 |
| A | 6 | case2_1_g2_transient_scheme_a_v1_2s_asymmetric_p1500_w3_inlet1_half | 0.188455 | 3.0 | G2 | 0.130005 | 84.831 |
| A | 7 | case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p1500_w3_inlet3_half | 0.180999 | 3.0 | G3 | 0.128262 | 94.893 |
| A | 8 | case3_streamwise_g3_transient_scheme_a_v1_2s_p0300_w3_equal | 0.180247 | 3.0 | G3 | 0.051289 | 73.372 |
| A | 9 | case3_crosswind_g3_transient_scheme_a_v1_2s_p1500_w1_equal | 0.173253 | 1.0 | G3 | 0.285969 | 168.978 |
| A | 10 | case1_g1_transient_scheme_a_v1_2s_p1000_w2_equal | 0.164092 | 2.0 | G1 | 0.080105 | 50.472 |
| B | 1 | case2_1_g2_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet2_half | 0.256454 | 3.0 | G2 | 0.041565 | 72.776 |
| B | 2 | case2_streamwise_g2_transient_scheme_a_v1_2s_p0100_w3_equal | 0.220373 | 3.0 | G2 | 0.022911 | 129.475 |
| B | 3 | case3_streamwise_g3_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet3_half | 0.212041 | 3.0 | G3 | 0.049385 | 143.024 |
| B | 4 | case2_1_g2_transient_scheme_a_v1_2s_asymmetric_p1500_w3_inlet1_half | 0.188436 | 3.0 | G2 | 0.130005 | 127.243 |
| B | 5 | case3_streamwise_g3_transient_scheme_a_v1_2s_p0300_w3_equal | 0.185846 | 3.0 | G3 | 0.051289 | 103.640 |
| B | 6 | case3_streamwise_g3_transient_scheme_a_v1_2s_p0300_w2_equal | 0.172570 | 2.0 | G3 | 0.082863 | 52.758 |
| B | 7 | case2_streamwise_g2_transient_scheme_a_v1_2s_asymmetric_p1000_w3_inlet1_half | 0.169663 | 3.0 | G2 | 0.045418 | 160.306 |
| B | 8 | case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet1_half | 0.168105 | 3.0 | G3 | 0.056614 | 52.401 |
| B | 9 | case2_1_g2_transient_scheme_a_v1_2s_asymmetric_p1500_w1_inlet1_half | 0.166186 | 1.0 | G2 | 0.271828 | 59.758 |
| B | 10 | case2_crosswind_g2_transient_scheme_a_v1_2s_p0100_w0_equal | 0.149096 | 0.0 | G2 | 0.158855 | 102.843 |


Best5：

| Pool | Case | L1 |
|---|---|---:|
| A | case2_streamwise_g2_transient_scheme_a_v1_2s_asymmetric_p1000_w0_inlet2_half | 0.051461 |
| A | case1_g1_transient_scheme_a_v1_2s_asymmetric_p1500_w1_inlet1_half | 0.059831 |
| A | case2_crosswind_g2_transient_scheme_a_v1_2s_p0100_w0_equal | 0.070823 |
| A | case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p1500_w0_inlet3_half | 0.075172 |
| A | case2_1_g2_transient_scheme_a_v1_2s_p0600_w0_equal | 0.075828 |
| B | case3_crosswind_g3_transient_scheme_a_v1_2s_asymmetric_p1500_w0_inlet3_half | 0.047638 |
| B | case1_g1_transient_scheme_a_v1_2s_asymmetric_p1500_w1_inlet1_half | 0.052395 |
| B | case1_g1_transient_scheme_a_v1_2s_p0300_w2_equal | 0.055359 |
| B | case2_streamwise_g2_transient_scheme_a_v1_2s_asymmetric_p1000_w0_inlet2_half | 0.056121 |
| B | case2_crosswind_g2_transient_scheme_a_v1_2s_asymmetric_p1500_w0_inlet2_half | 0.064291 |


## 14. Plume characteristics与误差

固定active阈值=1e−5 mass fraction，在查看结果前写入effective protocol，未调阈值。
GT max/mean、nonzero与active fraction、concentration center XYZ均从GT64直接计算。
integrated concentration proxy=mean(GT)×0.6³，等权endpoint grid近似，无density，不能称CH4质量。
浓度中心为各endpoint XYZ坐标的concentration-weighted平均。

| Descriptor | Pool A rho(L1) | Pool A rho(L2) | Pool B rho(L1) | Pool B rho(L2) |
|---|---:|---:|---:|---:|
| gt_max_concentration | 0.25148 | 0.27876 | 0.45225 | 0.35882 |
| gt_mean_concentration | -0.24530 | -0.25997 | -0.32098 | -0.39923 |
| nonzero_voxel_fraction | 0.17992 | 0.12432 | -0.17709 | -0.25019 |
| active_voxel_fraction | -0.21441 | -0.25071 | -0.31943 | -0.41905 |
| gt_integrated_concentration_proxy | -0.24530 | -0.25997 | -0.32098 | -0.39923 |
| concentration_center_x_m | 0.51145 | 0.51506 | 0.29472 | 0.34878 |
| concentration_center_y_m | -0.57375 | -0.62497 | -0.51892 | -0.57477 |
| concentration_center_z_m | 0.02291 | -0.01236 | 0.15753 | 0.12124 |


中心坐标和误差的相关可能反映wind/source共变；GT mean与integrated proxy只差常数，相关系数相同，
不算两份独立证据。所有相关仅解释失败模式，未用于模型选择。

## 15. Multi-seed结果

原seed20261011仅引用已有checkpoint和日志，没有重新训练其150-epoch正式实验。
seed20261012/13只改变training seed（初始化、train station sampling和shuffle）；架构、loss、AdamW、
batch2、V3–6、FP32、150epoch上限与patience20不变。为了隔离训练seed变异，
所有runs的validation selection seed固定20261011、epoch0、Pool A/V4，station IDs与reference逐case一致。
该评估seed与training seed明确分开记录在validation_protocol.json与checkpoint。

| Seed | Val-A MAE | Val-A RMSE | Val-A L1 | Val-A L2 | Val-B L1 | Val-B L2 | Best epoch (1-based) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261011 | 0.00045930037 | 0.0023858286 | 0.1328937 | 0.1516056 | 0.1259693 | 0.1549957 | 148.000 |
| 20261012 | 0.00048623262 | 0.0025320834 | 0.1386302 | 0.1587694 | 0.1356972 | 0.1683182 | 150.000 |
| 20261013 | 0.00049987426 | 0.0026862975 | 0.1441474 | 0.1696486 | 0.1329817 | 0.1601472 | 147.000 |
| mean | 0.00048180242 | 0.0025347365 | 0.1385571 | 0.1600079 | 0.1315494 | 0.1611537 | 148.333 |
| std (ddof=1) | 2.064655e-05 | 0.00015025204 | 0.0056272 | 0.0090850 | 0.0050196 | 0.0067181 | 1.528 |


Pool A relative L1=0.1385571 ± 0.0056272，
relative L2=0.1600079 ± 0.0090850；
Pool B relative L1=0.1315494 ± 0.0050196。
完整A/B MAE、RMSE、L1/L2、bestepoch mean/std见multiseed_metrics.csv。
seed std使用ddof=1；只有3 seeds，此稳定性评价有有限样本限制。
新runs均只用Pool A val loss选择best，Pool B评估发生在checkpoint冻结后。

## 16. Horizon probe

**Option A：fresh diagnostic training**，seed20261011、max_epochs220，其余optimizer/data/loss/seed语义不变。
原cosine T_max150的last.pt已将LR降到零，不能合理地直接改checkpoint scheduler继续；
因此没有篡改、覆盖或恢复正式last.pt，220诊断从随机初始化开始，cosine T_max220。
这从epoch0改变LR轨迹，测试的是更长/更慢decay的diagnostic protocol，非纯粹延长相同轨迹。
保持原patience20，实际完成220 epochs，
结束原因=max_epochs，bestepoch=220（1-based）。

| Condition | Best val loss | Mean L1 | Mean L2 |
|---|---:|---:|---:|
| Reference 150 | 0.0132928545 | 0.132893722 | 0.151605587 |
| Diagnostic 220 | 0.011636973 | 0.116344005 | 0.131589228 |

relative L1 absolute improvement=0.0165497178，
即1.654972 percentage points；预注册meaningful阈值=0.5个百分点（比例0.005）。
training horizon may be limiting B0；不替换正式 reference，先由负责人决定后续训练 horizon。
诊断best仍位于第220个epoch，若处于220上限则不能据此宣称220已经充分收敛。
只用val loss/L1/L2回答诊断问题，未根据test或Pool B选择更长horizon。
原正式B0、原150-epoch配置与原checkpoint仍保持原样。

## 17. Shortcut跨seed复查

同一Val-A-V4/reference视图；zero Y保留valid/path/geometry，wrong G循环错位全部station geometry，
包括origin/basis、valid/path、ray geometry、IDs，Y仍原顺序。仍复用原B0 shortcut语义。

| Seed | Condition | L1 | L2 | MAE | RMSE | L1 / normal |
|---|---|---:|---:|---:|---:|---:|
| 20261012 | normal | 0.1386302 | 0.1587694 | 0.00048623262 | 0.0025320834 | 1.000 |
| 20261012 | zero_measurement | 1.0014229 | 0.9999412 | 0.0036766496 | 0.016179268 | 7.224 |
| 20261012 | wrong_geometry | 0.8826320 | 0.8868136 | 0.0031672644 | 0.014268869 | 6.367 |
| 20261013 | normal | 0.1441474 | 0.1696486 | 0.00049987426 | 0.0026862975 | 1.000 |
| 20261013 | zero_measurement | 1.0012510 | 0.9999511 | 0.0036762811 | 0.016179434 | 6.946 |
| 20261013 | wrong_geometry | 0.8747770 | 0.8830191 | 0.0031286927 | 0.014204729 | 6.069 |


这检验原measurement/geometry结论能否跨训练seed复现，不对失败条件重选subset或改模型。

## 18–20. 限制、可以支持的结论与B1建议

冻结reference在预定义V2/3/4/5/6/8上都可推理；当前测试V增大时平均误差下降。
当前results支持defined acquisition family内的unseen station poses泛化，
支持相同V下宽coverage更有利。不能宣称arbitrary geometry、任意V或已经解决sparse-view tomography。

仍只有36个val plumes，同一Fluent来源和若干相关source families，未做独立现实测量域验证。
这些val案例也用于best checkpoint和early stopping选择，本阶段分数不是独立盲测test估计。
小组样本、three-seed数量、同case重复draw及acquisition family的边界限制外推；
standard deviations为各自明确的case/draw/seed变异，非泛化置信区间。
32³ lifting和3D卷积对窄/多源plume可能有细节与峰值误差。
coverage descriptor并不完全捕获半环方向、radius、height及具体poses。
Horizon probe改变cosine轨迹，单diagnostic seed不能区分纯训练时长效应和学习率日程效应。
CUDA grid_sample backward不承诺逐bit训练轨迹复现；station selection则完全确定性可重放。

B1门槛在effective_protocol.yaml预注册：Val-A L1 seed CV≤0.15；Pool B mean L1≤0.5且相对A恶化≤100%；
A/B V3→V6均改善；两个新seed的wrongG/zeroY L1至少normal的1.2倍；
horizon诊断L1改善<0.005。它们是本次分析的操作性判断，非通用物理界限。

| Predefined B1 criterion | Passed |
|---|---|
| multiseed_stable | True |
| pool_b_not_catastrophic | True |
| v3_to_v6_improves | True |
| shortcuts_replicated | True |
| horizon_not_limiting | False |


**建议：NO-GO; resolve failed baseline characterization checks first。**
若horizon检查失败，优先解决baseline horizon公平性，由负责人决定正式训练预算，
不自动替换reference，不直接实现B1。若后续所有门槛满足，优先比较简单的
geometry-conditioned attention pooling与固定mean fusion，先保持encoder/lifting/decoder不变；
全量set self-attention可作为后续对照，本轮未实现、未训练任何新网络结构。

## Integrity、tests与复现

Test set remained untouched in this phase：没有运行test推理/训练/超参数选择。
完整性QA仅只读hash全部frozen artifacts，包含test文件，不是test model evaluation。
Pool B始终evaluation-only；不用于training/model architecture/optimizer选择。

Ran 38 tests in 12.974s，全部OK。
测试覆盖V个唯一station、Pool隔离、draw可重放、draw变化、spread/clustered覆盖性质、
Pool B投影/geometry正确配对与真实model的同步view permutation、draw/case variability区分、
固定GT threshold/XYZ center、null metadata与small_n处理，以及此前全部31个tests。

本轮前后 frozen dataset=1262 files、source=714 files
inventory完全一致（size/mtime/inode/SHA256；historical inputs/保留旧QA的stat-only规则），
old supervised_b0 logs/checkpoints/figures完全未修改，model source hashes、manifest/split/geometry/statistics hashes一致。
完整性passed=True，reference checkpoint unchanged=True。

运行命令（仓库根目录，已安装torch/CUDA、NumPy、SciPy、PyYAML、Matplotlib）：

```bash
python scripts/evaluate_b0_generalization.py --stage reference
python scripts/train_b0_multiseed.py --seed 20261012
python scripts/train_b0_multiseed.py --seed 20261013
python scripts/run_b0_horizon_probe.py
python scripts/evaluate_b0_generalization.py --stage all
python scripts/summarize_b0_generalization.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
```

训练入口禁止重训reference seed和覆盖已有run；恢复新run时可加 --resume，原reference resume不涉及本轮。
所有artifacts位于 `/data/djc/GeoPhy-Set3D/outputs/evaluations/supervised_b0_generalization_v1`：reference.json/effective_protocol.yaml、summary.json、integrity_summary.json、
view_count_metrics.csv、angular_coverage_metrics.csv、multiseed_metrics.csv与各实验目录。
每个新run含effective config/environment、逐epochmetrics、best/last完整checkpoint、curves与fixed-val图。
主图同时导出PNG/PDF，源CSV/JSONL保留所有case；数据集/权重/日志/图不进入Git。
