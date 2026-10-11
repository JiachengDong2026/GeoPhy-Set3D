# Projection Pool 训练接口与 C64 投影数值审计 v1

执行日期：2026-10-11（Asia/Shanghai）。本轮只实现训练数据接口和测量审计。
冻结数据集、case split、二维 TDLAS 扫描定义、K_ref=256、K_train=96 和现有投影器均保持原值。
未实现重建网络、训练器或最终 loss。

## 文件与入口

新增实现：

- `src/geophy_set3d/data/projection_pool_dataset.py`：索引驱动、只读 Dataset。
- `src/geophy_set3d/data/view_sampling.py`：可复现在线 station 采样。
- `src/geophy_set3d/data/collate.py`：variable-view padding 与两类 mask。
- `src/geophy_set3d/data/projection_audit.py`：全量 train / Pool A 数值审计与绘图。
- `scripts/audit_c64_projection_mismatch.py`：审计 CLI。
- `scripts/validate_training_interface.py`：真实数据接口、配置与多 worker 验证。
- `scripts/validate_frozen_generation.py`：将既有生成 QA 写到外部目录的只读验证入口。
- `configs/training_data_v1.yaml`：训练数据与 DataLoader 配置。
- `tests/pool_fixture.py`：小规模冻结 schema fixture，文件路径刻意与 case_id 不同。
- `tests/test_projection_pool_dataset.py`、`tests/test_view_sampling.py`、
  `tests/test_variable_view_collate.py`、`tests/test_differentiable_projector_resolution.py`、
  `tests/test_projection_audit.py`：单元测试。

修改 `README.md`，新增本报告。原生成 Python/C++/YAML 文件没有修改。
本机增加被 Git 忽略的 `outputs/audits` 链接，目标是 `/data/djc/GeoPhy-Set3D/outputs/audits/`。
所有审计数组、CSV、JSON、日志和 PNG 均留在外部输出目录，不进入 Git。

## Dataset API 与 Pool 政策

```python
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate

dataset = ProjectionPoolDataset(
    dataset_root="/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1",
    split="train", geometry_pool="train",
    min_input_views=3, max_input_views=6, hold_views=1,
    base_seed=20261011, normalization="linear_p99_5",
    statistics_file="train_statistics.json", load_gt128=False,
)
dataset.set_epoch(0)
sample = dataset[0]
```

唯一 membership 入口为 `source_index.jsonl`；不扫描 case 目录推测样本。
使用索引中的 projection/GT64/source GT 路径，检查 complete 状态、split 与重复标识。
几何文件每个 Dataset 实例只读取一次。NPY 使用 `mmap_mode="r"`，每次只复制所选 station
和当前 GT；初始化不加载整个 Projection Pool，也不默认读取 GT128。

| split | geometry_pool | 含义 / 规则 |
| --- | --- | --- |
| train | train | 训练；只读取 Pool A 几何及投影 |
| train | ood | 默认拒绝；只有显式 `allow_train_ood_debug=True` 才可进行特殊 debug，本轮未使用 |
| val/test | train | 羽流泛化评估；新 case + Pool A |
| val/test | ood | 未见几何评估；新 case + Pool B |

真实接口验证对 val/test 的 A/B 只做 membership 与读取检查，不计算误差或选阈值。
Task A 的全部数值审计只使用 166 个 train case 和冻结 Pool A。

## 单样本字段与 tensor shape

下表 `S=input` 时 `N=V`（默认 3–6），`S=hold` 时 `N=V_hold`（默认 1）。
返回投影、几何和 GT tensor 位于 CPU；模型调用者负责设备搬运。

| 字段 | shape / dtype / 含义 |
| --- | --- |
| case_id / sample_id / split / geometry_pool | str |
| condition_meta | dict；保留 null，不猜测 source_mode |
| S_station_indices | `[N]` int64；冻结 candidate station ID |
| S_projection_raw | `[N,1,64,64]` float32；原始 ppm·m |
| S_projection | `[N,1,64,64]` float32；固定训练尺度变换 |
| S_valid_ray | `[N,1,64,64]` bool；ray 是否穿过重建域 |
| S_path_length_m / S_t_near / S_t_far | `[N,1,64,64]` float32；m |
| S_ray_direction | `[N,64,64,3]` float32；世界坐标 XYZ 单位方向 |
| S_station_origin_m | `[N,3]` float32；世界坐标 XYZ，m |
| S_station_azimuth_deg / S_station_radius_m / S_station_height_m | `[N]` float32 |
| S_center_direction / S_right_direction / S_up_direction | `[N,3]` float32；冻结 station 局部基 |
| horizontal_angle_deg / vertical_angle_deg | `[64]` float32；共享 steering 角数组 |
| projection_scale_ppm_m | scalar float64；训练 valid-ray p99.5 |
| gt_volume_64 | `[1,64,64,64]` float32；XYZ，CH4 mass fraction，范围 [0,1] |
| gt_volume_128 | 仅 `load_gt128=True` 返回 `[1,128,128,128]` float32 原始 GT |

几何来自冻结的 float64 NPZ，训练 tensor 转 float32；审计直接用原 float64 几何。
没有按 case max 归一化 GT，也没有将投影/mask/path length 强制拼成 3 channels。
valid-ray 的零甲烷 measurement 保持 True，invalid-ray 的零占位保持 False。

## 归一化

所有 split/pool 都读取同一训练统计文件，并检查统计的 train case membership、Pool A 声明和 ppm_m 单位。
尺度从 `projection_valid_rays["p99.5"]` 读取；没有硬编码数值。

```text
Y_ref = 32284.894580078173 ppm·m  # 本次读取结果
linear_p99_5: Y_normalized = Y_raw / Y_ref
log1p_p99_5:  Y_normalized = log1p(Y_raw / Y_ref)
```

默认 linear，不 clip，也不使用 val/test 统计或 per-image / per-case 最大值。
`*_projection_raw` 与磁盘数据逐元素一致，保留未来 measurement-domain loss 所需物理量。

## 采样、epoch 与复现

`V_input` 在闭区间 [3,6] 均匀采样，从全体 candidate station 无放回选择 input 与 hold，
再随机打乱 input 顺序。不固定使用 0,1,2,…；同一 case 不同 epoch 使用新的确定性随机流。

seed 仅由 base_seed、epoch、sample index、case_id 的稳定 BLAKE2b 摘要组成，
经 NumPy SeedSequence 创建局部 RNG，不使用 Python hash、全局 NumPy 或 Torch RNG。
同一 seed、epoch、sample index 和 case_id 重复访问重放相同 input/hold station IDs 及顺序；
worker ID、num_workers 和调度不影响结果。改变 sample index 或 epoch 会使用新的随机流。
2026-10-11 的修正同时删除了 Dataset 中 get_worker_info 的依赖。

新增测试确认 num_workers=0/2/4 在 epoch 0、1 及重置到 0 时逐元素一致；
2/4 workers 使用 persistent_workers=True，跨 epoch 共享更新正确，多数 case 组合改变。
真实冻结数据的前 8 个 train samples 也通过相同比较。
验证结果在 outputs/logs/supervised_b0/data_validation/validation_summary.json。

epoch 保存于共享内存 torch int64 scalar，`set_epoch` 对已有 persistent worker 同样生效，
fork 和 spawn 都已验证。必须在创建该 epoch 的 iterator 前调用，上一 epoch 应先完成；
不要在一个 iterator 正在预取时修改 epoch。DataLoader 的 shuffle 顺序应另用固定 seed 的
`torch.Generator`；配置的实际 loader smoke test 已按此执行。
评估默认 epoch=0，可重放同一组选定 station，除非调用者主动 set_epoch。

## Variable-view collate

`projection_pool_collate` 分别对 input/hold pad 到当前 batch 最大 view 数。
投影、方向、origin、路径、近远距离和 station 标量 padding 为零，valid_ray 为 False，
station_indices padding 为 -1。`condition_meta` 保留为 dict list，支持 null。

真实 case 的 V=3 与 V=6、batch size=2 已验证得到：

```text
input_projection / input_projection_raw / input_valid_ray / input_path_length_m:
    [2,6,1,64,64]
input_ray_direction: [2,6,64,64,3]
input_station_origin_m: [2,6,3]
view_mask: [2,6]  # [[True,True,True,False,False,False], [True]*6]
hold_projection / hold_projection_raw / hold_valid_ray / hold_path_length_m:
    [2,1,1,64,64]
hold_ray_direction: [2,1,64,64,3]
hold_station_origin_m: [2,1,3]
hold_view_mask: [2,1]
gt_volume_64: [2,1,64,64,64]
```

`view_mask` 是实际 view 与 padding 的区分，`valid_ray` 是实际射线的有效性区分。
不能用零 projection 代替任一 mask。单元测试另覆盖 variable hold 数量 1/2 和 hold=0。

## C64 与 C128 审计

审计覆盖全部 166 个 train case × 64 个 Pool A station，
共 23,570,174 个有效 case-ray 元素。CPU / float64 / 8 线程 / 4-station chunk，
用时 515.1 s。没有生成新 geometry 或 reference projection。

```text
E_operator = P_diff(C_GT128, K=96) − saved P_ref(C_GT128, K=256)
E_total64  = P_diff(C_GT64,  K=96) − saved P_ref(C_GT128, K=256)
E_resolution96 = P_diff(C_GT64,96) − P_diff(C_GT128,96)
```

正式误差仅在冻结 valid_ray=True 上计算。全局 L1 是绝对残差之和 / reference 绝对值之和；
L2 是残差平方和的平方根 / reference 平方和的平方根，使用 float64 累积。
不是平均逐 case 或逐 station 的相对误差。relative ratio 分母 floor 为 1e-20，仅处理零分母。
E_resolution96 的 relative 分母也使用 saved reference，便于对照两个主指标。bias 是 prediction − reference。

| 全局有效 ray 指标 | E_operator | E_total64 | E_resolution96 |
| --- | ---: | ---: | ---: |
| relative L1 | 0.095820% | 1.399296% | 1.403792% |
| relative L2 | 0.315723% | 3.077827% | 3.074133% |
| MAE (ppm·m) | 2.899666 | 42.344934 | 42.480979 |
| RMSE (ppm·m) | 23.098416 | 225.174749 | 224.904541 |
| maximum absolute error (ppm·m) | 1555.896647 | 17463.537177 | 17463.094929 |
| mean signed error / bias (ppm·m) | 0.531926 | 16.193873 | 15.661947 |

从 C128 到 C64，relative L1 增加 1.303476 个百分点，
relative L2 增加 2.762103 个百分点。
这是总误差范数的 increase；不能把两个范数的差直接解释为独立下采样算子的范数。
因此另外报告相同 K 下实际 C64−C128 measurement 差的 E_resolution96。
主要额外 mismatch 来自 endpoint-aligned 128³→64³ 的重采样和随后射线插值；
常量/仿射解析测试已排除分辨率改变时的 XYZ/ZYX 映射、长度和单位错误。

| station relative L1（有效 ray） | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| E_operator | 0.098895% | 0.215998% | 0.260514% | 0.344426% | 0.577112% |
| E_total64 | 1.360578% | 2.886987% | 3.432992% | 4.445332% | 7.796562% |
| E_resolution96 | 1.355246% | 2.919367% | 3.469726% | 4.445791% | 7.730152% |

所有 per-case 的六项指标在 per_case_metrics.jsonl；每个 case/station 的三个对照在
per_station_metrics.csv。audit_summary.json 另保存 case 与 station 的 L1/L2 分布、
flagged case ID 和输入哈希；run_config.yaml 记录全部 case ID、精度、K 与指标定义。

全局是否超过建议 review 水平（L1>2% 或 L2>3%）：True。
逐 case 超过至少一项的数量为 87/166，状态为 `needs review`。
最差 case relative L1 是 4.711562%：`case1_g1_transient_scheme_a_v1_2s_asymmetric_p0300_w3_inlet1_half`。
最差 case relative L2 是 7.276625%：`case1_g1_transient_scheme_a_v1_2s_p0100_w2_equal`。

判断：当前 64³ 的全局 mismatch 已超过用户给出的建议 review 水平，暂不建议直接把强
projection-domain loss 当作无建模误差的约束。应先审阅逐 case/station 的误差分布，
由负责人决定允许的误差、loss 权重或分辨率策略。该提示没有判定数据集无效。

本轮没有用 val/test/Pool B 选择阈值，没有修改 K_train、GT resolution 或冻结数据。

## 验证与只读核对

```bash
cd /home/djc/projects/GeoPhy-Set3D
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests \
  /scratch/djc/envs/gas-torch/bin/python -m unittest discover -s tests -v
/scratch/djc/envs/gas-torch/bin/python scripts/validate_training_interface.py
/scratch/djc/envs/gas-torch/bin/python scripts/validate_frozen_generation.py
/scratch/djc/envs/gas-torch/bin/python scripts/audit_c64_projection_mismatch.py \
  --samples 96 --split train --device cpu --dtype float64 --threads 8
```

21 个新增 unittest 全部通过。覆盖 Pool 策略、索引唯一入口、按索引路径访问、原始投影一致性、
训练统计校验、linear/log1p、valid zero 与 invalid zero、GT64/可选 GT128、几何只加载一次、
variable input/hold padding、多 worker 复现、persistent epoch 和 spawn worker。

64³/128³ 常量场在 float32/float64 下均符合 `1e6 × mole_fraction × path_length`，
两种分辨率结果几乎相同。非对称 XYZ 仿射场通过独立解析 midpoint 值、SciPy quad 积分和
midpoint 二阶误差上界验证；质量分数到摩尔分数是非线性的，仿射场的 K=96 midpoint 积分
也有可计算的截断误差。64³/128³ 仿射场投影一致且 autograd 梯度有限、非零。
原 `differentiable_project` 已支持任意立方网格，没有改动 XYZ/ZYX 或 align_corners。
另增加完整审计端到端测试，验证 JSON/CSV/4 张图的写出、常量场数值、train-only 读取和冻结目录写入拒绝。

真实数据接口验证覆盖全部 166 个 train case，train/val/test membership 是 166/36/35。
input V 实际覆盖 3/4/5/6，input/hold 无重叠，所有选中 raw measurement 与磁盘一致，
valid_ray 与冻结几何一致。4 worker 的独立 loader 同 seed/epoch 可重放，persistent worker
能收到 epoch 更新；配置的 batch size=2、shuffle 和 loader 参数也已 smoke test。

原生成 QA 在外部目录重跑通过：1,189 个 artifact SHA-256，714 个 source 文件指纹，
237 个 source GT QA，474 个 projection 数组，全部 GT64，以及原几何/slab、split leakage、
zero/constant/非对称仿射/偏心球/Gaussian、64/128/256/512 收敛、SciPy 对照和 autograd。
这些 QA 可读取 Pool B 作固定完整性检查，真实 GT 的投影器方法验证始终仅使用 train case。
没有运行会改写冻结目录的生成 `--audit-only` 或 `--resume`。

对本轮执行前后的完整文件清单、size、mtime、inode 与 SHA-256 进行对照：
冻结数据集 1,262 个文件的修改、新增、删除均为 0；源数据 714 个文件同样为 0。
历史 inputs 仅检查 stat，不读取数组内容。`immutability_summary.json` 保存核对结果。

输出目录：

```text
/data/djc/GeoPhy-Set3D/outputs/audits/c64_projection_mismatch_v1/
  audit_summary.json
  per_case_metrics.jsonl
  per_station_metrics.csv
  run_config.yaml
  audit.log
  float32_crosscheck.json
  figures/{relative_l1_histogram,relative_l2_histogram,c128_vs_c64_scatter,
           representative_projection_difference}.png
/data/djc/GeoPhy-Set3D/outputs/audits/training_interface_v1/
  validation_summary.json
  validation.log
  tests.log
  generation_regression/
  generation_regression.log
  inventory_before.json
  immutability_summary.json
```

## 限制与下一步

本轮使用真实冻结几何，不加入噪声、标定误差或 angular-coverage 分层策略。
这些应在后续实验阶段显式设计。没有 rank-aware 分布式采样或训练器。
每个 epoch 同 sample index/case 的重复访问重放同一组合；如需要 replacement sampler 的每次访问
均独立，需显式增加 draw 标识并验证恢复训练语义。

全量审计是 float64 数值基线。补充选取索引等间距的 10 个 train case、Pool A station
0/21/42/63，以相同 K=96 对照 float32 与 float64：64³ 相对 L1/L2 为
0.000187% / 0.000284%，128³ 为
0.000201% / 0.000336%。
此范围内 float32 差异远小于本轮 C64 mismatch，结果见 float32_crosscheck.json；
该检查没有覆盖全部 case/station，也没有验证 AMP，不能默认将几何计算置于 float16 autocast。
新增 Python 文件会改变原生成入口收集的 code_hashes，因此历史 Debug 验收与当前整个源码树
不会相等；冻结数据保留其原生成源码快照与哈希，本轮没有刷新验收或重新生成数据。

先按本次 measured mismatch 由项目负责人决定后续 projection-domain loss 策略，
再开发 B0 网络。本轮没有自动修改 K、GT 分辨率、reference projection 或 split。
