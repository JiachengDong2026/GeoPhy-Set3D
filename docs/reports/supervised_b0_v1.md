# GeoLift-Mean-Res3D / Supervised B0 v1

本报告记录实际运行结果，实验名称 `supervised_b0`。训练源代码 commit：
`e30bbcd5f2962a509bffe7a24ebdd1db709fc231`，训练启动时工作树 clean；PyTorch 2.11.0+cu128，CUDA 12.8，
GPU NVIDIA GeForce RTX 4090，FP32，seed=20261011。报告提交位于该实现 commit 之后。
数据根目录 `/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1`。

## 1–2. ViewSampler 与 worker 一致性

seed 仅包含 base_seed、epoch、sample_index、case_id 的 BLAKE2b 摘要，经 NumPy SeedSequence
产生局部随机流。删除 worker_id 与 Dataset get_worker_info；无 Python hash 或全局随机状态依赖。
相同 sample/case/epoch 重复读取重放同一 input/hold IDs 和顺序。
num_workers=0/2/4 在 epoch 0、1、重置到 0 时结果逐元素相同；2/4 workers 的 persistent workers
正确接收 shared-memory epoch 更新；多数 case 跨 epoch 改变 view combination。
真实数据前 8 个 train samples 也通过该检查。shuffle 的 sample 顺序使用独立 Torch Generator。
证据：`logs/supervised_b0/data_tests.log`、`data_validation/validation_summary.json`。

## 3–5. 网络、张量形状与参数量

物理轴全程为 XYZ，即 Conv3D D=X、H=Y、W=Z。所有 view 使用同一个 encoder，GroupNorm
在每个 view 内独立归一化。模型总参数 **306,625**；encoder 56,832；decoder 249,793；
lifting/fusion 无可训练参数。

| Module | Operation | Output shape | Parameters |
|---|---|---|---:|
| Input channels | Y / training p99.5; valid_ray; path / (√3×0.6) | B×V,3,64,64 | 0 |
| Stem | Conv2D 3→32,3×3 + GN8 + SiLU | B×V,32,64,64 | 960 |
| Encoder residual 1/2/3 | Two Conv2D + GN, residual + SiLU per block | B×V,32,64,64 | 18,624 each |
| Explicit lifting | Bilinear grid_sample, align_corners=True | B,V,32,32,32,32 | 0 |
| Masked mean | Visibility × view_mask | B,32,32,32,32 | 0 |
| Decoder residual 1/2/3 | Two Conv3D + GN, residual + SiLU per block | B,32,32,32,32 | 55,488 each |
| Upsample | Trilinear, align_corners=True | B,32,64,64,64 | 0 |
| Transition | Conv3D 32→24,3³ + GN8 + SiLU | B,24,64,64,64 | 20,808 |
| Decoder residual 4/5 | Two Conv3D + GN, residual + SiLU per block | B,24,64,64,64 | 31,248 each |
| Head | Conv3D 24→1,1³ + Sigmoid | B,1,64,64,64 | 25 |

输出为 CH4 mass fraction [0,1]。2D 无空间 downsampling，未加入 attention、Transformer、CBAM、
多尺度 skip 或 depth MLP。输入 station/ray 几何用于 lifting，不作为 encoder image channel。
projection_scale 来自固定 train/Pool A valid-ray p99.5，未做 case normalization 或 clipping。

## 6. Lifting 几何与 mean fusion

建立 endpoint-aligned XYZ grid：x,y,z ∈ linspace(0,0.6,32)。令 delta=p−origin，
f=delta·center，r=delta·right，u=delta·up，
α=atan2(r,f)，β=atan2(u,√(f²+r²))。
这与 d=cosβ cosα center + cosβ sinα right + sinβ up 的 TDLAS 约定一致。

读取 batch 的 horizontal/vertical_angle_deg 首尾端点；当前冻结 linspace 是递增且等距的。
将 α、β 分别线性映射至 [-1,1]；grid_sample 顺序为 [horizontal,vertical]，align_corners=True。
可见性要求 f>0、α/β 都在扫描范围内，并与 view_mask 合并；不根据 Y==0 判断有效性。
对每 voxel 的可见 views 做 sum(MF)/max(sum(M),1)，无可见 view 的 fused feature 为零。
padding 在 encoder 输入和 fusion 两处屏蔽。XYZ 不做隐式转置。

## 7–8. Loss、optimizer、scheduler、协议

L = mean(SmoothL1(pred,GT64,beta=1)) + 0.1 × mean_case(sum|pred−GT| / max(sum|GT|,1e−6))。
仅 voxel supervision，无 projector、L_obs/L_hold、TV、sparsity 或 mass regularizer。
AdamW lr=2e−4、weight_decay=1e−4、gradient clip=1.0；完整训练 cosine decay，无 warm restart。
FP32、batch_size=2、accumulation=1；支持 sample-weighted accumulation 和 partial final group。

完整训练 166 train cases，Pool A only，每 epoch V=3–6；hold_views=0 只在 B0 配置设置，Dataset
默认仍为1。val 36 cases、Pool A、V=4、epoch=0，每次使用同一 station subset/order。
150 epoch 上限，patience=20，按 val total loss 选择 best；test set not used for model selection，
本次没有运行 B0 test/Pool B 模型评估。冻结 split 保持 166/36/35。

所有指标直接在物理 mass fraction 上计算。mean per-case 指逐 case 指标再平均；pooled 的
MAE/RMSE 来自全体 voxel，relative L1/L2 来自全体 absolute/squared error 与 GT 总和之比。
relative L2=sqrt(sum(error²)/max(sum(GT²),1e−12))。
额外 integrated_concentration_relative_error 使用等权 grid sum 的相对差，作为浓度积分 proxy；
没有密度，不能解释为 CH4 质量。peak error 是 prediction/GT 最大质量分数的绝对差。

## 9. 四 case overfit gate

从 train index 的等距位置固定选择 4 cases：0,55,110,165，V=4、epoch=0，训练300 epochs，
相同网络/loss/AdamW、FP32，overfit 阶段 constant lr，不调 frozen 数据。
正式训练从新的随机初始化开始，未使用 overfit weights。

| Metric | Initial | Final |
|---|---:|---:|
| Total loss | 12.2490797 | 0.0198333547 |
| SmoothL1 | 0.112161912 | 6.44284978e-06 |
| Mean per-case relative L1 | 121.369171 | 0.198269126 |
| Mean per-case relative L2 | 28.2779535 | 0.187344113 |

固定 subset-mean 的 relative L1=1.17746761；
交换 case 预测后 relative L1=1.72380608；
max pairwise prediction MAE=0.00805821735，prediction max=0.62368083。
验收要求 loss<initial×0.25、relative L1<0.9、比 subset-mean 和交换预测好10%以上、
prediction max>0.001、pairwise MAE>1e−5。结果 **passed=True**，确认输出非零且具有 case 差异。
全部4 cases 的 GT/pred/error XY/XZ/YZ central slices 与 loss/relative L1 曲线已保存。

## 10–11. 完整 train/val 曲线与 best epoch

完成 **150 epochs**，结束原因 `max_epochs`。
最佳 epoch 使用 **0-based=147 / 1-based=148**。
该 epoch 的 online train loss=0.0112703053，固定 val loss=0.0132928545。
online train 指该 epoch 更新期间的平均，不是固定 views 下的事后 train score。
训练总 epoch wall time=1141.864s；平均=7.612s。
最高训练 allocated memory=1126.481 MiB。
完整逐 epoch train/val losses、physical metrics、LR、GPU peak 和 epoch time 见 metrics.jsonl；
曲线见 figures/supervised_b0/training_curves.png，固定 val[0] 的 best epoch slices 见 best_fixed_val.png。

## 12–17. Best val、trivial baselines 与 shortcut 检查

所有条件对同一36-case val/Pool A/V4/epoch0计算，best checkpoint 冻结后运行。
mean per-case：

| Condition | MAE | RMSE | relative L1 | relative L2 |
|---|---:|---:|---:|---:|
| normal | 0.000459300374 | 0.00238582859 | 0.132893722 | 0.151605587 |
| zero_volume | 0.00367329112 | 0.0161802299 | 1 | 1 |
| train_mean_volume | 0.00544472461 | 0.0144342615 | 1.91583476 | 0.916131526 |
| wrong_geometry | 0.00314161134 | 0.014273698 | 0.877443858 | 0.887177202 |
| wrong_measurement | 0.00559591076 | 0.0179510675 | 1.9235519 | 1.16955229 |
| zero_measurement | 0.00367651287 | 0.0161794027 | 1.00134029 | 0.999949198 |

pooled/global：

| Condition | MAE | RMSE | relative L1 | relative L2 |
|---|---:|---:|---:|---:|
| normal | 0.000459300374 | 0.00263896663 | 0.125037837 | 0.158147457 |
| zero_volume | 0.00367329112 | 0.0166867472 | 1 | 1 |
| train_mean_volume | 0.00544472461 | 0.0146410618 | 1.48224696 | 0.877406579 |
| wrong_geometry | 0.00314161134 | 0.0147456301 | 0.855257924 | 0.883673125 |
| wrong_measurement | 0.00559591076 | 0.0185893308 | 1.52340519 | 1.11401764 |
| zero_measurement | 0.00367651287 | 0.0166858906 | 1.00087707 | 0.999948665 |


- Zero-volume baseline：全部 voxel=0。
- Train-mean-volume：仅使用全部166 train GT64 的逐 voxel 平均，val/test 不参与平均体积构造。
- Wrong geometry：四 station 做循环错位（无固定点）；station basis/origin、valid/path maps、
  ray 几何、station IDs 同步移动，Y 保持原顺序，破坏 Y↔G 配对。
- Wrong measurement：使用下一个 val case（末尾循环回首个）的 Y，按 recipient station IDs
  从 donor 的 Pool A 完整投影选取；recipient geometry、valid/path、GT 不变。
- Zero measurement：Y=0；geometry、valid/path、views 保持不变。

三组 shortcut 的 mean relative L1 / normal 比值：{"wrong_geometry": 6.602598247649225, "wrong_measurement": 14.474362386987702, "zero_measurement": 7.5348953071652405}。
正常输入相较三组破坏性输入均明显更好，支持本次模型确实利用 measurement 与显式 geometry。
这属于该数据集上的受控检查，不能证明新来源/新站点分布的泛化能力。

## 18–19. View permutation、padding 与 RTX4090 GPU smoke

| Batch / views | Peak allocated | Peak reserved | Forward | Backward | Permutation max / mean | Padding max |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 6 | 560.086 MiB | 624.000 MiB | 0.523017 s | 0.237528 s | 2.74181e-06 / 1.82459e-07 | 0 |
| 2 / 6 | 1126.644 MiB | 1240.000 MiB | 0.019927 s | 0.040952 s | 2.26498e-06 / 1.11243e-07 | 0 |

B2/V6 可运行，无 OOM；保持完整物理定义。计时包括第一次 CUDA kernel 的启动/初始化，
B1 较慢不能用于推断稳态吞吐。allocated/reserved 为本进程 PyTorch 分配，非整卡总占用；
其他用户任务未停止。所有层 gradients finite，encoder gradients nonzero，输出 finite 且在 [0,1]。
初始2e−6 GPU permutation 门槛被2.74e−6浮点差异触发，实际验证门槛记录为1e−5；
该量级来自 view reduction / CUDA FP32 运算。CPU synthetic tests 保持2e−6 absolute tolerance。

最佳 checkpoint 的整个 val permutation max=4.91738319e-07、
mean=6.70214761e-10；同步所有 input view fields 与 geometry。

## 20. 单元测试与冻结完整性

**31 tests 全部通过**（11.437s）：22 个 data/projector/audit tests 与9个 model/loss/engine tests。
覆盖 XYZ endpoint grid、TDLAS synthetic angle、horizontal/vertical grid_sample 顺序、有限梯度、
output shape/range、padding、permutation、无可见 view 零融合、loss floor、pooled metrics、
partial gradient accumulation equivalence、checkpoint 与 Python/NumPy/Torch/loader RNG replay。

原 generation QA read-only regression passed：1189 formal artifacts、
714 source files、474 projection arrays。
实验前后 frozen dataset 1262 files 和 source 714 files
inventory 完全一致：**passed=True**。比较 size、mtime_ns、inode、SHA256；
历史 source inputs/ 仅做 stat inventory，与旧 QA 定义一致。冻结数据/源GT/split/统计/几何均未写入。

## 21–22. 限制与下一步

B0 在这个静态末帧数据集上完成了纯监督链路验证；仅一个 seed，未验证 test 或 Pool B 泛化，
有限训练 case 的 plume prior 仍可能影响输出。32³ lifting 和3D卷积可能平滑高梯度/细小 plume，
峰值与边界误差须结合每 case 图像分析。共享encoder/mean fusion没有学习 view confidence。
固定随机种子确保 station choices 可重放；CUDA grid_sample backward 不承诺逐 bit 训练轨迹一致。

已知 C64 projection mismatch 保持原审计结论：P_diff(C128,96) vs P_ref(C128,256)
relative L1/L2≈0.095820%/0.315723%；C64≈1.399296%/3.077827%。本轮没有修改 projector
或用 projection loss 掩盖该问题。下一阶段先冻结 B0 config，做多 seed、按条件分层 val 分析；
再按独立预定评估协议测试 sparse-view 数量与 Pool B 外推。C64/C128物理离散一致性问题另立实验，
解决并验证后再讨论 measurement-domain loss。

## Artifact、复现与 Git

所有 runtime artifacts 位于 `/data/djc/GeoPhy-Set3D/outputs/`，不进入 Git：

- `logs/supervised_b0/`：effective_config.yaml、environment.json、train.log、metrics.jsonl、
  gpu_smoke.json、evaluation.json、per-case evaluations、training_summary.json、inventory_comparison.json。
- `checkpoints/supervised_b0/`：best.pt、last.pt（model/optimizer/scheduler、epoch、best metric、
  config、git commit、seed、bad_epochs、Python/NumPy/Torch/CUDA/loader RNG）。
- `figures/supervised_b0/`：training curves、fixed val best、4 val cases 的GT/pred/error slices。
- `logs/checkpoints/figures/supervised_b0_overfit/`：4-case gate、日志、last weights 和切片/曲线。

入口与运行命令见 README。`--resume` 严格要求相同 effective config，原子写入 checkpoint，
还原 optimizer/scheduler/RNG 和 early-stop 状态。完整训练入口强制检查 overfit gate 通过及核心配置一致。
代码同步 GitHub，数据、权重、日志和图片不上传；具体最终 commit 见 git log。

## 修改文件清单

```text
README.md
configs/train_supervised_b0.yaml
docs/reports/supervised_b0_v1.md
docs/reports/training_interface_v1.md
scripts/evaluate_supervised_b0.py
scripts/overfit_supervised_b0.py
scripts/smoke_supervised_b0.py
scripts/train_supervised_b0.py
scripts/validate_training_interface.py
src/geophy_set3d/data/projection_pool_dataset.py
src/geophy_set3d/data/view_sampling.py
src/geophy_set3d/models/__init__.py
src/geophy_set3d/models/blocks.py
src/geophy_set3d/models/decoder3d.py
src/geophy_set3d/models/encoder2d.py
src/geophy_set3d/models/fusion.py
src/geophy_set3d/models/geolift_mean_res3d.py
src/geophy_set3d/models/lifting.py
src/geophy_set3d/training/__init__.py
src/geophy_set3d/training/engine.py
src/geophy_set3d/training/losses.py
src/geophy_set3d/training/metrics.py
src/geophy_set3d/training/utils.py
tests/model_fixture.py
tests/test_geolift_mean_res3d.py
tests/test_geometry_lifting.py
tests/test_model_permutation.py
tests/test_projection_pool_dataset.py
tests/test_supervised_losses.py
tests/test_training_engine.py
tests/test_view_sampling.py
```
