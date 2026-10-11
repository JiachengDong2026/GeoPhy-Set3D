# GeoPhy-Set3D

基于物理约束的三维气体羽流重建研究项目。

当前阶段：已实现 v1.3 协议要求的静态末帧数据集生成流程，包括独立 TDLAS 几何、参考投影器、物理验证、case 划分、断点续跑和冻结审计。

## 开发环境

已有 Conda 环境：`gas-torch`，位于 `/scratch/djc/envs/gas-torch`。

```bash
source /data/apps/miniconda3/etc/profile.d/conda.sh
conda activate gas-torch
cd /home/djc/projects/GeoPhy-Set3D
```

环境采用 Python 3.12.15、PyTorch 2.11.0 和 CUDA 12.8，已验证 RTX 4090 上的 GPU 运算。
`requirements.txt` 保存初始化时的完整 pip 依赖版本，包含训练、科学计算和 Jupyter 工具。
Notebook 可选择 `Python (gas-torch)` 内核。

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `src/geophy_set3d/data/` | 观测数据读取、预处理和数据集 |
| `src/geophy_set3d/geometry/` | 三维坐标、空间离散与观测几何 |
| `src/geophy_set3d/physics/` | 物理方程、边界条件和约束项 |
| `src/geophy_set3d/models/` | 模型组件 |
| `src/geophy_set3d/reconstruction/` | 重建与优化流程 |
| `src/geophy_set3d/visualization/` | 三维场和实验结果可视化 |
| `configs/` | 数据、物理参数和实验配置 |
| `scripts/` | 数据处理、训练、重建和评估入口 |
| `notebooks/` | 数据探索与方法验证 |
| `docs/` | 项目规划、研究笔记、参考资料和报告 |
| `data/raw/` | 原始数据入口 |
| `data/processed/` | 处理后数据入口 |
| `outputs/` | 模型权重、运行日志和图像输出 |

实际数据存放在 `/data/djc/GeoPhy-Set3D/data/`，项目的 `data/raw` 和 `data/processed` 是指向该位置的符号链接。
已有 Fluent 数据集可通过 `data/raw/Fluent/static_final_frame_partial_v1/` 访问，目录说明见 `docs/data-layout.md`。
原路径 `/data/djc/Fluent` 已保留为兼容链接。

实验输出存放在 `/data/djc/GeoPhy-Set3D/outputs/`；项目的 `outputs/checkpoints`、`outputs/logs` 和 `outputs/figures` 分别链接到对应目录。
`configs/example.yaml` 已配置数据集入口和外部输出路径。
实际数据、实验输出和本机符号链接不纳入 Git。

## 数据集生成

生效配置为 `configs/dataset_generation_v1.yaml`，执行入口为 `scripts/build_projection_pool.py`。
唯一数据源是既有静态子集的 `manifest.jsonl`；只读取 frame 60 / 2.0 s 的三维 GT，忽略原 `inputs/`。
派生数据写入 `/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/`，源数据保持只读。

```bash
python scripts/build_projection_pool.py --phase debug --resume
# 检查 debug/audits/figures/geometry_and_projection.png，并在 debug/debug_acceptance.json 记录视觉验收。
python scripts/build_projection_pool.py --phase mvp --resume
```

Debug 为 5 个训练样本、16/8 个站位、K=128；MVP 为全部准入样本、64/32 个站位、K=256。
每个站位包含 64×64 条独立 TDLAS 激光束；扫描半径约 6 m，水平 ±5°、垂直 ±4°。
原始投影保存为 float32 `ppm·m`；共享真实几何、valid-ray mask、64³ target、源 GT 引用和元数据一并保存。
在线视角组合、噪声和标定误差留给训练流程生成。

划分按 case / trajectory / 相同几何工况 / GT 内容重复项分组，再平衡可用工况字段。
保留的方法验证样本固定到训练集，投影器验证和 Debug 也仅使用训练集。
冻结后的配置、哈希、统计量和审计见数据集根目录；执行报告见 `docs/reports/dataset_generation_v1.md`。
`configs/example.yaml` 仍仅为早期配置结构示例。

## 训练数据接口与 64³ 投影审计

训练数据配置为 `configs/training_data_v1.yaml`。`ProjectionPoolDataset` 只从冻结的
`source_index.jsonl` 确定 membership，通过 mmap 读取选择的 station，默认不读取原始 128³ GT。
训练只允许 `split="train", geometry_pool="train"`（Pool A）；`val/test + train` 表示羽流泛化，
`val/test + ood` 表示独立几何评估。所有 split 共用训练集有效 ray 的 p99.5 归一化尺度。

```python
from torch.utils.data import DataLoader
import torch
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate

dataset = ProjectionPoolDataset(
    "/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1",
    split="train", geometry_pool="train", base_seed=20261011,
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                    collate_fn=projection_pool_collate,
                    generator=torch.Generator().manual_seed(20261011))
dataset.set_epoch(0)  # 每个 epoch 创建 iterator 前设置；支持 persistent_workers。
batch = next(iter(loader))
```

直接导入时需设置 `PYTHONPATH=src`。每个样本返回 3–6 个随机 input station 和 1 个不重叠的
held-out station，保留原始 `ppm·m`、固定尺度归一化投影、valid-ray mask、路径长度和冻结几何。
GT64 是 `[1,64,64,64]` XYZ 质量分数；`load_gt128=True` 才加载原始 GT。
Collate 按 batch 最大 view 数补零，返回 `view_mask` / `hold_view_mask`；station ID padding 为 -1。
`view_mask` 表示实际 view，`valid_ray` 表示实际射线穿过域，两者必须分别使用。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/validate_training_interface.py
python scripts/validate_frozen_generation.py
python scripts/audit_c64_projection_mismatch.py --samples 96 --split train
```

审计对全部 166 个训练 case 的相同 Pool A 计算 `P_diff(C128,96)` 和 `P_diff(C64,96)`，
与冻结的 `P_ref(C128,256)` 比较，正式指标仅使用 valid rays。默认采用 CPU / float64 / 8 线程。
输出在 `/data/djc/GeoPhy-Set3D/outputs/audits/`，项目 `outputs/audits` 是该目录的本机链接。
`validate_frozen_generation.py` 将原有几何、物理、收敛和完整性 QA 的结果写到外部，
适合检查冻结数据；原生成入口的 `--audit-only` 会刷新冻结文件，本轮不使用。
接口、复现规则、审计数值和测试结果见 `docs/reports/training_interface_v1.md`。

## GeoLift-Mean-Res3D / Supervised B0

站点采样只依赖 seed、epoch、sample index 和 case_id；num_workers=0/2/4 得到相同 station IDs
及顺序，支持 persistent workers。B0 使用共享 2D encoder、显式 XYZ→扫描角 lifting、
可见性 masked mean 与残差 3D decoder，输出 `[B,1,64,64,64]` CH4 质量分数。
训练仅使用 train/Pool A 的 voxel SmoothL1 + 0.1×relativeL1；固定四视图 val 选择 best checkpoint。

在已安装 torch/CUDA、NumPy、PyYAML、Matplotlib 的环境中，从仓库根目录执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/smoke_supervised_b0.py
python scripts/overfit_supervised_b0.py --cases 4 --epochs 300
python scripts/train_supervised_b0.py
python scripts/evaluate_supervised_b0.py
# 需要恢复训练时：
python scripts/train_supervised_b0.py --resume /data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b0/last.pt
```

完整训练入口要求过拟合 gate 通过；实验已存在时使用 `--resume` 或更改 experiment.name。
默认 FP32、batch_size=2、AdamW、cosine decay、最多150 epochs、val patience=20。
配置在 `configs/train_supervised_b0.yaml`；日志、权重和图片全部保存到外部
`/data/djc/GeoPhy-Set3D/outputs/{logs,checkpoints,figures}/`，不提交数据和权重。
评估入口仅运行 val/Pool A，包含 zero、train-mean、错配几何、跨 case measurement、zero measurement
和同步 view permutation。test 不参与模型选择。


本轮实测：150 epochs，best 为第148个 epoch；36-case val 的 mean per-case
MAE=0.0004593、RMSE=0.0023858、relative L1=0.13289、relative L2=0.15161。
wrong geometry / wrong measurement / zero measurement 的 relative L1 为0.87744 / 1.92355 / 1.00134，
正常输入明显更好；同步 view permutation 最大差4.92e−7。31项测试通过，冻结数据 inventory 未变。
结构、完整 per-case/pooled 指标、GPU显存、过拟合、完整训练和限制见
[Supervised B0报告](docs/reports/supervised_b0_v1.md)。

## B0 Generalization Evaluation Protocol v1

泛化画像保持 B0 网络、正式150-epoch配置和旧实验输出冻结，只在新的
`/data/djc/GeoPhy-Set3D/outputs/evaluations/supervised_b0_generalization_v1/` 保存结果。
协议在 `configs/evaluate_b0_generalization_v1.yaml`：先精确复现 Val-A-V4，再评估同一 acquisition
family 的未见 Pool B station poses、V=2/3/4/5/6/8 的五 deterministic draws、
V4/V6 的 spread/random/clustered、metadata 分层和 worst cases。所有 model evaluation 使用 val。

```bash
python scripts/evaluate_b0_generalization.py --stage reference
python scripts/train_b0_multiseed.py --seed 20261012
python scripts/train_b0_multiseed.py --seed 20261013
python scripts/run_b0_horizon_probe.py
python scripts/evaluate_b0_generalization.py --stage all
python scripts/summarize_b0_generalization.py
```

三个训练 seed 使用相同 reference val station IDs；原seed20261011直接引用已有checkpoint。
Horizon probe从头初始化，以cosine T_max=220诊断更长训练预算，不改写原checkpoint/scheduler。
新run支持`--resume`；不运行test、不训练Pool B、不实现B1。
主图导出PNG/PDF，同时保留CSV、逐case/draw JSONL和source/dataset/reference完整性记录。


本轮泛化画像已完成：reference精确复现，两个新增seed完成150 epochs，独立horizon diagnostic完成220 epochs。
三seed Val-A relative L1=0.13856±0.00563，Val-B=0.13155±0.00502；measurement/geometry shortcut结论跨seed复现。
220-epoch diagnostic relative L1=0.11634，比reference改善1.655个百分点，且best仍在220上限，
因此建议先确定公平的baseline训练预算，暂不进入B1。正式B0 reference未替换，38项测试通过，
source/dataset/model/checkpoint/旧实验完整性全部通过。
完整protocol、view count、coverage、分层、worst cases、multi-seed及horizon结论见
[B0 Generalization报告](docs/reports/supervised_b0_generalization_v1.md)。

## B0/B1 architecture comparison：固定300 epochs

本阶段按负责人指定冻结300-epoch协议，不再搜索训练horizon。
新增 `configs/train_supervised_b0_300.yaml` / `configs/train_supervised_b1_300.yaml`，
不覆盖B0-150 reference或220 diagnostic。两者均FP32、batch2、AdamW lr2e−4/weight_decay1e−4、
gradient clip1、cosine T_max300，禁用early stopping；train只用Pool A/V3–6/hold0，
固定Val-A-V4 minimum supervised loss选best。新配置默认GPU3，可在新的实验启动前修改device。

B1 `GeoLift-GCA-Res3D` 复用B0的encoder/lifting/decoder，仅替换fusion：
voxel级lifted feature与归一化α/β、scanner距离和station azimuth共同生成轻量view score，
在visibility AND view_mask上做安全masked softmax。支持 `model(batch, return_attention=True)`。
B0 306,625参数；B1 308,578参数，新增fusion仅1,953参数。

```bash
python scripts/compare_b0_b1_300.py --prepare
python scripts/train_supervised.py --config configs/train_supervised_b0_300.yaml --gate-config configs/train_supervised_b0.yaml
python scripts/overfit_supervised_b1.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b1_300.yaml
python scripts/compare_b0_b1_300.py --device cuda:3
# 恢复只允许同实验、同config的last checkpoint：
python scripts/train_supervised.py --config configs/train_supervised_b1_300.yaml --resume /data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b1_gca_300/last.pt
```

比较入口要求两个run均完成300 epochs，使用冻结best checkpoint，只评估Val-A/Val-B、
Pool A的V3/V4/V6各5个deterministic draws、clustered V4五draw、wrong-geometry/zero-measurement，
以及固定少量val案例的attention/entropy/permutation/padding诊断。
始终不做test模型评估，Pool B仅用于冻结checkpoint的评估。
输出在 `/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_vs_b1_300/`，
包含summary/CSV/effective protocol、逐case/draw结果、四张主图、完整性与test证据。
完整报告由完成后的比较入口生成：`docs/reports/supervised_b1_gca_v1.md`。
单seed结构比较不等于跨seed稳定性或统计显著性结论。

本轮两个run均完成300 epochs，best分别为第298/290 epoch。
B0/B1的Val-A relative L1=0.108330/0.103431，relative L2=0.122842/0.112474；
Val-B relative L1=0.104500/0.100378，relative L2=0.125769/0.111750。
B1主表L1改善0.489941个百分点（相对4.5227%），V3改善1.115847个百分点，clustered V4改善4.461874个百分点。
五项基本GO检查全部通过，但主表L1收益偏弱，单seed不足以支持跨seed稳定或显著优势；暂不建议直接进入B2。
46项测试全部通过，frozen dataset、历史结果、B0网络源文件和数值训练代码hash均未变化。
完整设计、主表、shortcut、attention与复现证据见 [B1 GCA报告](docs/reports/supervised_b1_gca_v1.md)。

## B2：Geometry-Conditioned Set Interaction

B2 `GeoLift-GCSI-Res3D` 在每个32³ voxel内加入一层view self-attention：
geometry embedding4→16→16，feature32+geometry16投影到token32，4 heads、FFN32→64→32、pre-norm residual。
interaction输出继续使用原B1 GCA pooling，encoder/lifting/decoder与supervised loss保持一致。
没有view-index positional embedding或pairwise geometry bias，不在voxel空间做attention。
B2共319,042参数，比B1增加10,464；B=2/V6 FP32前向反向已验证可在RTX4090运行。

只训练新的B2-300/seed20261011，复用被冻结的B0/B1 checkpoints与结果：

```bash
python scripts/compare_b0_b1_b2_300.py --prepare
python scripts/overfit_supervised_b2.py --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b2_300.yaml
python scripts/compare_b0_b1_b2_300.py --device cuda:3
```

继续使用固定300-epoch architecture-comparison protocol，不使用test，Pool B仅用于冻结best后的评估。
比较输出写到 `/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_b1_b2_300/`，
包含三模型主表、V3/V4/V6、clustered、spread V4、成对improvement表、interaction矩阵、四张图与完整性记录。
新报告在完成后的比较入口生成：`docs/reports/supervised_b2_gcsi_v1.md`。
B2判断优先关注V3与clustered收益；本轮不实现B3或追加seed/超参数搜索。

B2已完成300 epochs，best为第288轮，Val-A total supervised loss=0.009892290。
Val-A relative L1/L2=0.098907/0.106767，Val-B=0.096799/0.114124；
相对B1，Val-A L1改善4.3741%，V3从0.152994降至0.147769（改善3.4151%）。
clustered V4 L1仅从0.653769降至0.652042（改善0.2642%），Val-B L2反而恶化2.1247%。
冻结的五项L1工程GO检查通过，建议保留B2并进入B3；单seed结果不能证明已学会去冗余。
56项测试、trained permutation/padding、mask/all-invalid与受保护文件完整性检查全部通过。
完整主表、成对improvement、shortcut和interaction诊断见 [B2 GCSI报告](docs/reports/supervised_b2_gcsi_v1.md)。

## B3 前置 Gate：64³ Representation Capacity Audit

直接优化64³ concentration volume，检查naive GT64的projection mismatch是否能够通过
measurement-compatible representation消除；本阶段不训练网络或实现B3。
固定12个evenly-spaced train cases，共用16个spread Pool A fit stations及16个不重叠held-out stations。
目标为saved P_ref(C_GT128,K256)，复用未修改的P_diff/K96、frozen train p99.5归一化，
sigmoid volume参数化、Adam lr0.01、最多2000步，只按fit rays的SmoothL1优化和选择iterate。
held-out只在优化后评估，不参与loss、early stop或iterate selection；未使用test或Pool B。

```bash
python scripts/audit_64_representation_capacity.py --device cuda:3
# 已完成结果只重新生成报告，不重复优化：
python scripts/audit_64_representation_capacity.py --report-only
```

输出在 `/data/djc/GeoPhy-Set3D/outputs/audits/representation_capacity_64_v1/`：
protocol、selected cases/stations、operator/naive64/optimized64的raw valid-ray metrics、
optimized volumes、voxel plausibility diagnostics、一个zero-init sanity、两张图和完整性记录。
结果是有限优化得到的fitting result，不是exact theoretical representation floor。
工程Gate看fit/held-out projection errors；voxel similarity仅作diagnostic。

本轮12个case全部完成，工程分类为 **FAIL / NO-GO for strong 64³ projection-domain constraint**。
fit relative L1 mean/median/max=0.624358%/0.495559%/1.535897%；
held-out=1.577390%/1.040984%/4.552113%，naive held-out mean为1.664341%。
fit recovered_fraction mean/median=0.695069/0.693206；所有case的fit误差下降，但3个case的held-out恶化。
结论限定于本次有限优化，不证明64³存在理论capacity floor；本轮不进入强B3 projection constraint。
63项测试通过，dataset、B0/B1/B2 checkpoints、projector源码与旧实验未修改，未使用test或Pool B。
完整protocol、raw metrics、zero-init和plausibility诊断见 [64³ Capacity Audit报告](docs/reports/representation_capacity_64_v1.md)。
