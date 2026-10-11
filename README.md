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
