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
