# Fluent 数据存储与入口

## 物理存储位置

```text
/data/djc/GeoPhy-Set3D/
├── data/
│   ├── raw/
│   │   └── Fluent/
│   │       └── static_final_frame_partial_v1/
│   │           ├── README.md
│   │           ├── dataset_summary.json
│   │           ├── manifest.jsonl
│   │           ├── inputs/
│   │           ├── targets/
│   │           └── metadata/
│   └── processed/
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

Fluent 数据从 `/data/djc/Fluent` 整体迁移到以上位置；原路径已改为符号链接，仍可访问同一份文件。
迁移前后核对了 714 个文件的相对路径、文件系统标识、inode、大小与修改时间，均一致。
数据文件总计 2,019,906,881 字节，约 1.88 GiB。

## 数据集说明

以下信息来自数据集自带的 `dataset_summary.json` 和 `README.md`：

| 项目 | 值 |
| --- | --- |
| 数据集标识 | `clean300_final_frame_static_partial_v1` |
| 样本数量 | 237 |
| 观测输入 | `inputs/<sample_id>.npy` |
| 输入形状 | `[view, row, column] = [8, 64, 64]` |
| 输入类型与单位 | `float32`，`ppm_m` |
| 三维目标 | `targets/<sample_id>.npy` |
| 目标形状 | `[x, y, z] = [128, 128, 128]` |
| 目标类型与物理量 | `float32`，CH4 质量分数 |
| 时间帧 | 第 60 帧，物理时间 2.0 秒 |
| 样本索引 | `manifest.jsonl` |

这是由 Fluent 生成并导出的 NumPy 数据集。数据集说明指出部分数组采用硬链接，
应以只读方式使用，处理结果另存于 `processed/`。

## 项目访问路径

- 原始数据根目录：`data/raw/`
- 当前数据集：`data/raw/Fluent/static_final_frame_partial_v1/`
- 处理后数据：`data/processed/`
- 实验输出：`/data/djc/GeoPhy-Set3D/outputs/`

相对路径以 `/home/djc/projects/GeoPhy-Set3D/` 为起点。
符号链接是本机存储配置，不随 Git 分发；其他机器需要按实际存储位置重建链接或调整配置。
