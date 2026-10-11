# GeoPhy-Set3D v1 数据集生成执行报告

已按指定顺序阅读技术计划 v0.5 和服务器生成说明 v1.3，并完成 Debug 验证、全量 MVP 生成及冻结。
冻结时间：2026-10-10 21:02:54（Asia/Shanghai）。

## 生效需求

以 v1.3 的具体生成协议为准：固定 source root，唯一索引入口为 `manifest.jsonl`，
仅使用 frame 60 / t=2.0 s 的 128³ CH4 质量分数 GT。
v0.5 中的 0.75 m / 60° 相机模型由约 6 m 的多束 TDLAS 扫描替代。
原 `inputs/` 数组未读取，三维 GT 未重新模拟或逐样本归一化。

本次完成文档第 63.4 节规定的全部 MVP 数据：每样本 64 个 Pool A 站位、32 个独立 Pool B 站位、K_ref=256。
文档建议的 128/64 站位扩展属于后续 Final Pool 阶段。

## 生成结果

数据集位置：`/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/`。

| 项目 | 结果 |
| --- | --- |
| 源记录 / 准入 / 生成成功 | 237 / 237 / 237 |
| 训练 / 验证 / 测试 case | 166 / 36 / 35 |
| 保留方法验证样本 | 3 条，全部归入训练集 |
| Debug | 5 个训练样本，16/8 个站位，K=128 |
| MVP 投影 NPY | 474 个（每 case 两个） |
| MVP 站位测量图 | 22,752 张 |
| 每张测量图 | 64 个垂直角 × 64 个水平角 |
| 总测量数组元素 | 93,192,192（含 invalid-ray 的零占位） |
| 投影类型 / 单位 | float32 / ppm·m |
| 64³ 派生 target | 237 个 |
| GT 范围 / 裁剪 voxel | [0, 1] / 0 |
| 投影范围 | [0, 85547.7421875] ppm·m |
| 失败 / 缺文件 / NaN / Inf / 负投影 | 全部为 0 |
| 派生数据大小 | 约 616 MiB，含 Debug、几何、审计与源码快照 |

128³ GT 通过元数据引用源文件，未重复复制。64³ target 使用物理坐标端点对齐的三线性重采样，
不是有限体积 cell average；Projection Pool 始终从原始 128³ GT 生成。

## 几何与物理约定

站位在水平 x-z 半环，方位角 0°–180°、半径 5.5–6.5 m、高度 0.25–0.35 m。
中心方向指向 (0.3,0.3,0.3) m。
水平 steering 为 linspace(-5°,5°,64)，垂直 steering 为 linspace(-4°,4°,64)。
数组行对应垂直角，列对应水平角；行递增为局部向上，列递增为局部向右。

方向公式固定为球面 yaw/elevation：

```text
d = cos(beta) cos(alpha) forward
  + cos(beta) sin(alpha) right
  + sin(beta) up
right = normalize(cross(forward, world_up))
up = normalize(cross(right, forward))
```

只积分 ray 与 [0,0.6]³ 的交段。无效 ray 的 near/far/path length 均为零，并保存显式 valid_ray mask。
Pool A / Pool B 有效 ray 比例约为 54.165% / 54.290%，最大交段约为 0.8341 / 0.8346 m。
64 为偶数，实际角度数组不含零角；中心轴通过 center_direction 单独记录。

参考投影器按“先插值质量分数、再转摩尔分数、再积分”的顺序运行。
摩尔质量为 16.043 / 28.965；每条有效 ray 使用 256 个端点采样、梯形积分、float64 累积，
结果乘以 10^6 后存为 float32。实现使用 C++ / OpenMP，12 个 CPU 线程、4 站位分块。
可微对照采用独立 PyTorch grid_sample、96 点 midpoint 积分，验证了 autograd。

## 验证与审计

以下检查全部通过：零场、常量场、质量分数转换端点和单调性、非对称 XYZ 仿射场的独立积分、
偏心球体的位置与旋转、slab 边界和轴平行射线、Gaussian 与三个真实训练 GT 的采样收敛、
独立 SciPy 投影对照、可微投影梯度、几何可视化及投影视觉检查。

采样数比较覆盖 64/128/256/512。
三个真实 GT 的 256 vs 512 相对 L1 最大为 0.00484%，相对 L2 最大为 0.01294%；
Reference vs 96 点可微投影的相对 L1 最大为 0.06616%、L2 最大为 0.32199%。
验收阈值预先保存于配置和 projector_validation.json；测试达到阈值后采用 K_ref=256。
这些数值衡量采样误差和算子差异，不代表重建网络精度。

Case、trajectory、相同几何工况以及相同 GT 内容均按组划分，跨 split 重叠为零。
Debug 和投影器验证仅使用训练 case；保留方法验证样本未进入最终测试。
Pool A / Pool B 使用独立 seed，exact station pose 重叠为零。

只读审计覆盖源目录 714 个文件的文件清单、size、mtime、inode；
targets、metadata、manifest、README 和 summary 还进行了前后 SHA-256 对照。
历史 inputs 仅检查文件属性，未读取数组内容。
源文件变化、新增和删除数量均为零。

源清单保留了原 Windows CFD 审计路径，但服务器未提供这些原审计文件。
本次依赖源 README 的已审计来源声明，并独立复查全部可用 GT / 帧 / 元数据；未声称重跑原 CFD 物理审计。
source_mode 字段在 manifest / metadata 中缺失，按协议保存 null，未从文件名推断 equal / half。
条件标识 condition_id、压力、风速、孔数及 geometry_id 已保留，可在后续评估中使用。

## 统计与训练接口

train_statistics.json 仅使用 166 个训练 case 的 Pool A 和 GT，未使用验证集、测试集或 Pool B。
分位数为全部数据的精确统计。同时保存含无效零占位的全数组统计和仅有效 ray 的统计。

固定归一化推荐尺度为训练集有效 ray 的 p99.5：

```text
Y_ref = 32284.894580078173 ppm·m
Y_scaled = Y / Y_ref
# 或 log1p(Y / Y_ref)
```

源 projection 不做缩放或离线加噪。训练时从 Pool A 在线选择输入和 held-out station，
随机打乱顺序、按需要注入噪声和 G_true / G_obs 标定扰动，并在 loss 使用 valid_ray。
Pool B 仅用于独立几何测试。另保存 0°/36°/72°/108°/144°/180° 的 canonical 几何；
canonical 投影及视角组合未预生成。

## 交付文件与复现

```text
geophy_projection_pool_v1/
  README.md
  dataset_build_manifest.json
  generation_config.yaml
  source_index.jsonl
  train_statistics.json
  splits/{train_cases,val_cases,test_cases}.txt
  splits/split_summary.json
  geometry/{train_geometry,ood_geometry,canonical_fixed_geometry}.npz
  geometry/geometry_profile.json
  samples/<case_id>/
    projections_train.npy
    projections_ood.npy
    gt_volume_64.npy
    sample_meta.json
    frame_manifest.jsonl
  audits/
    source_validation.json
    source_fingerprint_before.json
    source_read_only_audit.json
    geometry_audit.json
    projector_validation.json
    split_leakage_audit.json
    projection_pool_audit.json
    artifact_hashes.json
    generation_failures.json
    figures/geometry_and_projection.png
  debug/
  logs/
  reproducibility/  # 生效 Python / C++ / YAML 源码快照与运行版本
```

主数据和几何文件共有 1189 个 SHA-256 索引项，另记录配置、source manifest、source index、
split、统计文件、源码和验收文件哈希。
随机种子：split=20261010、Pool A=20261011、Pool B=20261012。

```bash
cd /home/djc/projects/GeoPhy-Set3D
/scratch/djc/envs/gas-torch/bin/python scripts/build_projection_pool.py --phase mvp --resume
# 只重新审计、统计并刷新冻结清单：
/scratch/djc/envs/gas-torch/bin/python scripts/build_projection_pool.py --phase mvp --audit-only
```

Debug 的 --resume 路径已实际运行并验证跳过通过哈希和数组审计的完整样本。
完整样本通过临时目录写入后 rename 发布；失败样本会清理半成品并记录异常。
生效配置发生变化时程序拒绝沿用该版本，需升级数据集版本。
