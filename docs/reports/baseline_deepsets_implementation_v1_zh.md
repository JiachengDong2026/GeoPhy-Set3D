# 射线积分 DeepSets 基线实现记录 v1

本实现为标准 shared-MLP + 对称池化 + 独立解码器的 DeepSets 基础版本，实验名为 `ray_deepsets_v1`。没有复用最终主模型的可训练 Encoder、GeometryLifting、交互层、MRGR、融合层或 PRF。本文件仅记录实现和 CPU 验证，不声称已完成 GPU 试运行、正式训练或重建性能比较。

## 来源和代码许可边界

算法来源为 Manzil Zaheer、Satwik Kottur、Siamak Ravanbakhsh、Barnabas Poczos、Ruslan Salakhutdinov、Alexander Smola，*Deep Sets*，NeurIPS 2017，[arXiv:1703.06114](https://arxiv.org/abs/1703.06114)。2026-10-23 从 arXiv 摘要页核对题名、作者与排列不变学习定义。论文提出以共享映射和对称聚合构建集合函数。本基线选择常见的 **mean pooling** 变体；不声称 mean 对任意变长集合保留集合基数或具备 sum 形式定理的所有表达能力。

模型和测试均为本仓库独立编写，未复制或引入作者或其他第三方实现，因此没有新增第三方代码许可依赖。参考论文是算法引用，不能替代第三方代码许可证；本次没有搬运此类代码。固定 token 工具复用本仓库 `ray_senseiver_ss.py` 的 `ray_tokens` 与 `fourier`，不共享任何已训练权重。

## 接口和固定配置

工厂：`geophy_set3d.models.baseline_deepsets_v1.RayDeepSetsV1`。独立配置：`configs/baselines/deepsets/model_v1.json`。

| 项目 | 实现 |
|---|---|
| 输入 token | 每条有效射线 203 维 |
| shared φ | Linear(203,128) → GELU → Linear(128,128) → GELU → Linear(128,128) → GELU |
| 聚合 | masked mean，只以实际有效射线数作分母 |
| 空观测集合 | 零 latent；不计入 token 打包器的空集占位 token |
| 输出查询 | `[Q,3]` 米制 XYZ，固定 64³ 网格为各轴 `[0,0.6]` 含端点 |
| 查询编码 | XYZ 归一化到 `[-1,1]` 后，16 个线性频率、sin/cos 共 96 维 |
| 独立解码器 | `[latent128, query96]` → Linear(224,128) → GELU → Linear(128,128) → GELU → Linear(128,1) → sigmoid |
| 默认参数量 | **104,577**，全部可训练 |
| 默认查询块 | 16,384；训练时非重入 activation checkpoint |
| 完整输出 | FP32 `[B,1,64,64,64]`，CH4 质量分数 `[0,1]` |

`encode(batch)` 返回 `[B,128]`，`decode_queries(latent, queries)` 返回 `[B,Q,1]`；小查询仅用于工程测试，不会缩短正式输出网格。`forward(batch)` 始终解码完整 64³。

`output_bias_logit` 是显式工厂参数，默认 **0.0**，最终 bias 初始化为该值，其余 Linear 使用 PyTorch 默认初始化。默认初始输出约 0.5，可能与稀疏低浓度任务尺度不匹配，这应由预先登记的训练投影标尺初始化或 pilot 客观识别，不能通过测试 GT 调参。本配置没有使用 GT 标尺；若统一集成阶段预先从训练测量推导三个基线共用的输出先验，必须在正式配置和冻结记录中写入具体数值及推导来源，不能静默修改此默认值或根据测试误差搜索。

## 几何信息公平性

token 字段由 `ray_tokens` 固定提取：原始积分测量除以 `Y_SCALE=32284.894580078173`；射线入/出体积交点各 3 维；单位方向 3 维；路径长度除以 `sqrt(3)*0.6`；入点和出点各 96 维 Fourier 编码。总维数为 `1+3+3+3+1+96+96=203`。

只读取 `input_projection_raw`、`input_station_origin_m`、`input_ray_direction`、`input_t_near`、`input_path_length_m`、`input_valid_ray`、`view_mask`。未读取留出 H 测量、GT64/GT128、测试标签或其他额外射线。token 工具的递归 `assert_no_gt` 拒绝带 GT 字段的 batch。每条零积分但几何有效的射线仍参与集合；无效/补齐射线在读取特征前过滤，因此允许其存放 NaN。

MLP 编码和池化不区分射线排序，也不把视图或像素下标当作额外信息。mean pooling 对有效数量变化稳定，但平均会丢失射线重复次数信息；这属于已冻结基础架构的限制，不应在看到结果后添加复杂补偿模块。

计算量为输入端 `O(B*N*hidden²)`、查询端 `O(B*Q*hidden²)`；不构造 `N×N` 或 `Q×N` 注意力矩阵。分块 checkpoint 降低训练激活占用，但完整输出及物理投影仍需内存。具体 GPU 显存、速度和正式预算由统一入口实测，不在此估计。

## CPU 工程验证

在 `gas-torch` 环境、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2` 和 `torch.set_num_threads(2)` 下执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  /scratch/djc/envs/gas-torch/bin/python -m unittest discover \
  -s tests -p test_baseline_deepsets_v1.py -v
```

6 项测试全部通过，包含：观测排列不变；补齐视图和无效射线中 NaN 不影响输出；不同有效射线数；零观测安全输出；batch=1/2 一致；测量值变化能改变预测；分块 checkpoint 与无 checkpoint 前向逐值一致；有限且非零参数梯度；checkpoint 严格保存重载一致；拒绝 GT batch；完整 64³ FP32 有限/范围检查；通过现有可微物理算子的 observed consistency 完整输出反向；默认参数量和非法配置检查。

这些是合成合法射线的工程检查，没有读取任何数据集 GT。统一 A/B/H 完整自监督损失、两种子 pilot、GPU smoke、正式 checkpoint selector 和主模型回归由集成任务统一验证；本模型自身只提供单个场，不能将其误记为包含 PRF 的双阶段网络。
