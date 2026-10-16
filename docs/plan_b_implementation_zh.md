# 方案 B：集合条件连续浓度场

本实现依据《GeoPhy_Set3D_研究开发指南_v1.0》第 8 节。调研报告提出的特征金字塔、逐步开放频率属于后续候选；首版采用指南固定的 32³ 特征网格与四个始终启用的 Fourier 频带，不混入其他变体。

## 结构与物理接口

每站三通道测量输入经共享二维编码器，再按实际站位几何提升到 32³。可见性与 view_mask 屏蔽无效站位，特征采用 sum/4 对称融合；随后三个宽 32 的三维残差块生成条件特征。编码只使用输入测量及几何，不接收 H 测量。

物理 XYZ 坐标范围为 [0,0.6] m，归一化到 [-1,1]。查询向量由 32 维插值特征、3 维坐标、24 维 Fourier 编码及 2 维覆盖描述拼接，共 61 维。MLP 有四个宽 64 的 SiLU 隐藏层，末层 sigmoid 输出甲烷质量分数。覆盖描述为实际输入站位的可见比例及归一化 Tᵀ1；覆盖不是误差或置信区间。

```python
context = model.encode(measurements)
mass_fraction = model.query(context, xyz_m)
pic_ppm_m = model.integrate(context, target_geometry)
grid64 = model.sample_grid(context, 64)
```

context 只属于当前案例与当前一次编码，不跨案例或优化步缓存。存储的特征网格轴为 XYZ；三线性采样明确转换为 PyTorch 要求的 ZYX 查询顺序。

连续投影在每条有效射线的盒内有限线段上直接查询 Q96 个中点。先查询 C(x)，再做非线性质量分数→摩尔分数转换，最后按路径长度积分，单位为 ppm·m。投影绝不从导出的 C64/C128 体场计算。无效射线和 NaN padding 在参与运算之前屏蔽；有效射线的坏几何则报错。

默认 query_chunk=32768、ray_chunk=256。完整查询块（包括特征插值、MLP、χ 和归约）使用非重入梯度检查点重计算，保留对编码器的完整梯度。几何缓存只保存 detached 射线参数，实行有界 LRU；不缓存依赖模型参数的浓度或特征。

## 训练边界

数据与 A 保持一致：166 train / 36 val；Pool A 训练输入 A、B 各 3–5 站，H 为 2 站，三者互斥。验证固定 epoch=0、A/B/H 各 4 站。H 仅用于损失和选模。禁止读取体场 GT、最终测试集、Pool B 几何及外部监督检查点。

损失为 hold + 0.5 cross + 0.1 observed + 0.05 consistency。测量差异使用固定尺度 32284.894580078173、SmoothL1 β=0.01 与 0.1 倍 relative-L1。一致性在两个 context 的同一个固定 64³ XYZ 端点网格上计算，两支均反传。每案例先归约，再对案例平均。

初版采用 FP32、AdamW(lr=2e-4, weight_decay=1e-4)、梯度裁剪 1、有效 batch=2，尾组按实际案例数归一化。完整轮次后推进 cosine 调度；保存优化器、调度器、Python/NumPy/Torch/CUDA/loader RNG 及下次采样轮次，只允许从完整轮次恢复。唯一选模指标是验证留出测量平均 relative-L1。

首轮默认随机末层 bias 产生约 0.47 的初始浓度，20 次更新后的留出 relative-L1 仍为 246.737，失败记录保留。当前版只校准末层 bias：对完整 166 个训练案例、Pool A 全部 64 站的有效投影，使用 FP64 计算 `χ₀=ΣY/(10⁶Σℓ)`，再按物理关系逆变换得到 `C₀`，初始化 bias=`logit(C₀)`。实数为 χ₀=0.006776407923745372、C₀=0.003764666472415067。隐藏层及最后一层随机权重保持原值，未读取验证或 GT。该量只是采集加权的全局浓度先验，不是三维重建。

这项尺度统计在 smoke/gate/formal 中固定使用完整训练集，不随 2/16 案例优化子集变化；因此小规模验收只指参与梯度更新的案例数量。初始化的额外训练投影读取与哈希完整记录在 `initialization.json` 和 provenance 中，不能描述成整个流程只看过 2/16 个案例。恢复时从检查点加载，不重复覆盖 bias。

正式设置保留 300 轮、种子 20261011/12/13；本任务只进行实现与小规模验收，不启动方案 B 正式训练。正式模板需要本版真实 smoke、gate 和来源一致性证据才能晋级。

## 与正在运行的 A 隔离

B 代码位于 `research/plan_b/`，命令入口为 `scripts/train_plan_b.py` 和 `scripts/reconstruct_plan_b.py`。不往 `src/` 添加文件，不修改 A 冻结源码及其启动器，确保 A 的源身份与恢复条件不变。B 复用既有数据、二维编码器和 lifting 的只读导入，自行记录源代码与数据哈希。

正式 B 训练和 A 的耗时不能直接互推。B 的主要计算量来自大量连续坐标查询及重计算，必须单独实测。小规模 GPU 验证使用其他卡，避开 A 所在 GPU 0。

## 验证与使用

数值、接口、梯度、真实测量学习及耗时结果见 `docs/reports/plan_b_v1_zh.md`。64³ 是主要公平比较输出；128³ 仅为同一连续函数的更密采样，不能作为恢复真实高频信息的证明。独立三维 GT 评价必须等正式训练完成并冻结后另行开展。

```bash
# 真实2案例/20更新；默认目录已存在时使用新的 --run-name
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/djc/envs/gas-torch/bin/python \
  scripts/train_plan_b.py --config configs/research/scheme_b_smoke.yaml --device cuda:2

# 16案例/5轮/40更新 gate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/djc/envs/gas-torch/bin/python \
  scripts/train_plan_b.py --config configs/research/scheme_b_gate.yaml --device cuda:2

# 外部测量 NPZ -> C64 与直接连续重投影；可选 --resolution 128
/scratch/djc/envs/gas-torch/bin/python scripts/reconstruct_plan_b.py \
  --run-dir outputs/research/scheme_b_smoke_calibrated \
  --input measurements.npz --output reconstruction_b --device cuda:2
```

中途恢复用同一配置和 `--resume`，只接受完整轮次检查点；不改变已解析配置。`--stop-after-epochs` 供可重放检查使用，未到配置轮数不会生成完成冻结记录。

`scripts/verify_plan_b_checkpoint.py` 在固定 context 上比较 Q96/192/384，以及 C64 体素投影与连续投影的偏差。`scripts/evaluate_plan_b.py predict/score` 只允许完整 300 轮正式冻结后的验证集开发评价：先保存并校验全部预测，后独立读取 GT64。它保留最终测试集锁定；当前未运行 GT 评价。正式配置默认 `design_template_not_executable`；即使显式晋级，也必须验证本版真实 smoke/gate 已通过，不能复用 A 的证据。
