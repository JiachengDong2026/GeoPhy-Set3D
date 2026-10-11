# GeoPhy-Set3D v0.5
## 基于随机稀疏视角、Projection Pool 与测量物理一致学习的气体羽流三维重建技术计划

**文档版本**：v0.5  
**研究对象**：甲烷羽流三维质量分数场重建  
**核心定位**：Geometry-generalizable + Variable-cardinality + Measurement-physics-consistent + Direct volumetric reconstruction  
**建议工作名称**：GeoPhy-Set3D

> **Markdown / 公式统一规范**
>
> - 正文变量统一使用反引号，例如 `Y_i`、`G_i`、`C_hat`、`V`。
> - 简单尺寸直接写成 `64 × 64`、`128 × 128 × 128`。
> - 所有数学关系统一使用独立公式块。
> - 公式块上下必须留空行。
> - 不使用 `$...$` 行内公式。
> - 不在 Markdown 表格单元格中嵌套 LaTeX。
> - 张量尺寸、配置、接口和伪代码统一使用代码块。
>
> 本版本明确规定：**现有数据集仅作为三维 Fluent Ground Truth 数据源使用，不使用其已经生成的固定 8 视角投影作为本研究正式输入。所有二维投影、射线几何、Projection Pool、噪声和几何扰动均由本项目重新生成。**

---

# 1. 研究目标

已有工作包括：

1. 基于经典层析算法的三维气体重建与测量几何分析；
2. 固定视角条件下，基于 Bi-LSTM、注意力和 CBAM 的二维切片监督重建，并通过二维切片堆叠形成三维结果。

第三篇工作的目标是进一步解决以下问题：

- 固定测量角度依赖；
- 固定输入视角数量依赖；
- 固定视角顺序依赖；
- 逐二维切片重建造成的三维空间建模不足；
- 对完整三维逐体素标签监督的依赖；
- 训练几何与测试几何不一致时的泛化问题。

最终目标是建立统一三维重建函数：

$$
\hat{C}
=
F_{\theta}
\left(
\left\{
Y_i,
G_i
\right\}_{i=1}^{V}
\right)
$$

其中：

- `Y_i`：第 `i` 个视角的二维路径积分甲烷信号；
- `G_i`：对应测量几何；
- `V`：输入视角数量，可变化；
- `C_hat`：预测的三维甲烷质量分数场。

---

# 2. 核心科学问题

## 2.1 Geometry Generalization

训练阶段未出现的非均匀角度组合，是否能够由同一个模型直接重建？

重点不是简单改变固定角度，而是研究：

```text
Seen Geometry Combination
        ↓
Unseen Irregular Geometry Combination
```

时模型性能是否保持稳定。

## 2.2 Variable-cardinality Reconstruction

同一个 checkpoint 是否能够同时处理：

```text
V = 3
V = 4
V = 5
V = 6
V = 7
V = 8
```

而不需要针对不同输入数量重新训练？

## 2.3 3D-label-free Measurement-domain Learning

训练阶段是否能够在不使用逐体素三维 GT loss 的条件下，通过：

```text
3D Reconstruction
      ↓
Differentiable Projection
      ↓
2D Measurement Consistency
```

学习稳定的三维重建器？

## 2.4 Direct Volumetric Reconstruction

直接三维模型是否能比逐二维切片独立重建更好地利用：

- 垂直方向连续性；
- plume 空间连通性；
- 跨高度相关性；
- 三维形态先验。

---

# 3. 方法创新重新定位

本工作不将以下内容单独作为主要创新：

```text
2D Encoder
→ Geometry Backprojection / Lifting
→ 3D Decoder
```

该结构作为几何感知三维重建的基础组件。

主要创新聚焦在以下三个方面。

## 3.1 Innovation 1：Geometry-generalizable Reconstruction

通过大规模随机测量几何生成和独立 Geometry-OOD 测试，使网络学习：

```text
Measurement + Geometry
          ↓
     3D Volume
```

而不是：

```text
Fixed View Index
        ↓
Fixed Reconstruction Mapping
```

## 3.2 Innovation 2：Variable-cardinality Geometry-conditioned Set Fusion

将视角集合表示为：

$$
\mathcal{S}
=
\left\{
h_1,
h_2,
\ldots,
h_V
\right\}
$$

而不是固定长度序列。

网络需要满足：

- permutation invariance；
- variable view count；
- 无固定 view index；
- 显式跨视角互补关系建模。

## 3.3 Innovation 3：Masked Measurement-domain Learning

训练时将随机测量集合拆成：

```text
Input Views
Held-out Views
```

网络只能访问 `Input Views`。

预测三维体需要同时解释：

```text
Input Measurements
+
Held-out Measurements
```

建议论文使用：

> **3D-label-free projection-domain learning**

或：

> **Masked-view measurement supervision**

---

# 4. 数据集在本研究中的使用方式

## 4.1 只使用三维 GT

本研究只使用：

```text
CH4 Mass Fraction Volume
shape = [128, 128, 128]
dtype = float32
```

不使用已有：

```text
fixed 8-view PIC
historical ray geometry
view00 ~ view07
```

作为正式训练输入。

## 4.2 三维 GT 物理量

三维标签定义为 CH4 质量分数：

$$
Y
$$

其物理范围为：

$$
0
\leq
Y
\leq
1
$$

网络最终输出也定义为 CH4 mass fraction，而不是 ppm、ppm·m 或任意无物理意义的归一化浓度。

## 4.3 三维空间范围

统一 reconstruction domain：

```text
x ∈ [0, 0.6] m
y ∈ [0, 0.6] m
z ∈ [0, 0.6] m
```

GT 网格：

```text
128 × 128 × 128
```

轴顺序：

```text
[x, y, z]
```

每轴使用：

```text
linspace(0, 0.6, 128)
```

因此采样间隔：

$$
\Delta
=
\frac{0.6}{127}
$$

约为：

```text
4.724409 mm
```

必须注意：

> 该网格是包含端点的规则采样网格，不能未经定义直接当成 128 个等宽有限体积单元中心。

---

# 5. 数据划分

## 5.1 Split 最小单位

必须按独立 CFD trajectory / simulation case 划分。

禁止：

```text
同一 trajectory 的 frame 10 → train
同一 trajectory 的 frame 40 → test
```

## 5.2 两阶段使用时间帧

### Phase A：MVP

优先使用每条轨迹 `t = 2 s` 末帧，用于快速验证：

- projector；
- lifting；
- network；
- variable-view；
- loss。

### Phase B：正式训练

可使用训练轨迹中的多个非零时间帧：

```text
t = 1/30 s
...
t = 2 s
```

但必须明确：

> 多时间帧增加训练样本数，不增加独立 CFD case 数。

验证和测试建议按 trajectory 汇总。

---

# 6. 新数据生成总流程

```text
                Fluent CFD
                    ↓
         128³ CH4 Mass Fraction GT
                    ↓
             Case-level Split
                    ↓
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
Train Geometry Pool       OOD Geometry Pool
       │                         │
       ▼                         ▼
 High-accuracy Reference Projector
                    │
                    ▼
          Offline Projection Pool
                    │
                    ▼
      Random Variable-view Sampling
                    │
              ┌─────┴─────┐
              ▼           ▼
          Input Views   Held-out Views
              │
              ▼
          GeoPhy-Set3D
              │
              ▼
            C_hat
              │
              ▼
       Differentiable Projector
              │
        ┌─────┴─────┐
        ▼           ▼
      L_obs       L_hold
```

---

# 7. 新测量几何定义

由于不使用历史固定 8-view geometry，本研究重新定义统一、物理上自洽的 acquisition geometry。

## 7.1 坐标约定

```text
x: downwind direction
y: vertical direction
z: crosswind direction
```

重建域中心：

$$
c
=
\left(
0.3,
0.3,
0.3
\right)
$$

## 7.2 基础半圆扫描

第一版 sensor 放置在：

> 以 `y` 轴为竖直轴的水平 `x-z` 半圆上。

sensor 基准高度：

```text
y = 0.30 m
```

基础半径：

```text
R = 0.75 m
```

方位角：

```text
theta ∈ [0°, 180°]
```

sensor 位置：

$$
p
\left(
\theta
\right)
=
c
+
\left[
R \cos \theta,
0,
R \sin \theta
\right]
$$

光轴始终指向 reconstruction center。

相机竖直方向与世界坐标 `+y` 对齐。

---

# 8. Random Geometry Sampler

## 8.1 MVP 范围

```text
azimuth:
    Uniform(0°, 180°)

radius:
    0.75 m ± 10%

height:
    0.30 m ± 0.03 m

horizontal FOV:
    60°

vertical FOV:
    60°

projection:
    64 × 64
```

## 8.2 Geometry Mixture

候选 geometry 不只使用完全均匀随机。

推荐：

```text
60%:
    irregular random

20%:
    near-uniform + angular jitter

20%:
    difficult geometry
    including clustered / limited-span cases
```

## 8.3 后期 Hard Geometry

可逐步加入：

```text
small angular span
clustered views
radius variation
height variation
small pointing variation
```

MVP 优先随机：

```text
azimuth
radius
height
```

---

# 9. Ray Geometry

每个 detector pixel 对应一条 ray：

$$
r
\left(
t
\right)
=
o
+
t d
$$

其中：

- `o`：ray origin；
- `d`：单位方向向量。

计算 ray 与：

```text
[0, 0.6]³
```

的交点：

```text
t_near
t_far
```

满足：

```text
t_far > t_near
```

才是 valid ray。

保存：

```text
ray_origin
ray_direction
path_length
valid_ray
```

---

# 10. Projection 的物理定义

网络 GT 是 CH4 mass fraction，而二维 measurement 定义为甲烷摩尔分数路径积分，单位 `ppm·m`。

因此不能直接积分质量分数。

## 10.1 Mass Fraction → Mole Fraction

设：

```text
M_CH4 = 16.043
M_air = 28.965
```

摩尔分数：

$$
\chi
\left(
Y
\right)
=
\frac{
Y / M_{\mathrm{CH4}}
}{
Y / M_{\mathrm{CH4}}
+
\left(
1-Y
\right)
/ M_{\mathrm{air}}
}
$$

## 10.2 Path-integrated Measurement

$$
PIC
=
10^6
\int_{\mathrm{ray}}
\chi
\left(
Y
\left(
x
\right)
\right)
\,dl
$$

单位：

```text
ppm·m
```

完整链：

```text
CH4 Mass Fraction
       ↓
Mole Fraction
       ↓
Ray Sampling
       ↓
Path Integration
       ↓
× 10^6
       ↓
ppm·m
```

---

# 11. Reference Projector

Reference Projector 只用于离线生成 Projection Pool，不要求 autograd。

目标：

- 数值精度高；
- 与训练 projector 不完全相同；
- 单位严格正确；
- 可重复。

## 11.1 MVP 实现

推荐：

```text
128³ GT
    ↓
High-resolution Interpolation
    ↓
Dense Ray Sampling
    ↓
Trapezoidal Integration
    ↓
PIC
```

设置：

```text
K_ref = 256
```

后续可增加到：

```text
K_ref = 512
```

## 11.2 更高精度扩展

可尝试：

```text
128³ GT
    ↓
tricubic / spline upsampling
    ↓
256³ or higher intermediate field
    ↓
dense ray integration
```

或者实现 Joseph / Siddon 类投影器。

但如果使用基于 voxel-cell 的方法，必须先明确当前规则采样点与 voxel cell 的定义关系。

---

# 12. Differentiable Projector

训练阶段使用独立 GPU differentiable projector：

```text
Predicted Volume
      ↓
Mass Fraction → Mole Fraction
      ↓
Ray Marching
      ↓
3D Trilinear grid_sample
      ↓
Path Integration
      ↓
ppm·m
```

MVP：

```text
K_train = 96
```

正式版本：

```text
K_train = 96 ~ 128
```

---

# 13. 避免 Inverse Crime

Reference Projector 和 Differentiable Projector 故意不同。

推荐：

```text
Reference:
    GT = 128³
    K_ref = 256
    high-resolution integration
    trapezoidal integration

Training:
    prediction = 64³ / 128³
    K_train = 96
    trilinear grid_sample
    midpoint / rectangle integration
```

即：

```text
P_ref != P_diff
```

目的不是制造过大的模型失配，而是避免使用完全一致的离散 forward operator 同时生成 observation 和执行 inversion。

---

# 14. Projector 单元测试

## 14.1 Uniform Cube

设置常数质量分数。

理论投影：

```text
constant mole fraction × path length
```

用于严格检查：

- 单位；
- 路径长度；
- `10^6` 系数；
- mass-to-mole 转换。

## 14.2 Compact Object

构造小立方体或球体，检查：

- 几何位置；
- 坐标轴；
- view rotation；
- projection center。

## 14.3 3D Gaussian

检查：

- 多角度连续性；
- Reference / Differentiable Projector 偏差。

## 14.4 Real Fluent Volume

统计：

```text
Relative L1
Relative L2
Max Error
```

两种 projector 不要求完全一致，但误差应稳定并明显小于主要 reconstruction error。

---

# 15. Projection Pool

对于每个三维 GT：

$$
C_n
$$

生成：

$$
\mathcal{D}_n
=
\left\{
\left(
Y_m,
G_m
\right)
\right\}_{m=1}^{M}
$$

## 15.1 MVP

```text
M_train = 64 views / volume
M_ood = 32 independent views / volume
```

## 15.2 正式版本

```text
M_train = 128
M_ood = 64
```

## 15.3 Geometry Pool A / B

### Geometry Pool A

训练分布，用于：

```text
Input Views
Held-out Views
```

### Geometry Pool B

完全独立的 OOD geometry，从不参与训练，用于：

```text
Geometry-OOD
Double-OOD
```

---

# 16. Projection Pool 存储类型

原始物理单位 Projection Pool 默认保存：

```text
float32
```

不建议直接保存未缩放 `float16`。

原因：

```text
FP16 最大有限值约 65504
```

而 `ppm·m` 数值可能超过这一范围。

如果未来为了压缩空间使用 FP16，必须：

```text
先按固定全局尺度归一化
再保存 FP16
同时保存恢复尺度
```

MVP 一律优先使用：

```text
float32
```

---

# 17. Projection Pool 目录

```text
projection_pool/
├── case_xxxx/
│   ├── volume_gt.npy
│   ├── projections_train.npy
│   ├── geometry_train.npz
│   ├── projections_ood.npy
│   ├── geometry_ood.npz
│   └── metadata.json
```

数据形状：

```text
volume_gt:
    [128, 128, 128]

projections_train:
    [M_train, 64, 64]

projections_ood:
    [M_ood, 64, 64]
```

---

# 18. Projection Pool 生成优化

所有 volume 可以共享 geometry pool。

预先计算：

```text
ray origin
ray direction
t_near
t_far
valid mask
```

然后：

```text
Geometry Pool
     ↓
Precomputed Ray Metadata
     ↓
Batch Volume Projection
```

不建议长期保存：

```text
[M, H, W, K, 3]
```

完整 sampling grid。

应按 view chunk 在线生成 sampling points。

---

# 19. 在线 Measurement Set 构造

每个 iteration 从当前 volume 的 Projection Pool 随机选择 views。

例如：

```text
selected = 7
input = 5
held-out = 1
unused = 1
```

下一次：

```text
selected = 5
input = 3
held-out = 1
unused = 1
```

---

# 20. Variable View Count

MVP：

```text
V_input ~ Uniform(3, 6)
V_hold = 1
```

正式模型：

```text
V_input ~ Uniform(3, 8)
V_hold ~ Uniform(1, 2)
```

极端条件：

```text
V = 1
V = 2
```

作为 robustness experiment。

---

# 21. View Combination 采样

不能只随机 view count，还要控制 angular coverage。

建议：

```text
50%:
    broad-span irregular views

30%:
    medium-span random views

20%:
    clustered / limited-angle hard cases
```

同时记录：

```text
angular span
minimum angular gap
mean pairwise angle
```

用于后续分析。

---

# 22. Measurement Noise

Projection Pool 保存 clean measurement。

训练时在线加入：

```text
Gaussian noise
multiplicative gain
additive offset
random ray dropout
missing scan rows / columns
```

MVP：

```text
initially clean
then SNR = 20 ~ 40 dB
```

正式 robustness：

```text
10 / 20 / 30 / 40 dB
```

---

# 23. Geometry Calibration Error

必须区分：

```text
G_true
G_obs
```

真实 measurement：

$$
Y
=
\mathcal{P}_{\mathrm{ref}}
\left(
C_{\mathrm{GT}};
G_{\mathrm{true}}
\right)
$$

网络获得：

$$
G_{\mathrm{obs}}
=
G_{\mathrm{true}}
+
\Delta G
$$

网络输入：

```text
Y
G_obs
```

而不是用 `G_obs` 重新生成新的 `Y`。

后续分别测试：

```text
position error
angle error
height error
pointing error
```

---

# 24. 输入归一化

不允许：

```text
每张 projection 单独除以自身最大值
```

因为会破坏绝对强度信息。

推荐固定训练集尺度：

$$
Y_{\mathrm{scaled}}
=
\frac{Y}{Y_{\mathrm{ref}}}
$$

其中 `Y_ref` 只从训练集统计。

也可使用：

$$
Y_{\log}
=
\log
\left(
1
+
\frac{Y}{Y_{\mathrm{ref}}}
\right)
$$

MVP 比较：

```text
Fixed Linear Scaling
vs
log1p Scaling
```

---

# 25. 网络输入

```text
projection:
    [B, Vmax, 1, 64, 64]

geometry:
    structured geometry

view_mask:
    [B, Vmax]
```

geometry 最终转换成：

```text
camera pose
camera intrinsics
ray representation
```

---

# 26. Shared 2D Encoder

所有视角完全共享：

$$
F_i
=
E_{2D}
\left(
Y_i
\right)
$$

建议：

```text
Input:
    1 × 64 × 64

Stage 1:
    32 × 32 × 32

Stage 2:
    64 × 16 × 16

Stage 3:
    128 × 8 × 8
```

MVP 使用：

```text
64 × 16 × 16
```

做 lifting。

---

# 27. 不使用固定 View Index

禁止固定：

```text
view0 embedding
view1 embedding
...
```

网络只通过：

```text
projection content
+
geometry
```

识别测量方向。

每次 training iteration 对 view 顺序随机 shuffle。

---

# 28. Geometry-aware 3D Lifting

使用：

```text
32 × 32 × 32
```

coarse voxel grid。

对 voxel center：

$$
x_j
=
\left(
x_j,
y_j,
z_j
\right)
$$

投影到第 `i` 个 view：

$$
\left(
u_{ij},
v_{ij}
\right)
=
\Pi
\left(
x_j;
G_i
\right)
$$

从 2D feature map 采样：

$$
f_{ij}
=
\operatorname{GridSample}
\left(
F_i,
u_{ij},
v_{ij}
\right)
$$

---

# 29. Depth / Ray-aware Geometry Feature

路径积分图中，同一 ray 上不同深度 voxel 对应相同 2D pixel。

因此 lifting 不能只使用 2D sampled feature。

必须加入：

```text
sensor-to-voxel distance
normalized ray depth
voxel xyz
ray direction
camera optical direction
relative angle
```

记为：

$$
r_{ij}
$$

最终：

$$
h_{ij}
=
\operatorname{MLP}
\left(
\left[
f_{ij},
r_{ij}
\right]
\right)
$$

这是避免同一射线所有 voxel 特征完全一致的关键设计。

---

# 30. Variable-view Set Fusion

## 30.1 B0：Mean Fusion

$$
z_j
=
\frac{1}{V}
\sum_{i=1}^{V}
h_{ij}
$$

模型名称建议：

> **GeoLift-Mean-Res3D**

## 30.2 B1：Attention Pooling

$$
a_{ij}
=
w^{T}
\sigma
\left(
W h_{ij}
\right)
$$

$$
\alpha_{ij}
=
\operatorname{Softmax}_{i}
\left(
a_{ij}
\right)
$$

$$
z_j
=
\sum_i
\alpha_{ij}
h_{ij}
$$

## 30.3 Full：Geometry-conditioned Set Interaction

```text
Per-voxel View Tokens
        ↓
Set Self-Attention
        ↓
Geometry-aware Pair Interaction
        ↓
Attention Pooling
```

pairwise geometry 包括：

```text
relative azimuth
sensor baseline
relative direction
angular separation
```

用于学习 view complementarity。

---

# 31. 3D Decoder

MVP：

```text
32³ × C
   ↓
Residual 3D Block × 2
   ↓
Upsample
   ↓
64³
   ↓
Residual 3D Block × 2
   ↓
Conv3D(1)
```

---

# 32. 输出定义与激活

输出是 CH4 mass fraction，因此使用：

$$
\hat{C}
=
\operatorname{Sigmoid}
\left(
C_{\mathrm{raw}}
\right)
$$

保证：

$$
0
\leq
\hat{C}
\leq
1
$$

不再默认使用 `Softplus`。

---

# 33. 输出分辨率路线

## MVP

```text
GT:
    128³

Supervised Target:
    64³ downsampled volume

Lifting:
    32³

Output:
    64³
```

Reference Projection Pool 始终由原始：

```text
128³ GT
```

生成。

## Final Model

```text
Lifting:
    32³

Intermediate:
    64³

Output:
    128³
```

避免长时间维持：

```text
128³ × high channel count
```

只在 decoder 尾部提升到 `128³`。

---

# 34. Loss

主损失：

$$
\mathcal{L}
=
\lambda_{\mathrm{obs}}
\mathcal{L}_{\mathrm{obs}}
+
\lambda_{\mathrm{hold}}
\mathcal{L}_{\mathrm{hold}}
+
\lambda_{\mathrm{tv}}
\mathcal{L}_{\mathrm{tv}}
$$

MVP 不默认使用 sparse loss。

---

# 35. Observed-view Loss

$$
\hat{Y}_i
=
\mathcal{P}_{\mathrm{diff}}
\left(
\hat{C};
G_i
\right)
$$

$$
\mathcal{L}_{\mathrm{obs}}
=
D
\left(
\hat{Y}_i,
Y_i
\right)
$$

推荐：

```text
Charbonnier
+
Relative L1
```

---

# 36. Held-out View Loss

$$
\hat{Y}_{\mathrm{hold}}
=
\mathcal{P}_{\mathrm{diff}}
\left(
\hat{C};
G_{\mathrm{hold}}
\right)
$$

$$
\mathcal{L}_{\mathrm{hold}}
=
D
\left(
\hat{Y}_{\mathrm{hold}},
Y_{\mathrm{hold}}
\right)
$$

作用：

> 约束输出三维体不仅解释输入视角，还能解释未作为网络输入的额外 measurement。

必须明确：

> `L_hold` 不能理论保证三维解唯一。

---

# 37. Projection Loss 尺度

不建议直接对 raw `ppm·m` 使用未经归一化的 MSE。

推荐相对误差：

$$
D_{\mathrm{rel}}
=
\frac{
\left\|
\hat{Y}
-
Y
\right\|_1
}{
\left\|
Y
\right\|_1
+
\epsilon
}
$$

同时可在固定 log-scaled measurement space 中计算 Charbonnier。

denominator 需要设置由训练集统计得到的 minimum floor，避免近零 projection 产生异常放大。

---

# 38. 3D Physical-space TV

$$
\mathcal{L}_{\mathrm{tv}}
=
\operatorname{mean}
\left[
\rho
\left(
\frac{\partial C}{\partial x}
\right)
+
\rho
\left(
\frac{\partial C}{\partial y}
\right)
+
\rho
\left(
\frac{\partial C}{\partial z}
\right)
\right]
$$

使用真实空间步长。

MVP：

```text
lambda_tv = 1e-4
```

---

# 39. Sparse Loss

MVP：

```text
disabled
```

简单浓度均值惩罚可能造成：

```text
peak underestimation
mass underestimation
```

只有出现明显 diffuse haze 后再做可选实验。

---

# 40. Supervised Upper Bound

正式进入 projection-domain training 前，必须训练完全监督版本。

推荐：

$$
\mathcal{L}_{\mathrm{sup}}
=
\mathcal{L}_{\mathrm{Huber}}
+
\lambda_{\mathrm{fg}}
\mathcal{L}_{\mathrm{foreground}}
$$

目的：

- 验证 2D → 3D architecture；
- 验证 lifting；
- 验证 geometry；
- 建立性能上界。

---

# 41. Weak Supervision

后续测试：

```text
0%
5%
10%
20%
100%
```

三维标签比例。

研究少量 volumetric GT 是否能减少 sparse tomography ambiguity。

---

# 42. 训练阶段

## Stage 0：Projector Validation

```text
Geometry Sampler
Ray Generator
Reference Projector
Differentiable Projector
Analytical Phantom Tests
```

## Stage 1：Projection Pool

先用：

```text
10 ~ 20 cases
```

生成：

```text
M_train = 64
M_ood = 32
```

确认正确后全量生成。

## Stage 2：Supervised B0

```text
2D Encoder
+
Geometry Lifting
+
Mean Fusion
+
3D Decoder
```

使用三维 GT。

## Stage 3：Random V + Random View Combination

```text
V = 3 ~ 6
```

## Stage 4：Attention / Set Interaction

比较：

```text
Mean
Attention Pooling
Set Interaction
```

## Stage 5：Observed Projection Training

```text
L_obs
+
L_tv
```

## Stage 6：Held-out Measurement Training

加入：

```text
L_hold
```

## Stage 7：Noise / Calibration Robustness

加入：

```text
measurement noise
G_true / G_obs mismatch
missing rays
```

## Stage 8：Final 128³ Model

```text
output = 128³
V = 3 ~ 8
M_train = 128
```

---

# 43. Curriculum

```text
Phase 1:
    V = 6
    broad angular span
    clean measurement

Phase 2:
    V = 4 ~ 6
    irregular views

Phase 3:
    V = 3 ~ 8
    SNR 20 ~ 40 dB

Phase 4:
    clustered / limited-angle

Phase 5:
    calibration perturbation
```

---

# 44. Optimizer

```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-4

scheduler: cosine
warmup_epochs: 5

amp: true
gradient_clip: 1.0

batch_size: 1-2
gradient_accumulation: 4-8

epochs: 150-300
```

---

# 45. Differentiable Projector 复杂度

近似：

$$
O
\left(
B
V
H
W
K
\right)
$$

例如：

```text
B = 1
V = 6
H = 64
W = 64
K = 96
```

sampling points：

```text
6 × 64 × 64 × 96
≈ 2.36 million
```

MVP 可接受。

---

# 46. 高分辨率 Projector 优化

只有 profiler 确认 projector 成为瓶颈后再使用：

```text
Stochastic Ray Sampling
Chunked Projection
Mixed Precision
View-wise Streaming
```

例如 loss 仅抽取：

```text
25% ~ 50% rays
```

而输入二维图仍完整进入 encoder。

---

# 47. Test-time Refinement

仅作为增强实验。

初始输出：

$$
C_0
=
F_{\theta}
\left(
S_{\mathrm{in}}
\right)
$$

refinement：

$$
C^{*}
=
\arg\min_C
\left[
\mathcal{L}_{\mathrm{proj}}
+
\lambda_{\mathrm{tv}}
\mathcal{L}_{\mathrm{tv}}
+
\lambda_{\mathrm{prior}}
\left\|
C-C_0
\right\|_2^2
\right]
$$

不建议只优化 projection error。

---

# 48. Test Split

## Split A：Plume-OOD

```text
Unseen CFD Cases
+
In-distribution Geometry
```

## Split B：Geometry-OOD

```text
Seen-style CFD Cases
+
Geometry Pool B
```

## Split C：Variable-view

```text
V = 1
2
3
4
5
6
8
```

## Split D：Double-OOD

```text
Unseen CFD Cases
+
Geometry Pool B
```

## Split E：Hard Geometry

```text
narrow angular span
clustered views
```

## Split F：Hard Sensing

```text
10 dB noise
missing rays
angle error
position error
pointing error
```

---

# 49. Geometry Generalization 的严格表述

论文中不建议笼统使用：

> arbitrary geometry

建议使用：

> **unseen irregular geometries within the defined acquisition family**

因为 geometry 仍受到：

```text
sensor radius range
height range
FOV range
half-ring acquisition family
```

约束。

---

# 50. Shortcut / Prior Reliance Test

必须检查网络是否只学习 Fluent plume prior。

| 输入条件 | 目的 |
|---|---|
| Normal measurement | 正常性能 |
| Shuffle projection between cases | 检查 observation dependence |
| Zero projection | 查看 learned prior |
| Mean projection | 检查平均 plume tendency |
| Correct projection + wrong geometry | 检查 geometry dependence |
| Correct geometry + wrong projection | 检查 measurement dependence |

如果错误 projection 下仍输出接近 GT 的 plume，则说明模型过度依赖训练集 morphology prior。

---

# 51. 数据分布偏差分析

即使重新生成 projection，原 CFD case 本身仍可能存在：

```text
geometry
wind
pressure
source pattern
```

的不均衡。

最终结果必须按：

```text
geometry family
wind speed
pressure
equal / half
hole count
time
```

分层报告。

不能只报告全数据平均值。

---

# 52. 外部 Baseline

## 52.1 Classical

```text
ART
SART
Tikhonov
TV-regularized Reconstruction
NNLS
```

## 52.2 Previous Work

固定 canonical geometry 下加入：

```text
LGV-Net
```

作为第二篇工作的直接对照。

## 52.3 Geometry-aware Deep Baseline

```text
2D Encoder
→ Geometry Lifting
→ Fixed / Mean Fusion
→ 3D Decoder
```

作为最接近的 geometry-aware baseline。

## 52.4 Neural Implicit Baseline

```text
NAF-style
或
NILAT-style
```

per-instance optimization。

比较：

```text
Accuracy
Inference Time
Optimization Time
Geometry Flexibility
```

---

# 53. 内部 Ablation

| Model | Geometry Lift | Fusion | Random V | Random Geometry | Held-out |
|---|---|---|---|---|---|
| B0 | ✓ | Mean | ✓ | ✓ | × |
| B1 | ✓ | Attention Pool | ✓ | ✓ | × |
| B2 | ✓ | Set Interaction | ✓ | ✓ | × |
| B3 | ✓ | Set Interaction | ✓ | ✓ | ✓ |
| Full | ✓ | Geometry-conditioned Set | ✓ | ✓ | ✓ |

额外消融：

```text
Fixed Geometry
vs
Random Geometry

Fixed V
vs
Random V

w/o Depth Feature

w/o Geometry

w/o L_hold

w/o TV

w/o Noise Augmentation
```

---

# 54. 评价指标

## 54.1 三维 Voxel Metrics

```text
RMSE
NRMSE
MAE
PSNR
3D SSIM
Slice-averaged SSIM
Pearson Correlation
```

## 54.2 Plume Quantitative Metrics

```text
Peak Mass Fraction Error
Integrated Concentration Error
Plume Centroid Error
Leak Source Localization Error
```

## 54.3 Morphology

```text
Dice
IoU
Plume Volume Error
Hausdorff Distance
```

阈值必须固定定义。

## 54.4 Measurement-space Metrics

```text
Input-view Projection Error
Held-out-view Projection Error
Independent OOD-view Projection Error
```

必须与 3D object-space metrics 分开报告。

---

# 55. MVP 配置

```text
GT:
    128³

Training Target:
    64³

Projection:
    64 × 64

Train Projection Pool:
    64 views / volume

OOD Projection Pool:
    32 views / volume

Input Views:
    3 ~ 6

Held-out:
    1

Reference K:
    256

Differentiable K:
    96

2D Encoder:
    ResNet-lite

Lifting:
    32³

Fusion:
    Mean → Attention

Output:
    64³

Activation:
    Sigmoid

Loss:
    L_obs
    + L_hold
    + 1e-4 L_tv

Noise:
    initially clean
    then 20 ~ 40 dB
```

---

# 56. Final Model 目标配置

```text
GT:
    128³

Projection:
    64 × 64

Train Projection Pool:
    128 views / volume

OOD Pool:
    64 views / volume

Input Views:
    3 ~ 8

Held-out:
    1 ~ 2

Lifting:
    32³

Intermediate:
    64³

Output:
    128³

Fusion:
    Geometry-conditioned Set Interaction

Training:
    3D-label-free main model
    + optional weak supervision
```

---

# 57. 推荐工程目录

```text
project/
│
├── data/
│   ├── fluent_gt/
│   ├── splits/
│   └── projection_pool/
│
├── geometry/
│   ├── geometry_sampler.py
│   ├── camera_model.py
│   ├── ray_generator.py
│   ├── ray_box_intersection.py
│   └── calibration_noise.py
│
├── projectors/
│   ├── mass_to_mole.py
│   ├── reference_projector.py
│   ├── differentiable_projector.py
│   └── projector_tests.py
│
├── datasets/
│   ├── gt_volume_dataset.py
│   ├── projection_pool_dataset.py
│   ├── view_set_sampler.py
│   └── measurement_augmentation.py
│
├── models/
│   ├── encoder2d.py
│   ├── geometry_lifting.py
│   ├── set_fusion.py
│   ├── decoder3d.py
│   └── geophy_set3d.py
│
├── losses/
│   ├── projection_loss.py
│   ├── tv3d.py
│   └── supervised_loss.py
│
├── scripts/
│   ├── build_projection_pool.py
│   ├── validate_projector.py
│   └── build_splits.py
│
├── train_supervised.py
├── train_measurement.py
├── evaluate.py
└── refine.py
```

---

# 58. Dataset 输出

```python
sample = {
    "input_projection": Y_in,
    "input_geometry": G_obs_in,
    "input_mask": input_mask,

    "hold_projection": Y_hold,
    "hold_geometry": G_obs_hold,
    "hold_mask": hold_mask,

    "gt_volume": C_gt,

    "true_geometry_in": G_true_in,
    "true_geometry_hold": G_true_hold,

    "case_id": case_id,
    "frame_id": frame_id,
    "meta": meta,
}
```

---

# 59. 主训练伪代码

```python
for batch in dataloader:

    Y_in = batch["input_projection"]
    G_in = batch["input_geometry"]

    Y_hold = batch["hold_projection"]
    G_hold = batch["hold_geometry"]

    C_hat = model(
        projections=Y_in,
        geometry=G_in,
        view_mask=batch["input_mask"],
    )

    Y_in_hat = diff_projector(
        volume=C_hat,
        geometry=G_in,
    )

    Y_hold_hat = diff_projector(
        volume=C_hat,
        geometry=G_hold,
    )

    L_obs = projection_loss(
        Y_in_hat,
        Y_in,
    )

    L_hold = projection_loss(
        Y_hold_hat,
        Y_hold,
    )

    L_tv = physical_tv(
        C_hat,
        voxel_spacing,
    )

    loss = (
        lambda_obs * L_obs
        + lambda_hold * L_hold
        + lambda_tv * L_tv
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

主训练循环中不调用 Reference Projector。

Reference Projector 只在离线 Projection Pool 构建阶段使用。

---

# 60. 开发顺序

```text
1. 冻结 case-level train / val / test split

2. 统一 128³ GT 的 xyz 坐标和物理单位

3. 编写新的 Camera / Geometry Sampler

4. 编写 Ray Generator 和 Ray-box Intersection

5. 实现 Mass Fraction → Mole Fraction 转换

6. 实现 Reference Projector

7. 完成 Phantom Projector Tests

8. 实现 Differentiable Projector

9. 比较 Reference / Differentiable Projector

10. 生成小规模 Projection Pool

11. 训练 Supervised GeoLift-Mean-Res3D

12. 验证 Random View Combination

13. 验证 Variable V

14. 加 Attention Pooling

15. 加 Set Interaction

16. 切换到 L_obs 训练

17. 加 L_hold

18. 做 Geometry-OOD

19. 做 Plume-OOD

20. 做 Double-OOD

21. 做 Shortcut Tests

22. 加 Noise / Calibration Error

23. 升级到 128³ 输出

24. 最后做 Weak Supervision 和 Test-time Refinement
```

---

# 61. 阶段性成功标准

## M1：Projection Physics 正确

- 质量分数到摩尔分数转换正确；
- `ppm·m` 单位正确；
- ray path length 正确；
- valid ray mask 正确；
- 坐标轴无交换；
- phantom 结果符合理论。

## M2：Supervised Direct-3D 成立

- 64³ 输出可稳定恢复 plume；
- geometry shuffle 后明显恶化；
- view 顺序 shuffle 不影响输出；
- 没有明显三维 artifact。

## M3：Variable-view 成立

同一个 checkpoint 对：

```text
V = 3 ~ 8
```

均可直接运行。

## M4：Geometry Generalization 成立

独立 Geometry Pool B 下：

- 不发生结构崩溃；
- random-geometry training 优于 fixed-geometry training；
- geometry-conditioned fusion 优于不使用 geometry 的模型。

## M5：3D-label-free Learning 成立

不用 volumetric GT loss 时：

- reconstruction 稳定；
- 三维 metrics 优于 classical baseline；
- held-out loss 对 object-space metrics 有实际改善；
- shortcut tests 表明模型确实依赖 measurement。

---

# 62. 主要风险

## Risk 1：独立 CFD case 数仍偏少

Projection 数量增加不会增加独立 plume morphology。

对策：

```text
trajectory-level split
use multiple train frames
report case-level metrics
avoid overly large network
```

## Risk 2：Measurement Consistency 不能保证唯一三维解

对策：

```text
Held-out Views
3D Population Prior
Huber-TV
Weak Supervision
Prior-constrained Refinement
```

## Risk 3：Projection Pool Geometry 太容易

对策：

```text
independent OOD pool
limited-angle cases
clustered views
variable angular span
```

## Risk 4：网络记住平均 plume prior

对策：

```text
Projection Shuffle
Geometry Shuffle
Zero Measurement
Double-OOD
```

## Risk 5：Projection Loss 主导导致过平滑

对策：

```text
monitor peak error
monitor plume morphology
reduce TV
compare supervised upper bound
```

## Risk 6：高分辨率三维显存过高

对策：

```text
32³ lifting
late 128³ upsampling
small batch
AMP
gradient accumulation
chunked projector
```

---

# 63. 最终论文建议表述

本工作的核心方法可以概括为：

> 提出一种面向稀疏气体羽流层析的 geometry-generalizable amortized 3D reconstruction framework。该方法仅以 Fluent 三维甲烷质量分数场作为基础数据，通过随机 acquisition simulator 离线生成大规模 Projection Pool，并将数量可变的二维路径积分测量表示为带显式几何信息的无序集合。模型通过 geometry-aware volumetric lifting 和 geometry-conditioned set interaction 直接预测三维甲烷质量分数场，再利用可微路径积分模型建立 reconstruction–measurement 闭环，实现无需逐体素三维标签作为主要监督的 measurement-domain learning。

---

# 64. 最终核心原则

> **GT comes from CFD.**  
> **Measurements come from our acquisition simulator.**  
> **Geometry is explicit.**  
> **View number is variable.**  
> **Physics closes the measurement loop.**  
> **The network reconstructs a true 3D volume.**

对应中文：

```text
CFD 只负责提供可信三维 Ground Truth

所有二维投影由本研究统一生成

测量几何显式输入

输入视角数量可变

预测三维场必须重新解释测量

最终输出是真正的三维质量分数体
```

这一数据定义使第三篇工作的 acquisition geometry、Projection Pool、训练协议、OOD 测试和可微物理模型完全由本研究统一控制，也避免受到原固定 8 视角投影配置的限制。
