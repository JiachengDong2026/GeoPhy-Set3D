# GeoPhy-Set3D 数据集生成详细说明
## 基于服务器静态末帧 Fluent GT 构建 TDLAS 扇形扫描 Projection Pool

**文档版本**：v1.3  
**对应技术方案**：GeoPhy-Set3D v0.5  
**原始数据来源**：既有 Fluent 甲烷瞬态模拟数据集  
**目标数据集**：面向随机稀疏视角三维重建的 Projection Pool 数据集  
**核心原则**：每条 Fluent 轨迹仅使用最后一帧 `frame 60 / t = 2.0 s` 的三维 Ground Truth；测量模型采用多条独立 TDLAS 激光束组成的远距离扇形扫描，扫描半径约 `6 m`；所有射线几何、Projection Pool、噪声和几何扰动均由本项目重新生成。

**服务器原始数据目录**：

```text
/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/
```

**服务器派生数据输出目录**：

```text
/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/
```

> 本版本面向服务器端 Codex 直接实现。数据入口固定为静态末帧子集，不再扫描原始 Fluent `runs/*` 轨迹目录。


> **Markdown / 公式统一规范**
>
> - 正文变量统一使用反引号，例如 `C_gt`、`Y_pic`、`G_true`、`G_obs`。
> - 简单尺寸直接写成 `128 × 128 × 128`、`64 × 64`。
> - 所有数学公式统一使用独立公式块。
> - 公式块上下留空行。
> - 不使用 `$...$` 行内公式。
> - 不在 Markdown 表格单元格内嵌套 LaTeX。
> - 张量尺寸、接口、配置和伪代码使用代码块。

---

# 1. 文档目的

本说明定义 GeoPhy-Set3D 项目的正式数据集生成流程。

原始数据由 Fluent 完成三维甲烷扩散模拟，并导出统一规则网格上的甲烷质量分数体场。本项目不使用原数据集中已经生成的固定 8 视角投影作为正式训练输入，而是仅使用三维 Ground Truth，重新建立统一的随机测量几何与路径积分观测模型。

最终数据链定义为：

```text
Server Static Final-frame Dataset
    ↓
manifest.jsonl
    ↓
targets/<sample_id>.npy
    ↓
128³ CH4 Mass Fraction Ground Truth
frame 60 / t = 2.0 s
    ↓
Case-level Train / Validation / Test Split
    ↓
Random Geometry Pool
    ↓
High-accuracy Reference Projector
    ↓
Offline Projection Pool
    ↓
Online Variable-view Sampling
    ↓
GeoPhy-Set3D Training / Evaluation
```

该生成协议主要解决以下问题：

1. 消除原固定 8 视角配置对新算法的限制；
2. 支持随机、非均匀、数量可变的稀疏视角；
3. 支持独立 Geometry-OOD 测试；
4. 保证三维 GT 与二维路径积分测量之间的物理量关系正确；
5. 避免每个 training iteration 在线重新生成 Ground Truth projection；
6. 为 measurement-domain training 提供可重复的 Projection Pool；
7. 为噪声、几何标定误差和未来真实测量扩展预留统一接口。

---

# 2. 原始 Fluent 数据在本项目中的角色

## 2.1 只使用三维 Ground Truth

本项目正式使用的数据源为：

```text
CH4 mass fraction volume
shape = [128, 128, 128]
dtype = float32
```

不使用原数据集中已经生成的以下内容作为正式训练测量：

```text
fixed 8-view projection
historical ray geometry
view00 ~ view07
historical PIC inputs
```

因此本项目的数据定义为：

> Fluent 数据集负责提供三维真实场；GeoPhy-Set3D 项目负责重新生成完整 tomography acquisition dataset。

---

# 2.2 三维 GT 的物理量

原始三维标签为：

> CH4 mass fraction

记为：

$$
C_{\mathrm{gt}}
$$

其物理范围为：

$$
0
\leq
C_{\mathrm{gt}}
\leq
1
$$

网络最终输出也定义为：

```text
CH4 mass fraction
```

不使用每样本归一化后的无量纲伪浓度作为最终物理输出。

---

# 2.3 原始空间范围与轴顺序

原始规则 GT 的重建区域为：

```text
x ∈ [0, 0.6] m
y ∈ [0, 0.6] m
z ∈ [0, 0.6] m
```

坐标含义：

```text
x: 顺风方向
y: 竖直方向
z: 横向方向
```

规则场尺寸：

```text
128 × 128 × 128
```

轴顺序：

```text
[x, y, z]
```

每一轴采用：

```text
linspace(0, 0.6, 128)
```

因此相邻采样点间距为：

$$
\Delta
=
\frac{0.6}{127}
$$

约为：

```text
4.724409 mm
```

需要特别注意：

> 该网格是包含两端点的规则采样点，不应未经说明直接当成 128 个等宽有限体积 cell center。

这会影响：

- Reference Projector；
- GT 下采样；
- 物理空间 TV；
- Siddon / Joseph 等 cell-based projector 的实现；
- volume integration 指标。

---

# 3. 服务器数据源与读取入口

## 3.1 固定 Source Root

服务器上的原始数据已经整理到：

```text
/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/
```

本项目将该目录记为：

```text
SOURCE_ROOT
```

Codex 和所有数据生成脚本不得再自行搜索其他 Fluent 数据目录。

---

# 3.2 服务器源目录结构

预期结构：

```text
static_final_frame_partial_v1/
├── README.md
├── dataset_summary.json
├── manifest.jsonl
├── inputs/
│   └── <sample_id>.npy
├── targets/
│   └── <sample_id>.npy
└── metadata/
    └── <sample_id>.json
```

本项目真正使用：

```text
manifest.jsonl
targets/
metadata/
```

本项目明确忽略：

```text
inputs/
```

原因是 `inputs/` 中保存的是原数据集已经生成的固定视角投影，本工作将全部二维 TDLAS measurement 重新生成。

---

# 3.3 Manifest 是唯一正式索引入口

正式数据生成程序必须优先读取：

```text
/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/manifest.jsonl
```

而不是仅使用：

```text
glob("targets/*.npy")
```

原因：

- `manifest.jsonl` 提供 `sample_id`；
- 提供 target 相对路径；
- 提供场景和工况 metadata；
- 可以统一建立 case-level split；
- 避免只根据文件名猜测物理工况。

推荐：

```python
from pathlib import Path
import json

SOURCE_ROOT = Path(
    "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1"
)

records = [
    json.loads(line)
    for line in (SOURCE_ROOT / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    if line.strip()
]
```

---

# 3.4 GT 文件读取

每条 manifest record 的目标文件由：

```text
record["target_path"]
```

给出。

推荐：

```python
import numpy as np

gt_path = SOURCE_ROOT / record["target_path"]

C_gt = np.load(
    gt_path,
    mmap_mode="r",
    allow_pickle=False,
)
```

必须检查：

```text
shape == [128, 128, 128]
dtype == float32
finite == true
minimum >= 0
maximum <= 1 + tolerance
```

---

# 3.5 只使用最后一帧

该 source dataset 已经是静态末帧子集。

每条样本固定为：

```text
frame_index = 60
physical_time_s = 2.0
```

因此 Codex 不需要：

```text
扫描 frame 0 ~ 60
检查 61 帧是否全部存在
从轨迹中选择最后一帧
构建 Temporal Pool
```

每条 manifest record 直接对应一个基础三维样本。

---

# 3.6 每条记录只生成一个 Projection Pool

正式映射关系：

```text
1 manifest record
    ↓
1 final-frame 128³ GT
    ↓
1 Train Projection Pool
+
1 OOD Projection Pool
```

即：

```text
1 CFD case
→
1 static 3D GT sample
→
many candidate TDLAS station measurements
```

需要再次强调：

> 增加随机 TDLAS station 数量只增加 measurement geometry diversity，不增加独立 plume morphology 数量。

---

# 4. 原始 GT 的物理定义

每条 target 为：

```text
CH4 mass fraction
shape = [128, 128, 128]
dtype = float32
axes = [x, y, z]
```

其物理范围：

$$
0
\leq
C_{\mathrm{gt}}
\leq
1
$$

网络最终输出也定义为：

```text
CH4 mass fraction
```

---

# 4.1 空间坐标

重建区域：

```text
x ∈ [0, 0.6] m
y ∈ [0, 0.6] m
z ∈ [0, 0.6] m
```

坐标：

```text
x: 顺风方向
y: 竖直方向
z: 横向方向
```

每轴：

```text
linspace(0, 0.6, 128)
```

因此：

$$
\Delta
=
\frac{0.6}{127}
$$

约：

```text
4.724409 mm
```

该网格是包含端点的规则采样点。

---

# 5. Source Dataset 只读约束

服务器上的：

```text
/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/
```

必须视为：

```text
READ-ONLY
```

禁止：

```text
修改源 NPY
原地归一化
原地 clip
覆盖 target
删除或移动源文件
向 source root 写入生成结果
```

允许：

```text
np.load(..., mmap_mode="r", allow_pickle=False)
只读 JSON / JSONL
读取 metadata
```

所有派生数据统一写入：

```text
/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/
```

---

# 6. Case Manifest 与元数据使用

由于 `manifest.jsonl` 已经存在，本项目不再重新构建另一个原始 Fluent case manifest 作为主索引。

可以在输出目录生成一个派生索引：

```text
source_index.jsonl
```

它只负责记录：

```text
sample_id
source target path
source metadata path
split
generated projection paths
generation status
```

如果 manifest 中已经提供以下字段，应直接继承：

```text
geometry_id
pressure_pa
wind_speed_m_s
active_hole_count
source family
condition_id
role
```

不要从文件名重新解析已经存在的字段。

如果某字段在原 manifest 中不存在，则：

```text
优先读取 metadata/<sample_id>.json
```

仍然不存在时：

```text
写 null
```

不得自行猜测。


# 7. Train / Validation / Test 划分

## 7.1 正确顺序

必须：

```text
manifest.jsonl records
    ↓
Case-level Split
    ↓
Load target GT
    ↓
Projection Generation
```

禁止：

```text
Generate all derived projections before case-level split
    ↓
Random sample split
```

---

# 7.2 同一 Case 不得跨 Split

同一 `case_id` 的以下内容必须位于同一 split：

```text
final-frame GT
all generated projections
all geometry combinations
all noise variants
all downsampled GT
all derived samples
```

---

# 7.3 推荐比例

完整数据冻结后可先采用：

```text
Train = 70%
Validation = 15%
Test = 15%
```

但应尽量按以下属性分层：

```text
geometry family
hole count
wind speed
pressure
equal / half
```

---

# 7.4 原 CFD 分布偏差

重新生成 projection 不能消除原 Fluent case 的分布不均衡。

因此 split 和评估必须保留：

```text
geometry_id
wind speed
pressure
hole count
source mode
time
```

最终结果应按这些因素分层报告。

---

# 7.5 Split 输出

```text
splits/
├── train_cases.txt
├── val_cases.txt
├── test_cases.txt
└── split_summary.json
```

`split_summary.json` 应统计各条件分布。

---

# 8. GT 预处理

## 8.1 不做逐样本最大值归一化

禁止：

```text
C_gt = C_gt / C_gt.max()
```

因为会破坏绝对质量分数信息。

GT 始终保持：

```text
physical CH4 mass fraction
```

---

# 8.2 数值范围检查

允许浮点误差级别的小范围越界修正，例如：

```text
clip to [0,1]
```

但必须记录：

```text
number of clipped voxels
max underflow
max overflow
```

如果存在明显物理越界，应标记异常 case，而不是静默修正。

---

# 8.3 64³ MVP Target

MVP 网络可输出：

```text
64 × 64 × 64
```

因此生成：

```text
C_gt_64
```

推荐使用：

```text
trilinear interpolation
或
physical-space average pooling
```

更推荐体积意义清晰的下采样方法。

必须注意：

> Projection Pool 始终从原始 `128³` GT 生成，不从 `64³` target 生成。

这样 MVP 自然包含适度的 representation mismatch。

---

# 9. 新 Acquisition Geometry：多束 TDLAS 扇形扫描

## 9.1 测量系统定义

本项目不再采用“相机 + 面阵 detector + 宽视场角”的成像模型。

新的测量模型定义为：

> **多个独立 TDLAS line-of-sight 激光测量束组成的扇形扫描。**

每一条 beam 都产生一个独立的路径积分浓度测量。

单条 TDLAS beam 的测量本质为：

```text
scanner / laser origin
        ↓
one independent line-of-sight beam
        ↓
intersection with reconstruction domain
        ↓
CH4 mole-fraction path integration
        ↓
one scalar PIC measurement
```

为了保持与网络二维编码器兼容，一个扫描 station 表示为一个规则的二维 angular scan grid：

```text
64 horizontal steering angles
×
64 vertical steering angles
=
4096 independent TDLAS rays
```

这里的 `64 × 64` 不再表示相机图像像素，而表示：

> **4096 条具有已知方向的独立 TDLAS 激光路径积分测量。**

---

# 9.2 世界坐标

重建域：

```text
x ∈ [0, 0.6] m
y ∈ [0, 0.6] m
z ∈ [0, 0.6] m
```

中心：

$$
c
=
\left(
0.3,
0.3,
0.3
\right)
$$

坐标定义：

```text
x: 顺风方向
y: 竖直方向
z: 横向方向
```

主扫描器位置围绕重建区域布置在近似水平 `x-z` 半环上。

---

# 9.3 扫描半径

原方案 `0.75 m` 的半径取消。

新的基准扫描距离设置为：

```text
R0 = 6.0 m
```

训练几何随机范围：

```text
R ∈ [5.5, 6.5] m
```

即约为：

```text
6.0 m ± 0.5 m
```

扫描器位置：

$$
p
\left(
\theta
\right)
=
c
+
\left[
R \cos\theta,
\Delta y,
R \sin\theta
\right]
$$

其中：

```text
theta:
    scanner station azimuth

R:
    scanner-to-domain-center distance

Δy:
    scanner height offset relative to domain center
```

---

# 9.4 Scanner Station Azimuth

扫描器站位方位角：

```text
theta ∈ [0°, 180°]
```

这里的 `theta` 表示整个 TDLAS 扇形扫描器相对于重建域中心的位置方位角。

每一个 station 内部还会产生一组 fan-beam directions。

---

# 9.5 Scanner Height

基准高度：

```text
y0 = 0.30 m
```

MVP 随机范围：

```text
scanner y ∈ [0.25, 0.35] m
```

即：

```text
0.30 m ± 0.05 m
```

扫描器中心 beam 指向：

```text
(0.3, 0.3, 0.3) m
```

---

# 10. TDLAS 扇形 Beam Geometry

## 10.1 中心 Beam

对于 scanner origin：

$$
o
$$

中心 beam 的方向：

$$
d_0
=
\frac{
c-o
}{
\left\|
c-o
\right\|
}
$$

中心 beam 始终瞄准重建域中心。

---

# 10.2 水平扇形角宽

新的水平 fan angular width：

```text
full horizontal fan angle = 10°
half angle = ±5°
```

在约 `6 m` 距离上，中心平面覆盖宽度近似为：

$$
W_h
=
2R
\tan
\left(
5^\circ
\right)
$$

当：

```text
R = 6 m
```

时约为：

```text
1.05 m
```

大于重建域水平尺度 `0.6 m`，并为立方体斜向投影和站位扰动保留余量。

---

# 10.3 垂直扫描角宽

为了形成真正的三维观测，每个 station 同时进行垂直 steering。

新的垂直 angular span：

```text
full vertical scan angle = 8°
half angle = ±4°
```

在 `6 m` 距离上覆盖高度近似为：

$$
W_v
=
2R
\tan
\left(
4^\circ
\right)
$$

约为：

```text
0.84 m
```

能够覆盖 `0.6 m` 高的重建区域，并保留 margin。

---

# 10.4 64 × 64 的重新解释

每一个 station 的 measurement array：

```text
[64, 64]
```

定义为：

```text
64 horizontal beam steering angles
×
64 vertical beam steering angles
```

也就是：

```text
4096 independent TDLAS rays / station
```

推荐 beam angular coordinates：

```text
horizontal:
    linspace(-5°, +5°, 64)

vertical:
    linspace(-4°, +4°, 64)
```

因此二维数组只是为了方便存储和进入 2D encoder，其物理含义是：

> **二维角度索引下的 TDLAS 路径积分测量阵列，而不是光学相机图像。**

---

# 10.5 Beam Direction 构造

首先为 scanner station 建立局部坐标基。

中心方向：

```text
forward = normalize(center - origin)
```

世界竖直参考：

```text
up_ref = [0, 1, 0]
```

局部水平方向：

```text
right = normalize(cross(forward, up_ref))
```

重新正交化：

```text
up = normalize(cross(right, forward))
```

对每一组水平角 `alpha` 和垂直角 `beta`，生成单位方向：

```text
d(alpha, beta)
```

并要求：

```text
norm(d) = 1
```

---

# 10.6 与真实 TDLAS 的对应关系

数据集中每一个 array element：

```text
Y[i, j]
```

对应一条独立 TDLAS beam：

```text
horizontal steering index = j
vertical steering index = i
```

每条 beam 只有一个标量观测：

```text
PIC [ppm·m]
```

建议术语：

```text
TDLAS fan-beam measurement map
angular scan map
path-integrated measurement map
```

不再称为普通 camera image。

---

# 11. Ray-box Intersection

每条 TDLAS beam 定义为：

$$
r
\left(
t
\right)
=
o
+
td
$$

只积分 beam 与重建域：

```text
[0,0.6] × [0,0.6] × [0,0.6]
```

的相交部分。

通过 slab intersection 得到：

```text
t_near
t_far
path_length
valid_ray
```

有效条件：

```text
t_far > t_near
```

如果 `d` 已单位化：

$$
L
=
t_{\mathrm{far}}
-
t_{\mathrm{near}}
$$

注意：

> scanner 距离约 `6 m`，但真正进入 methane reconstruction domain 的积分路径长度通常只有 `0 ~ 约 1.04 m`，不会把 scanner 到重建域之间的空白空气路径当作甲烷场积分区域。

---

# 12. Random Scanner Geometry

## 12.1 MVP Station 参数

```text
station azimuth:
    Uniform(0°, 180°)

station radius:
    Uniform(5.5, 6.5) m

station height:
    Uniform(0.25, 0.35) m

center target:
    (0.3, 0.3, 0.3) m

horizontal fan:
    -5° ~ +5°

vertical steering:
    -4° ~ +4°

beam grid:
    64 × 64
```

---

# 12.2 Geometry Pool A

训练候选 stations：

```text
MVP:
    64 stations

Final:
    128 stations
```

每一个 station 含：

```text
4096 TDLAS beams
```

---

# 12.3 Geometry Pool B

独立 OOD scanner stations：

```text
MVP:
    32 stations

Final:
    64 stations
```

使用独立随机 seed，不参与训练。

---

# 12.4 Hard Geometry Set

训练时从 Pool A 组合多个 stations。

建议：

```text
50%:
    broad station-angle coverage

30%:
    medium coverage

20%:
    clustered / limited-angle stations
```

例如 4 个输入 stations：

```text
easy:
    10°, 55°, 110°, 165°

hard:
    40°, 48°, 56°, 64°
```

这里的角度表示 scanner station azimuth，不是 station 内部 fan beam 的 `±5°` steering。

---

# 13. Geometry 文件内容

建议每个 station 保存：

```text
station_origin_m       [M,3]
station_azimuth_deg    [M]
station_radius_m       [M]
station_height_m       [M]

center_direction       [M,3]

horizontal_angle_deg   [64]
vertical_angle_deg     [64]

ray_direction          [M,64,64,3]
t_near                 [M,64,64]
t_far                  [M,64,64]
path_length_m          [M,64,64]
valid_ray              [M,64,64]
```

不再使用：

```text
camera FOV
detector plane
pinhole projection
camera pixel
```

作为主要物理术语。

---

# 14. Geometry Profile

新的 geometry profile：

```text
tdlas_fan_half_ring_r6m_v1
```

核心定义：

```text
scanner trajectory:
    horizontal half ring

nominal radius:
    6.0 m

radius range:
    5.5 ~ 6.5 m

station azimuth:
    0° ~ 180°

height:
    0.25 ~ 0.35 m

horizontal fan:
    10° full width

vertical scan:
    8° full width

beam grid:
    64 × 64

target:
    reconstruction center
```

以下内容变化时必须升级 profile：

```text
radius range
station angle range
fan angle
vertical scan angle
beam count
beam indexing
station height
target point
```


# 15. 甲烷质量分数到摩尔分数

路径积分 measurement 定义为 CH4 mole fraction 的积分。

采用：

```text
M_CH4 = 16.043
M_air = 28.965
```

质量分数到摩尔分数：

$$
\chi
\left(
C
\right)
=
\frac{
C / M_{\mathrm{CH4}}
}{
C / M_{\mathrm{CH4}}
+
\left(
1-C
\right)
/ M_{\mathrm{air}}
}
$$

其中：

- `C`：CH4 mass fraction；
- `chi`：CH4 mole fraction。

---

# 16. 路径积分物理定义

对有效 ray：

$$
PIC
=
10^6
\int_{\mathrm{ray}}
\chi
\left(
C
\left(
x
\right)
\right)
dl
$$

单位：

```text
ppm·m
```

离散形式：

$$
PIC
\approx
10^6
\sum_{k=1}^{K}
\chi_k
w_k
$$

其中 `w_k` 单位为 m。

---

# 17. 推荐的物理处理顺序

正式实现建议固定为：

```text
CH4 Mass Fraction Field
        ↓
Spatial Interpolation at Ray Samples
        ↓
Mass Fraction Sample Values
        ↓
Mass Fraction → Mole Fraction
        ↓
Path Integration
        ↓
× 10^6
        ↓
ppm·m
```

如果实现为了效率选择：

```text
whole-volume mass-to-mole conversion
→ interpolation
```

必须通过数值测试确认与标准实现的差异可接受，并固定唯一正式版本。

---

# 18. Reference Projector

## 18.1 角色

Reference Projector 只用于：

```text
Offline Projection Pool Generation
```

不要求可微。

优先级：

```text
accuracy
unit correctness
reproducibility
```

高于单次速度。

---

# 18.2 MVP 实现

```text
Input:
    128³ CH4 mass fraction

Ray samples:
    K_ref = 256

Interpolation:
    trilinear or validated higher-order interpolation

Integration:
    trapezoidal

Output:
    [64,64] ppm·m
```

---

# 18.3 正式版本

建议比较：

```text
K_ref = 64
128
256
512
```

并评估数值收敛。

如果：

```text
256 vs 512
```

误差远小于模型预期重建误差，可正式采用：

```text
K_ref = 256
```

---

# 18.4 与 Differentiable Projector 分离

训练时另使用：

```text
Differentiable Projector
```

建议：

```text
Reference:
    K_ref = 256
    high-accuracy integration
    trapezoidal

Training:
    K_train = 96 ~ 128
    trilinear grid_sample
    midpoint / rectangle integration
```

即：

```text
P_ref != P_diff
```

这样可以减轻 inverse crime。

---

# 19. Reference Projector 单元测试

## 19.1 Zero Field

```text
C = 0
```

必须得到：

```text
PIC = 0
```

---

# 19.2 Constant Field

若：

```text
C = constant
```

则理论值：

$$
PIC
=
10^6
\chi
\left(
C
\right)
L
$$

这是检查单位、路径长度和 mass-to-mole 转换的关键测试。

---

# 19.3 Compact Cube / Sphere

用于检查：

- 空间位置；
- scanner orientation；
- 图像上下左右；
- view rotation；
- ray-box intersection。

---

# 19.4 3D Gaussian

用于检查：

- 多角度投影平滑性；
- `K_ref` 收敛；
- Reference 与 Differentiable Projector 的数值偏差。

---

# 20. Mass-to-Mole 单元测试

至少测试：

```text
C = 0
C = 0.001
C = 0.01
C = 0.1
C = 0.5
C = 1
```

要求：

```text
C = 0  → chi = 0
C = 1  → chi = 1
```

并满足单调性：

```text
C1 < C2
→
chi(C1) < chi(C2)
```

---

# 21. Projection Pool

对于一个三维 GT：

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

其中每一个 `Y_m` 是一个独立单视角 projection。

---

# 21.1 MVP Pool

每个 volume：

```text
Train candidate views:
    64

OOD candidate views:
    32
```

---

# 21.2 正式 Pool

```text
Train candidate views:
    128

OOD candidate views:
    64
```

---

# 21.3 不预生成组合

不要存：

```text
3-view sample
4-view sample
5-view sample
...
```

只存 individual view。

训练时在线选择。

这样 128 个候选视角可以形成大量不同组合。

---

# 22. Projection Pool 数据类型

物理 `ppm·m` 数据默认保存：

```text
float32
```

不建议无缩放保存为：

```text
float16
```

原因：

```text
float16 maximum finite value ≈ 65504
```

实际 PIC 数值可能超过该范围。

如果未来需要 FP16 压缩，必须：

```text
fixed global scaling
→ float16 storage
→ save inverse scaling metadata
```

---

# 23. Projection Pool 目录结构

建议：

```text
geophy_projection_pool_v1/
├── README.md
├── dataset_build_manifest.json
├── generation_config.yaml
├── source_index.jsonl
├── train_statistics.json
│
├── splits/
│   ├── train_cases.txt
│   ├── val_cases.txt
│   ├── test_cases.txt
│   └── split_summary.json
│
├── geometry/
│   ├── geometry_profile.json
│   ├── train_geometry.npz
│   └── ood_geometry.npz
│
├── samples/
│   └── <case_id>/
│       ├── frame_manifest.jsonl
│       ├── projections_train/
│       └── projections_ood/
│
├── audits/
│   ├── geometry_audit.json
│   ├── projector_validation.json
│   ├── projection_pool_audit.json
│   └── split_leakage_audit.json
│
└── logs/
    └── generation_*.log
```

---

# 24. 单 Case 的组织方式

由于本工作只使用最后一帧，每个 CFD case 只需要保存一组 Projection Pool：

```text
samples/<case_id>/
├── sample_meta.json
├── projections_train.npy
└── projections_ood.npy
```

其中：

```text
frame_index = 60
physical_time_s = 2.0 s

projections_train:
    [M_train, 64, 64]

projections_ood:
    [M_ood, 64, 64]

dtype:
    float32
```

三维 GT 默认通过 `sample_meta.json` 引用原始：

```text
mass_fraction_ch4_000060.npy
```

不需要为同一 case 建立多帧目录。

原 GT 可以通过 manifest 引用原始只读文件，不必全部复制。

---

# 25. 是否复制 GT

单个 `128³ float32` GT 大约：

```text
8 MiB
```

而单帧：

```text
64 train views × 64 × 64 × float32
```

约为：

```text
1 MiB
```

因此 GT 复制成本高于 projection。

推荐：

> Projection Pool 数据集默认只记录原 GT 的只读路径，不重复复制全部 GT。

如果后续需要完全自包含的论文冻结数据集，再单独执行 export。

---

# 26. Reference Projector 计算复杂度

复杂度近似：

$$
O
\left(
N_{\mathrm{volume}}
M
H
W
K_{\mathrm{ref}}
\right)
$$

单 volume，若：

```text
M = 64
H = 64
W = 64
K_ref = 256
```

采样点数量：

```text
64 × 64 × 64 × 256
= 67,108,864
```

约 6710 万 sampling points。

如果：

```text
M = 128
```

则约：

```text
134 million sampling points / volume
```

因此这是离线任务，需要：

```text
view chunking
GPU batching
resume support
```

---

# 27. View Chunking

推荐：

```text
view_chunk = 4
或
view_chunk = 8
```

例如一次处理：

```text
8 × 64 × 64 × 256
```

ray samples。

避免一次性构造所有 views 的巨大 sampling grid。

---

# 28. 分阶段生成策略

## Phase 1：Debug Pool

```text
cases:
    5

frames:
    [60]

train views:
    16

ood views:
    8

K_ref:
    128
```

用于检查整个 pipeline。

---

## Phase 2：MVP Pool

```text
all selected cases
frame:
    60

train views:
    64

ood views:
    32

K_ref:
    256
```

用于早期网络实验。

---

## Phase 3：Final Pool

确认 geometry diversity 有收益后扩展：

```text
M_train = 128
M_ood = 64
```

---

# 29. 不建议一开始生成最终全量数据

必须先冻结：

```text
geometry_profile
TDLAS scanner / beam convention
mass-to-mole formula
Reference Projector implementation
K_ref
integration rule
```

否则一旦修改 projector，整个 Projection Pool 都需要重新生成。

---

# 30. Noise 不离线写入

Projection Pool 保存：

```text
clean noiseless measurement
```

训练时在线增加：

```text
Gaussian noise
multiplicative gain
additive offset
ray dropout
scan-line dropout
```

这样 clean baseline 永远可用。

---

# 31. Gaussian Noise 定义

可用：

$$
\mathrm{SNR}
=
20
\log_{10}
\left(
\frac{
\mathrm{RMS}
\left(
Y
\right)
}{
\sigma
}
\right)
$$

因此：

$$
\sigma
=
\frac{
\mathrm{RMS}
\left(
Y
\right)
}{
10^{\mathrm{SNR}/20}
}
$$

推荐：

```text
MVP:
    clean
    20 ~ 40 dB

Robustness:
    10 / 20 / 30 / 40 dB
```

---

# 32. Geometry Calibration Error

Projection Pool 保存：

```text
Y
G_true
```

训练时再构造：

```text
G_obs
```

满足：

$$
G_{\mathrm{obs}}
=
G_{\mathrm{true}}
+
\Delta G
$$

真实 measurement 保持：

$$
Y
=
\mathcal{P}_{\mathrm{ref}}
\left(
C_{\mathrm{gt}};
G_{\mathrm{true}}
\right)
$$

不能根据 `G_obs` 重新生成新的 `Y`。

这样才能真实模拟：

```text
position error
angle error
height error
pointing error
```

---

# 33. Online View Sampling

Dataset 每次读取一个 frame：

1. 从 train Projection Pool 中选 views；
2. 随机打乱 view order；
3. 分成 input 和 held-out；
4. 在线增加噪声；
5. 可选生成 `G_obs`；
6. 返回网络输入。

例如：

```text
selected:
    7

input:
    5

held-out:
    1

unused:
    1
```

---

# 34. View Count

MVP：

```text
V_input ~ UniformInteger(3,6)
V_hold = 1
```

正式训练：

```text
V_input ~ UniformInteger(3,8)
V_hold ~ UniformInteger(1,2)
```

测试增加：

```text
V = 1
V = 2
```

---

# 35. Angular Coverage Sampling

训练不能只随机 view count。

建议 measurement set 分为：

```text
50%:
    broad-span irregular views

30%:
    medium-span random views

20%:
    clustered / limited-angle cases
```

每个 set 计算：

```text
angular_span_deg
minimum_gap_deg
mean_pairwise_angle_deg
```

用于后续区分：

```text
view count effect
vs
angular coverage effect
```

---

# 36. Training Sample 输出

```python
sample = {
    "input_projection": Y_in,
    "input_geometry": G_obs_in,
    "input_valid_ray": valid_in,
    "input_view_mask": view_mask,

    "hold_projection": Y_hold,
    "hold_geometry": G_obs_hold,
    "hold_valid_ray": valid_hold,

    "gt_volume_128": C_gt_128,
    "gt_volume_64": C_gt_64,

    "true_geometry_in": G_true_in,
    "true_geometry_hold": G_true_hold,

    "case_id": case_id,
    "frame_index": frame_index,
    "physical_time_s": time_s,

    "condition_meta": condition_meta
}
```

---

# 37. Projection 输入归一化

原始 Projection Pool 始终保存：

```text
ppm·m
```

训练时可做固定尺度变换。

禁止：

```text
每张 projection 单独 / max(Y)
```

推荐：

$$
Y_{\mathrm{scaled}}
=
\frac{Y}{Y_{\mathrm{ref}}}
$$

或者：

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

`Y_ref` 只能由 train split 统计。

---

# 38. Train Statistics

冻结数据后生成：

```text
train_statistics.json
```

至少包含：

```text
projection mean
projection std
projection p50
projection p90
projection p95
projection p99
projection p99.5
projection max

GT mean
GT p95
GT p99
GT max
```

模型所有 normalization 参数均从 train statistics 获取。

---

# 39. Valid Ray Mask

无效 ray 可以在 projection 中保存为：

```text
0
```

但 loss 必须使用：

```text
valid_ray
```

区分：

```text
invalid ray zero
```

和：

```text
valid ray with zero methane
```

因此 valid mask 是正式 geometry 的必要组成部分。

---

# 40. Geometry QA

自动检查：

```text
ray direction norm ≈ 1
t_far >= t_near
path_length >= 0
valid ray fraction reasonable
scanner center-beam axis points toward reconstruction FOV
no NaN in valid ray geometry
```

同时保存 debug 图：

```text
scanner-station top view
scanner-station side view
ray bundle preview
valid-ray mask
```

---

# 41. Projection QA

每个生成 batch 检查：

```text
projection is finite
projection >= 0
invalid ray projection == 0
no NaN
no Inf
```

记录：

```text
min
max
mean
nonzero ratio
```

---

# 42. Projector 收敛实验

选择若干 phantom 和 Fluent GT。

比较：

```text
K_ref = 64
128
256
512
```

统计：

```text
relative L1
relative L2
max absolute error
```

若：

```text
256 vs 512
```

差异已经足够小，则采用 `K_ref = 256`。

---

# 43. Reference / Differentiable Projector 对照

选择真实 GT：

```text
Y_ref = P_ref(C_gt)
Y_diff = P_diff(C_gt)
```

统计：

```text
relative L1
relative L2
per-view error
```

目标不是完全一致，而是：

> 数值 mismatch 明显小于主要 reconstruction error，且不存在坐标、方向或单位的系统偏差。

---

# 44. 数据泄漏审计

必须自动检查：

```text
same case_id in train and test
same trajectory in train and test
same rerun condition across split
retained method-validation case reused as untouched final test
```

Geometry Pool A 可以被多个 train cases 共用。

但：

```text
Geometry Pool B
```

不能进入训练。

---

# 45. 随机种子

至少固定：

```text
split_seed
train_geometry_seed
ood_geometry_seed
```

训练阶段：

```text
view subset seed
noise seed
```

可随 epoch 变化。

---

# 46. 数据集版本管理

建议名称：

```text
geophy_projection_pool_v1
```

以下任何改变必须升级版本：

```text
source GT version
case admission rule
case split
geometry profile
TDLAS beam geometry model
FOV
projection resolution
mass-to-mole formula
Reference Projector
K_ref
integration rule
offline preprocessing
```

---

# 47. dataset_build_manifest

建议生成：

```text
dataset_build_manifest.json
```

例如：

```json
{
  "dataset_name": "geophy_projection_pool_v1",
  "source_dataset": "static_final_frame_partial_v1",
  "source_gt_quantity": "CH4 mass fraction",
  "source_gt_shape": [128, 128, 128],
  "domain_bounds_m": [
    [0.0, 0.6],
    [0.0, 0.6],
    [0.0, 0.6]
  ],
  "projection_quantity": "path-integrated CH4 mole fraction",
  "projection_unit": "ppm_m",
  "projection_shape": [64, 64],
  "geometry_profile": "tdlas_fan_half_ring_r6m_v1",
  "reference_projector_version": "ref_projector_v1",
  "split_version": "case_split_v1",
  "train_geometry_count": 64,
  "ood_geometry_count": 32
}
```

---

# 48. 推荐配置文件

```text
configs/dataset_generation_v1.yaml
```

示例：

```yaml
dataset:
  source_root: "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1"
  output_root: "/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1"
  manifest_path: "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/manifest.jsonl"
  target_root: "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/targets"
  metadata_root: "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/metadata"
  ignore_existing_inputs: true
  source_read_only: true

gt:
  shape: [128, 128, 128]
  bounds_m:
    x: [0.0, 0.6]
    y: [0.0, 0.6]
    z: [0.0, 0.6]

frames:
  mode: "final_only"
  frame_index: 60
  physical_time_s: 2.0

geometry:
  profile: "tdlas_fan_half_ring_r6m_v1"
  train_count: 64
  ood_count: 32
  azimuth_deg: [0.0, 180.0]
  radius_m: [5.5, 6.5]
  height_m: [0.25, 0.35]
  horizontal_fan_deg: 10.0
  vertical_scan_deg: 8.0
  projection_hw: [64, 64]

projector:
  methane_molar_mass: 16.043
  air_molar_mass: 28.965
  reference_samples: 256
  integration: "trapezoidal"
  output_unit: "ppm_m"
  dtype: "float32"

split:
  train: 0.70
  val: 0.15
  test: 0.15
  seed: 20261010
```

所有生效配置还必须复制到最终 dataset manifest。

---

# 49. 推荐脚本结构

```text
scripts/
├── validate_static_source.py
├── build_source_index.py
├── build_case_splits.py
├── build_geometry_pool.py
├── validate_geometry_pool.py
├── validate_reference_projector.py
├── build_projection_pool.py
├── audit_projection_pool.py
├── compute_train_statistics.py
└── freeze_dataset_version.py
```

---

# 50. 数据生成完整顺序

## Step 1：读取服务器静态末帧 manifest

读取：

```text
SOURCE_ROOT/manifest.jsonl
```

验证每条 record 对应的：

```text
target_path
metadata_path
```

并生成：

```text
source_index.jsonl
```

## Step 2：质量准入

检查：

```text
audit
GT shape
dtype
finite
range
final-frame completeness
```

## Step 3：冻结 Case Split

输出：

```text
train_cases.txt
val_cases.txt
test_cases.txt
```

## Step 4：生成 Geometry Pool

固定随机 seed 生成 Pool A / Pool B。

## Step 5：验证 Geometry

完成 geometry QA 和可视化。

## Step 6：验证 Reference Projector

完成 analytical phantom tests。

## Step 7：生成 Debug Pool

仅少量 case。

## Step 8：人工检查

检查：

```text
orientation
projection position
numerical range
valid mask
```

## Step 9：生成 MVP Pool

```text
frame = 60
M_train = 64
M_ood = 32
```

## Step 10：Audit

检查完整性和异常值。

## Step 11：计算 Train Statistics

只统计 train split。

## Step 12：冻结数据版本

保存：

```text
manifest
config
hash
logs
audits
```

---

# 51. Projection Pool 构建伪代码

```python
FINAL_FRAME_INDEX = 60
FINAL_TIME_S = 2.0

SOURCE_ROOT = Path(
    "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1"
)

OUTPUT_ROOT = Path(
    "/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1"
)

records = load_jsonl(
    SOURCE_ROOT / "manifest.jsonl"
)

for record in records:

    sample_id = record["sample_id"]

    gt_path = SOURCE_ROOT / record["target_path"]

    C_gt = np.load(
        gt_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    validate_gt(C_gt)

    Y_train = reference_projector(
        volume=C_gt,
        geometry=train_geometry_pool,
        output_unit="ppm_m",
    )

    Y_ood = reference_projector(
        volume=C_gt,
        geometry=ood_geometry_pool,
        output_unit="ppm_m",
    )

    save_projection_pool(
        sample_id=sample_id,
        frame_index=60,
        physical_time_s=2.0,
        train_projection=Y_train,
        ood_projection=Y_ood,
        gt_path=str(gt_path),
        output_root=OUTPUT_ROOT,
    )
```

---

# 52. Reference Projector 伪代码

```python
def project(volume, rays):

    samples = sample_mass_fraction_along_rays(
        volume=volume,
        rays=rays,
        num_samples=K_ref,
    )

    samples = samples.clamp(0.0, 1.0)

    mole_fraction = mass_fraction_to_mole_fraction(
        samples,
        methane_molar_mass=16.043,
        air_molar_mass=28.965,
    )

    pic = integrate_along_ray(
        mole_fraction,
        rays.path_length,
        method="trapezoidal",
    )

    pic_ppm_m = pic * 1e6

    pic_ppm_m[~rays.valid] = 0.0

    return pic_ppm_m.float()
```

---

# 53. Sample Metadata

每个 case 保存：

```text
frame_manifest.jsonl
```

每行：

```json
{
  "frame_index": 60,
  "physical_time_s": 2.0,
  "gt_path": "/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/targets/<sample_id>.npy",
  "train_projection_path": "projections_train/projection_000060.npy",
  "ood_projection_path": "projections_ood/projection_000060.npy",
  "gt_shape": [128, 128, 128],
  "projection_shape": [64, 64],
  "projection_unit": "ppm_m"
}
```

---

# 54. Projection Pool Audit

最终输出：

```text
projection_pool_audit.json
```

至少包含：

```text
total accepted cases
total final-frame samples
train / val / test case count
generated projection arrays
failed final-frame samples
NaN count
Inf count
negative measurement count
GT range violations
projection min
projection max
valid-ray statistics
missing file count
hash summary
```

---

# 55. 数据生成失败处理

如果某 frame 投影失败：

```text
不要保存半成品 sample
```

记录：

```text
case_id
frame_index
exception
geometry chunk
timestamp
```

最终冻结时明确报告：

```text
failed_count
```

---

# 56. 断点续跑

`build_projection_pool.py` 应支持：

```text
--resume
```

逻辑：

```text
if output exists and audit passed:
    skip
else:
    regenerate
```

建议先写临时文件，再执行原子 rename。

---

# 57. Hash 与可复现性

冻结版本保存：

```text
case manifest hash
split hash
geometry pool hash
projector config hash
generation config hash
```

如条件允许，也保存 GT source file hash 或最少保存：

```text
path
size
mtime
```

---

# 58. Canonical Fixed Geometry

虽然主模型使用随机 geometry，仍建议额外定义：

```text
canonical_fixed_geometry
```

例如：

```text
6 evenly spread views
```

用途：

- LGV-Net 对比；
- SART / TV benchmark；
- 固定视角消融；
- 论文主图。

Canonical geometry 不替代随机 train pool。

对于 TDLAS 系统，canonical set 建议定义为：

```text
6 scanner stations
station azimuth approximately:
    0°, 36°, 72°, 108°, 144°, 180°

each station:
    64 × 64 independent fan-beam measurements
```


---

# 59. 最小数据集验收标准

只有以下全部满足，数据集才能冻结。

## GT

- case split 无泄漏；
- shape / dtype 正确；
- 无 NaN / Inf；
- 物理范围正确。

## Geometry

- TDLAS scanner / beam convention 正确；
- ray direction 单位化；
- ray-box intersection 正确；
- Pool A / Pool B exact poses 独立。

## Projector

- zero phantom 正确；
- constant phantom 正确；
- compact object 位置正确；
- sampling convergence 通过。

## Projection Pool

- shape 正确；
- `float32`；
- 单位 `ppm·m`；
- 无负值；
- 无 NaN / Inf；
- invalid ray 为 0；
- manifest 与文件一一对应。

## Reproducibility

- config 已冻结；
- random seed 已保存；
- version 已记录；
- audit 已保存。

---

# 60. MVP 推荐配置

```text
Source:
    Fluent 128³ CH4 mass fraction

Frames:
    final frame only

GT:
    128³
    plus 64³ downsampled target

Projection:
    64 × 64

Geometry:
    multi-beam TDLAS fan scan
    scanner stations on horizontal x-z half-ring
    station azimuth 0° ~ 180°
    radius 5.5 ~ 6.5 m
    nominal radius 6.0 m
    height 0.25 ~ 0.35 m
    horizontal fan ±5°
    vertical steering ±4°
    beam grid 64 × 64

Train Geometry Pool:
    64 views

OOD Geometry Pool:
    32 views

Reference Projector:
    K_ref = 256
    float32
    output = ppm·m

Offline Noise:
    none

Online Training:
    V_input = 3 ~ 6
    V_hold = 1
```

---

# 61. 正式论文数据配置建议

```text
Source:
    all accepted Fluent trajectories, one final frame per trajectory

Frames:
    final frame only
    frame 60
    t = 2.0 s

GT:
    128³

Projection:
    64 × 64

Train Geometry Pool:
    128 views

OOD Geometry Pool:
    64 views

Reference Projector:
    K_ref = 256 or validated equivalent

Input Views:
    3 ~ 8

Held-out:
    1 ~ 2

Noise:
    online

Calibration Error:
    online

Geometry-OOD:
    frozen Pool B

Double-OOD:
    test cases + Pool B
```

---

# 62. 与 GeoPhy-Set3D 训练流程的接口

离线：

```text
Fluent GT
    ↓
Reference Projector
    ↓
Projection Pool
```

在线训练：

```text
Projection Pool
    ↓
Random View Subset
    ↓
Input / Held-out Split
    ↓
Noise / Geometry Perturbation
    ↓
GeoPhy-Set3D
    ↓
C_hat
    ↓
Differentiable Projector
    ↓
L_obs + L_hold + L_tv
```

因此：

> Reference Projector 不应出现在训练主循环中。

这是保证训练速度的关键设计。

---


# 63. Codex 服务器执行约束

以下规则视为强制要求。

## 63.1 固定输入

```text
SOURCE_ROOT =
/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1/
```

唯一索引入口：

```text
SOURCE_ROOT/manifest.jsonl
```

GT：

```text
SOURCE_ROOT/targets/
```

metadata：

```text
SOURCE_ROOT/metadata/
```

明确忽略：

```text
SOURCE_ROOT/inputs/
```

---

## 63.2 固定输出

```text
OUTPUT_ROOT =
/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/
```

Codex 不得把任何派生文件写入 `SOURCE_ROOT`。

---

## 63.3 固定源数据规则

```text
one manifest record
=
one final-frame GT
=
frame 60
=
t = 2.0 s
```

不得：

```text
扫描其他 Fluent runs
读取其他时间帧
重新生成三维 GT
使用原 inputs/ 作为训练 projection
```

---

## 63.4 第一阶段只生成 Debug Dataset

正式全量生成前，首先实现并运行：

```text
5 samples
16 Train stations
8 OOD stations
K_ref = 128
```

必须通过：

```text
geometry QA
zero-field test
constant-field test
beam visualization
projection visualization
unit audit
source read-only audit
```

之后再切换到 MVP：

```text
all accepted manifest records
64 Train stations
32 OOD stations
K_ref = 256
```

---

# 64. TDLAS 扫描参数最终摘要

```yaml
tdlas_scan:
  geometry_profile: "tdlas_fan_half_ring_r6m_v1"

  reconstruction_domain_m:
    x: [0.0, 0.6]
    y: [0.0, 0.6]
    z: [0.0, 0.6]

  station:
    azimuth_deg: [0.0, 180.0]
    nominal_radius_m: 6.0
    radius_range_m: [5.5, 6.5]
    height_range_m: [0.25, 0.35]
    target_m: [0.3, 0.3, 0.3]

  fan_scan:
    horizontal_full_angle_deg: 10.0
    horizontal_half_angle_deg: 5.0
    vertical_full_angle_deg: 8.0
    vertical_half_angle_deg: 4.0
    horizontal_beams: 64
    vertical_beams: 64
    rays_per_station: 4096

  projection_pool:
    mvp_train_stations: 64
    mvp_ood_stations: 32
    final_train_stations: 128
    final_ood_stations: 64

  reference_projector:
    samples_per_ray: 256
    output_unit: "ppm_m"
    dtype: "float32"
```

原先的：

```text
R ≈ 0.75 m
60° horizontal FOV
60° vertical FOV
camera / pinhole detector interpretation
```

全部取消。

新的 `64 × 64` measurement tensor 表示：

```text
64 horizontal steering angles
×
64 vertical steering angles
```

组成的独立 TDLAS fan-beam measurement map。

---

# 65. 最终数据生成原则

本版本首先固定一个最重要的数据约束：

> **每条 Fluent CFD 轨迹只使用最后一帧 `frame 60 / t = 2.0 s`，一条轨迹只产生一个基础三维 GT 样本。**

整个数据集生成流程遵循：

> **Fluent 提供三维物理场。**  
> **本项目重新定义 tomography acquisition。**  
> **所有 Ground Truth projections 离线生成。**  
> **所有 view combinations 在线随机构造。**  
> **噪声和标定误差在线注入。**  
> **Train / Validation / Test 首先按 CFD case 划分。**  
> **Geometry-OOD 在数据生成阶段独立冻结。**  
> **所有物理量保持可逆、可追溯、可复现。**

最终得到的数据集不是固定八视角数据集，而是：

> **以 Fluent 三维 GT 为基础、支持随机稀疏视角、可变视角数量、独立 Geometry-OOD 和 measurement-domain learning 的三维甲烷层析重建数据集。**
