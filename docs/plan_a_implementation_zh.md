# 方案 A：实现、复现与验收

> **版本说明：** 当前代码已加入性能优化 v2，推荐使用 `configs/research/*perf_v2*.yaml`；实测、配置与限制见[性能优化说明](plan_a_performance_v2_zh.md)。下文的 smoke/gate 数值及 v1 预检为优化前的历史验收，原始产物保留；v1 证据不能替代当前源码的验收。

> **可选执行优化 v3：** 保留 v2 源码与配置，另有通过独立 smoke/gate 的 FP32 先验编译＋cuDNN 自动选择启动器。使用方法、运行身份与降精度实验边界见[第二轮优化说明](plan_a_runtime_v3_zh.md)；v3 配置须由 `scripts/train_plan_a_runtime.py` 运行。

本实现依据 [调研与设计报告](自由视角三维气体重建_调研与设计报告.md) 与 [研究开发指南第 7 节](GeoPhy_Set3D_研究开发指南_v1.0.md)，对应独立协议 [`scheme_a_measurement_only_v1`](../protocols/plan_a_v1/protocol.md)。历史模型、训练协议、数据划分及冻结检查点继续保留。本文件区分已实现的工程能力与尚需训练验证的研究结论。

初版 v1 按用户确认的范围交付实现与小规模验收，未启动正式训练。真实两案例 A2 smoke 完成 20 次更新并通过，外部 measurement-only NPZ 推理已完成；16 案例 gate 完成 5 轮、40 次更新且全部所列验收通过，历史数值见第 7 节及[v1 验收报告](reports/plan_a_v1_zh.md)。当前优化版的独立复验、速度与三种子正式配置见[性能优化说明](plan_a_performance_v2_zh.md)。尚无方案 A 正式三维精度结论。

## 1. 代码与版本

| 位置 | 职责 |
| --- | --- |
| [`models/set_unrolled3d.py`](../src/geophy_set3d/models/set_unrolled3d.py) | 集合初始化、A0/A1/A2/A3/A-control、显式迭代与诊断 |
| [`physics/explicit_ray_operator.py`](../src/geophy_set3d/physics/explicit_ray_operator.py) | 固定几何 gather/scatter、中点积分、精确 JVP/VJP、覆盖和预条件器 |
| [`training/research_selfsup.py`](../src/geophy_set3d/training/research_selfsup.py) | 独立配置验证、measurement-only A/B/H 训练、续训与验收 |
| [`evaluation/research.py`](../src/geophy_set3d/evaluation/research.py) | 重建 API、冻结检查、独立留出站位、三维指标和按案例配对 bootstrap |
| [`scripts/reconstruct_plan_a.py`](../scripts/reconstruct_plan_a.py) | 外部 measurement-only NPZ 的冻结检查点重建 |
| [`configs/research/`](../configs/research/) | 可执行 synthetic/smoke/gate/formal 配置 |
| [`tests/research/`](../tests/research/) | 数值、梯度、集合不变性、续训与评价边界测试 |

模型由 `build_set_unrolled_model(config)` 构建，不修改历史 `models.build_model` 的注册和行为。配置必须标注 `status: executable`、协议及 stage；研究文档中的设计模板不能直接作为训练配置运行。

| 版本 | 初始化 | 迭代 | 默认正式步数 |
| --- | --- | --- | --- |
| A0 | 历史 DeepSets sum/4 | 无 | 0 |
| A1 | 同 A0 | 精确物理梯度及几何预条件，无学习先验 | 4 |
| A2 | 同 A0 | 物理更新及共享三维先验 | 4 |
| A3 | 历史 OARI | 同 A2 | 4 |
| A-control | 同 A0 | 归一化残差图提升后输入共享先验，`Z=C`，不调用 JVP/VJP | 4 |

A0/A1/A2 在同一随机种子下的初始化子网络参数逐元素一致，沿用历史默认 sigmoid 初始化。新增模块在初始化器之后构建；全部训练版本独立 fresh 初始化，不把 A0、smoke 或 gate 检查点当作正式训练 warm start。A3 保留 OARI 的原初始化规则。

A-control 与同设置 A2 的全部可学习参数名称、形状、数量和初始值严格相同。其第三先验通道为 `tanh(tau * lift(y-p)/0.02)`，提升使用输出网格及 sum/4，第二通道为未做物理步的当前场。它仍计算前向测量残差与覆盖，参数相等不代表算力相等。大残差可能使该压缩饱和，诊断保留 `signal_saturation_fraction`，不能把控制失败直接解释为物理步骤的因果优势。

## 2. 单位、坐标与物理更新

输出 `C` 是甲烷**质量分数**，范围 `[0,1]`，张量 `[B,1,N,N,N]` 的空间顺序为 XYZ。正式输出为 64³ 端点网格，物理域 `[0,0.6]³ m`。测量 `Y` 保存为 FP32 `ppm·m`，尺度 `sY=32284.894580078173 ppm·m` 来自冻结训练 Pool A p99.5；不能先把原始测量转 FP16。

设 `a=1/16.043`、`b=1/28.965`，使用质量分数到摩尔分数的非线性转换：

```text
chi(c)  = a*c / [b+(a-b)*c]
chi'(c) = a*b / [b+(a-b)*c]^2
c_rq    = sum_j w_rqj C[idx_rqj]
p_r(C)  = (10^6/sY)*(length_r/Q)*sum_q chi(c_rq)
y_r     = Y_r/sY
E(C)    = sum_valid_r (p_r(C)-y_r)^2 / (2*m)
g(C)    = J(C)^T (p(C)-y) / m
```

**先三线性插值 C，再转换 chi，再沿射线积分。** `m` 是每个案例当前输入集合的有效射线数，包含有效的零测量；padding 和无效射线不计入，任一案例没有有效射线时所有版本均明确报错。内部只读取当前输入分支，不读取 H。

`ExplicitRayOperator.forward(C,batch_or_stencil)` 返回原始 `ppm·m`；`project_normalized`、`jvp`、`vjp` 都对应除以 `sY` 后的算子。Jacobian 转置由固定三线性权重和 scatter_add 实现。物理几何可 detach 并缓存，`C`、残差和 `chi'(C)` 保留完整自动微分链；推理支持外层 `torch.no_grad()`。

令 T 为去掉 chi、保留归一化单位的线性积分算子：

```text
d_max = 28.965/16.043
D0 = (d_max^2/m) * T^T(T*1)
eps_D = max(1e-3 * mean(D0[D0>0]), 1e-12)   # 每案例独立
D = maximum(D0,eps_D)
update = where(D0>0, g/D, 0)
tau_k = sigmoid(step_logits[k])             # logits 初值0，tau初值0.5
Z_k = clip(C_k-tau_k*update,0,1)
```

D0 是几何行和对角 majorizer，不是精确 `diag(J^T J)`，也不构成整个学习网络的全局收敛保证。首版不启用回溯。未覆盖节点的物理更新严格为零，先验可在这些节点补值。每步 E 允许小幅非单调，低输入残差也不证明三维恢复唯一或准确。

独立消融 `preconditioner: none` 的定义严格为 `update=unpreconditioned_scale*g`，默认 scale 为 1，不做隐藏归一化；它不属于首轮默认 A2 协议。

## 3. 主线网络与计算设置

初始化器为共享 2D 编码器：3→32 卷积、GN(8)、SiLU、3 个宽度 32 的残差块；真实站位及 f/r/u 基决定角度提升，在 32³ 端点网格双线性采样。融合严格采用可见特征之和除以固定 4，**不按实际视角数取均值**。解码器保持历史 32³→64³、32→24 通道结构及默认输出 sigmoid。

A2 的 K=4 步共享同一个宽度 32 的三维先验：9→32 卷积、GN(8)、SiLU、3 个三维残差块、32→1 卷积；末层权重和偏置为零。其输入依次为：

1. 当前 `C_k`；
2. 物理更新后的 `Z_k`；
3. `tanh(update/0.02)`；
4. 每案例最大值归一化后的 `log1p(T^T 1)`；
5. 输出网格上的可见站位数/真实站位数；
6. 当前步 4 维可学习 embedding，广播到全体节点。

输出 `C_{k+1}=clip(Z_k+0.02*tanh(prior(inputs)),0,1)`。0.02 是候选先验幅度，不是物理常数。物理更新本身不经过上述 tanh 压缩。

积分保持 `Q=96`；API 默认 `ray_chunk=256`，性能 v2 配置设为 1024。固定几何使用各 512 MiB 的 CPU/CUDA LRU，模型与训练外层投影共享缓存；GPU 缓存超出预算时淘汰，设置 `device_cache_budget_bytes=0` 可回退分块传输。预算只限制缓存持有的数据，完整反向图的显存另计。v1 启用非重入先验重计算；当前性能 v2 在实测显存可容纳时关闭 `checkpoint_prior`，保持完整梯度并减少重算。显存统计须注明实际 batch、K/Q、诊断和 checkpoint 设置。

`model(batch)` 默认返回浓度 Tensor；`model(batch,return_diagnostics=True)` 返回 `volume`、`C0`、每步场与 E/残差、物理和先验更新范数、tau、clip/近零/近一比例、覆盖、D0、eps_D 等。诊断保留计算图及中间体场，长期推理应使用默认精简返回。

## 4. 自监督、数据与追溯

训练数据保持 166 train/36 val/35 test 的冻结划分。训练及模型选择只读 train/val Pool A 测量；测试集不开放，Pool B 仅在正式冻结后独立评价。训练 A/B 各随机 3–5 站，H 为 2 个站，三者互不重叠；验证固定 A4/B4/H4，采样清单保持 epoch 0。

外层保留历史 robust/self-supervised 协议：`hold + 0.5*cross + 0.1*observed + 0.05*consistency`，测量项为归一化 SmoothL1（beta=0.01）加权 0.1 的相对 L1。它与内层平方 E 的作用不同。使用 FP32、AdamW(lr=2e-4,weight_decay=1e-4)、cosine、梯度范数裁剪 1、有效 batch 2，无 early stopping。配置使用物理 batch 1、累积 2，尾批按实际案例数处理。

每个 run 独立输出解析配置、源码和数据 SHA256、Git commit、环境、采样清单、initial/zero baseline、逐轮指标、测量干预、读取审计、best/last 检查点和验收记录。`GTReadGuard` 拒绝 GT64/GT128、训练之外的 NPY 和外来/监督检查点读取；H 从不进入模型内部。模型选择仅用平均留出测量相对 L1。

资产冻结清单见 [`assets_before.json`](../protocols/plan_a_v1/assets_before.json)。当前已保存的 [`assets_audit.json`](reports/artifacts/plan_a_v1/assets_audit.json) 检查旧源码、数据元信息及基线检查点，并实际回放历史基线测量指标；[`operator_numerics.json`](reports/artifacts/plan_a_v1/operator_numerics.json) 保存 CPU FP64/CUDA FP32 数值检查。重跑审计时使用新输出路径，保留已有证据。

## 5. 实际命令与阶段约束

以下命令均在仓库根目录执行，使用现有环境和入口；正式训练命令属于后续操作示例，**不是已执行或已通过的记录**。已存在的 run 不能以 fresh 方式覆盖；重做时新建 run 名及配置版本。

```bash
cd /home/djc/projects/GeoPhy-Set3D
export PYTHONPATH=src:tests
PY=/scratch/djc/envs/gas-torch/bin/python

"$PY" -m pytest tests/research -q
"$PY" scripts/verify_plan_a_operator.py --device auto --output outputs/research/operator_numerics_recheck.json
"$PY" scripts/audit_plan_a_assets.py --replay-baseline --device cuda:0 --output outputs/research/assets_audit_recheck.json
"$PY" scripts/profile_plan_a.py --config configs/research/scheme_a_smoke.yaml --device cuda:0 --output outputs/research/plan_a_profile_recheck.json
```

`profile_plan_a.py` 对真实 train/Pool A 案例完成全 A/B/H 前向和 backward，不执行 optimizer 更新；若 OOM，写出失败 JSON，不能把失败当作已满足显存条件。算子验证使用解析场及合成几何，不读取体场标签。

合成 smoke 只检查工程链路，使用显式标记的小网格/Q 设置，不替代真实 64³/Q96 证据：

```bash
"$PY" scripts/train_research_selfsup.py --config configs/research/scheme_a_synthetic_smoke.yaml --run-name scheme_a_A2_synthetic20_recheck
```

真实 smoke 已按固定 2 train / 2 val 案例、20 次更新完成，实际运行命令为（再次执行需更换 run 名）：

```bash
"$PY" scripts/train_research_selfsup.py --config configs/research/scheme_a_smoke.yaml --device cuda:0 --run-name scheme_a_A2_smoke_v1
```

真实 gate 为固定 16 train 案例、8 val 案例，受 `max_updates=2000`、`max_seconds=1800` 双重预算约束。根据真实 smoke 的实测耗时，在任何 gate 更新前另行登记了 5 epochs / 40 更新的预算配置；决定和估算见 [`gate_budget_decision.json`](../protocols/plan_a_v1/gate_budget_decision.json)。原 80 epochs 草案配置保留。预算到达但完整 gate 未结束时登记未完成/NO-GO：

```bash
"$PY" scripts/train_research_selfsup.py --config configs/research/scheme_a_gate_budget_v1.yaml --device cuda:0
```

验收记录包括 finite、优于零体场、优于初始化、非零梯度、案例间差异、改变测量后的输出响应及禁读尝试为零。正式入口要求同模型设置、同数据、同源码版本的真实 smoke/gate 都通过；合成产物或配置验证成功不能代替这两项证据。

已创建三个独立 A2 正式配置，acceptance 指向本轮的真实 smoke 和预算 gate；三份均通过配置验证。`--validate-only` 只检查配置结构，不替代正式入口的验收、源码和数据内容核对。以下训练命令保留供后续使用，本轮未执行：

```bash
for seed in 20261011 20261012 20261013; do
  "$PY" scripts/train_research_selfsup.py --config "configs/research/scheme_a_a2_formal_v1_seed${seed}.yaml" --validate-only
done
# 后续正式训练示例；本轮不启动。
for seed in 20261011 20261012 20261013; do
  "$PY" scripts/train_research_selfsup.py --config "configs/research/scheme_a_a2_formal_v1_seed${seed}.yaml" --device cuda:0
done
```

三次 run 独立 fresh，各 300 epochs；全量 166 train、有效 batch 2 对应每 seed 24,900 次 optimizer 更新，三 seed 合计 74,700 次。三种 seed 固定为 20261011/12/13。A0/A1/A3/A-control 的正式模板也已提供，但每一变体都要先生成匹配自身 `model` 字段的 smoke/gate 配置与证据，不能直接复用 A2 的验收。

续训只接受已原子保存的完整 epoch 边界，并恢复模型、AdamW、scheduler、采样 epoch、Python/NumPy/Torch/CUDA/loader RNG。配置、源码、Git commit、数据/几何/统计量不符会拒绝恢复。对上述真实 smoke 恢复需使用相同 CLI 覆盖项：

```bash
"$PY" scripts/train_research_selfsup.py --config configs/research/scheme_a_smoke.yaml --device cuda:0 --run-name scheme_a_A2_smoke_v1 --resume
```

epoch 中断后只承认最近完整检查点，不承认未提交更新。新的消融、尺度或初始化设置须创建新实验，不以 `--resume` 偷换协议。

## 6. 推理与独立评价

外部输入提供 [`reconstruct_plan_a.py`](../scripts/reconstruct_plan_a.py) CLI 和 Python `ResearchReconstructor` API。CLI 接收 measurement-only NPZ，字段对应 [指南 5.1 的单分支 batch 契约](GeoPhy_Set3D_研究开发指南_v1.0.md#51-单分支-batch-的契约)。NPZ 已有 batch 维，不接收 GT 或 H 字段，也不从图像推测站位。

| NPZ 字段 | 形状与含义 |
| --- | --- |
| `input_projection_raw` | `[B,V,1,64,64]`，原始 FP32 ppm·m |
| `input_valid_ray`、`view_mask` | 分别 `[B,V,1,64,64]`、`[B,V]`，布尔掩码 |
| `input_path_length_m`、`input_t_near` | `[B,V,1,64,64]`，米 |
| `input_ray_direction` | `[B,V,64,64,3]`，真实 XYZ 单位方向 |
| `input_station_origin_m` | `[B,V,3]`，米 |
| `input_center_direction`、`input_right_direction`、`input_up_direction` | `[B,V,3]`，与冻结扫描几何一致的 f/r/u 基 |
| `horizontal_angle_deg`、`vertical_angle_deg` | `[B,64]`，递增角度轴，水平对应列、垂直对应行 |
| `input_projection` | 可选，同 raw 形状；若提供必须等于 raw/检查点训练尺度 |
| `input_station_azimuth_deg` | `[B,V]`；A3/OARI 需要该几何描述，其余版本可省略 |
| `input_t_far`、`input_station_indices`、`projection_scale_ppm_m` | 可选；CLI 可补 far 和 station index，提供尺度时须与检查点匹配 |

```bash
"$PY" scripts/reconstruct_plan_a.py --run-dir outputs/research/scheme_a_A2_smoke_v1 --input external_measurements.npz --output outputs/research/external_reconstruction_v1 --device cuda:0
```

真实 smoke 已完整冻结，已用真实 train 测量导出的 NPZ 实际调用本入口；[推理记录](reports/artifacts/plan_a_v1/external_inference.json)保存输出形状 `[1,1,64,64,64]`、有限值及无 GT 读取证据。上述示例命令需自行提供 NPZ 路径；检查点未完成或哈希不符会拒绝。工程 smoke 检查点可用于工程推理，formal run 会额外经过完整冻结审查。输出新目录保存 `field_mass_fraction_xyz.npy`、`input_reprojection_ppm_m.npy`、可用时的 `coverage.npy` 以及含阶段、输入/检查点指纹、逐步标量诊断和耗时的 `reconstruction.json`。工程推理产物不能替代正式模型的独立 GT 验收。

需要直接集成时，可先准备同一契约的 measurement-only batch，再调用 API：

```python
import torch
from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
from geophy_set3d.evaluation.research import ResearchReconstructor

# checkpoint_path 指向可信本地检查点；input_batch 已按既有数据接口构建，且不含 GT/H。
payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
device = 'cuda:0'
model = build_set_unrolled_model(payload['config']).to(device)
model.load_state_dict(payload['model'])
result = ResearchReconstructor(model).reconstruct(input_batch, output_grid=64)
volume_xyz = result['field_pred']
raw_input_reprojection = result['input_reprojection_ppm_m']
```

`input_batch` 的 Tensor 需已放在相同 device；结果声明质量分数、XYZ、域范围，并分别记录完整重建与重投影耗时。诊断可用 `options={'diagnostics': True}`，会增加保存中间结果的开销。API 本身可用于工程检查点推理；进入正式 GT 评价必须另外通过冻结边界。

现有 CLI 是**固定 val 数据集的冻结模型评价**，只接纳已完成 300 epochs、无 GT 训练/选择且检查点哈希匹配的 formal run：

```bash
"$PY" scripts/evaluate_research_selfsup.py predict --run-dir outputs/research/scheme_a_A2_v1_seed20261011 --device cuda:0 --views 4 --layout random --draw 0 --pool train
"$PY" scripts/evaluate_research_selfsup.py score --predictions-dir outputs/research/scheme_a_A2_v1_seed20261011/evaluation/val_train_random_v4_draw0
```

`predict` 用真实角度选定输入布局，额外选 4 个不重叠 H 站位；模型只接收输入站位，输出 `observed` 和 `heldout` 测量指标，并先保存全部预测和 SHA256 清单。`score` 重新核对边界、所有预测哈希及 val membership 后才打开 GT64。支持 `--layout random|spread|clustered`、`--pool train|ood`、draw 和 views；这不是已经跑完五档几何/三 seed 或最终 test 的声明。

三维评价包含 relative L1/L2、NMSE、MAE/RMSE、峰值与位置、浓度积分、质心、Dice/IoU 和背景浓度积分。端点网格物理积分使用梯形权重；质量分数积分没有额外密度模型时不能称甲烷总质量。`paired_case_bootstrap` 先按案例平均布局/噪声差异，再按案例抽样；该函数不自动消除训练 seed 的不确定性。

## 7. 已验证能力与最终验收登记

当前工程测试覆盖常量场、旧前向一致性、JVP 差分、伴随恒等式、VJP 与 autograd、完整残差梯度及高阶梯度；真实两步卷积网络覆盖初始化器/步长/先验关键参数差分、checkpoint 数值与梯度一致性、乱序、padding、混合 V、no_grad、NaN padding、全无效输入，以及 A-control 的参数匹配和不调用 Jacobian。GPU 检查只支持所测设备/版本，不能外推所有部署环境。

本轮执行范围为实现与小规模验收。正式三 seed、300 epochs 训练未启动，独立 GT64 三维评价和 test 评价未执行。三个正式配置均通过 `--validate-only`，并通过当前源码、完整数据内容与真实 smoke/gate 的[只读验收预检](reports/artifacts/plan_a_v1/formal_acceptance_preflight.json)；未创建正式训练模型或 optimizer，未执行正式更新。这与完整研究验收分开登记。

相关回归共 **70 passed、5 warnings**，完整命令如下，日志见 [pytest.log](reports/artifacts/plan_a_v1/pytest.log)；这是相关测试集合，不代表全仓库测试：

```bash
PYTHONPATH=src:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/scratch/djc/envs/gas-torch/bin/python -m pytest -q \
  tests/research tests/test_deepsets_comparison.py tests/test_selfsup_mrgr.py \
  tests/test_selfsup_evaluation_boundary.py tests/test_differentiable_projector_resolution.py \
  tests/test_variable_view_collate.py
```

CPU FP64 的旧前向对齐、JVP 差分、转置恒等式及二阶差分误差分别为 8.56e-17、5.71e-11、1.23e-16、2.81e-10；部署 CUDA FP32 检查也通过预定容差。真实 train/ood 几何轴、路径与方向核对通过，见 [geometry_audit.json](reports/artifacts/plan_a_v1/geometry_audit.json)。

真实 smoke 在 `outputs/research/scheme_a_A2_smoke_v1` 完成 20 次更新，K4/Q96、lift32/output64、先验宽32、FP32、microbatch1/累积2及精确 checkpoint。初始 H 相对 L1 为 103.447910，最终/最佳为 0.982465，零场为 1；训练 loss 从 23.205119 降至 0.252650。峰值 GPU allocated 4.184 GiB；同几何测量乘 1.2 后，体场平均绝对变化 4.85443e-4。全部所列工程验收通过，三个禁读计数均为零。

真实 gate 在 `outputs/research/scheme_a_A2_gate16_budget_v1` 完成固定 16 train/8 val、5 epochs、40 次更新，未触发 1800 秒上限。初始 H 相对 L1 为 71.118650，五轮依次为 1.478407、0.439847、0.338928、0.296043、**0.289196**，零场为 1；训练 loss 从首轮 6.459786 降至末轮 0.089051。记录的训练加逐轮测量验证共 1360.0 秒（不含首次验证及收尾），峰值 GPU allocated 为 4.143 GiB。同几何测量乘 1.2 后，体场平均绝对变化 4.41741e-4。全部九项验收为 true，三个禁读计数均为零；smoke/gate 与训练时 v1 源码指纹一致，不能替代当前 v2 源码的验收。预算方案在 gate 更新前登记，模型、损失和通过条件未在运行中修改。

外部 measurement-only NPZ 的实际推理输出为 `[1,1,64,64,64]`，数值有限且在 `[0,1]`。单次共享 GPU 重建 2.113 秒、另行重投影 0.160 秒，不作为稳定性能基准。工程模型的三维输出没有进行 GT64 精度评分。

历史 36 val 基线测量回放相对 L1 为 0.03275178736，与原记录差 5.85e-10；[收尾资产审计](reports/artifacts/plan_a_v1/assets_after_implementation.json)再次确认历史数值源码、数据元信息和基线检查点未改。完整源码、配置、数据及检查点 SHA256 见[生成报告](reports/plan_a_v1_zh.md)和各 run 的 provenance/freeze 记录；[环境快照](reports/artifacts/plan_a_v1/environment_snapshot.json)保存实测依赖与 GPU 信息，未进行全新环境安装验证。

以上结果支持当前工程实现可训练、可推理；尚无方案 A 优于历史 MRGR/DeepSets 的三维精度结论。
