# MRGR 轻量多尺度空间路径：匹配短程筛查

本实验已完成：在保留历史 MRGR 的视角交互、几何融合、测量残差反馈和自监督目标的前提下，增加轻量多尺度空间上下文，固定第 20 轮的普通留出投影误差相对降低 **7.14%**；追加的五布局聚簇困难测量面板相对降低 **13.26%**。参数仅增加 **3.80%**，独立同卡完整 epoch 测速增加约 **3.74%**。这支持将该方向列为下一轮正式验证候选；尚不能证明三维精度或 300 轮收敛优越性。

## 实现与预注册

- 实现：[`probe_mrgr_multiscale_v1.py`](../../scripts/probe_mrgr_multiscale_v1.py)。所有历史 `src`、配置和训练脚本保持不变；直接复用冻结 selfsup v1 的 dataset、sampler、objective、run_epoch。
- 处理组仅在共享 decoder 的三个 32³ 残差块之后，增加 `32³→16³→8³→16³→32³` 空间残差路径。低分辨率通道数为 12，末层 1×1×1 输出卷积的权重、偏置均初始化为零。
- 参数量从 **332,070** 增至 **344,678**，新增 **12,608（3.80%）**，满足新增不超过 10% 的预算。
- 两臂均从同一 seed `20261011` 全新随机初始化，不读取历史 checkpoint；所有共享张量逐个严格相同。CPU decoder 和 GPU 完整模型的初始前向均通过 **bitwise equal** 检查。
- 全部 **166 个训练案例、36 个 Val-A 案例**；每臂 20 个完整 epoch、1,660 次 optimizer 更新。训练 A/B 各 3–5 个视角，H2；验证固定 A4/B4/H4。A/B/H 两两不重叠，H 不进入重建网络。
- 沿用历史 FP32、batch2、AdamW `lr=2e-4`、`wd=1e-4`、clip1；损失沿用 hold1、cross0.5、observed0.1、consistency0.05。LR 是 **300 轮 cosine 的前 20 轮**，没有改为 20 轮 cosine。
- 主要比较固定第 20 轮的 Val-A H 相对 L1；短程最佳 H 仅作辅助。每案例统计先对 A/B 两次重建的 H 误差取平均，以 36 个案例为配对单位计算 bootstrap 区间。

原始注册及源文件 SHA256：[`protocol.json`](../../outputs/research/mrgr_multiscale_v1/protocol.json)。初始站点清单：[`station_epoch0.json`](../../outputs/research/mrgr_multiscale_v1/station_epoch0.json)。

## 正确性与资源检查

无更新 smoke 中，新增分支输出层梯度非零且有限；完整目标的 backward、梯度裁剪通过，所有模型权重保持不变。两臂初始完整前向严格一致。

GPU0 共享负载下，不含 optimizer step 的完整目标 backward 平均为对照 **0.797 秒**、处理 **0.808 秒**。这是两次热身后计时样本，不能当作正式吞吐估计。原本串行两臂的保守估算为 57.7 分钟，触发原 40 分钟资源门槛。

随后在任何 optimizer 更新前登记了[`资源修订`](../../outputs/research/mrgr_multiscale_v1/resource_amendment.json)：GPU3 同类 backward 为 **0.942 秒**，因此对照 GPU0、处理 GPU3 并行各 20 轮。两卡均为 RTX 4090；FP32、TF32 关闭等数值设置一致。修订仅涉及设备分配和并行资源预算，未修改数值源文件。预计墙钟约 30–36 分钟，合计 GPU 时间约 58–65 分钟；实际训练时间将单独记录。物理 GPU 和其他用户负载不同，运行时间不能作为受控架构加速比较。

另在空闲 GPU2 上，对两臂分别从头执行两个完整 epoch，直接使用相同 166 个训练案例和 36 个验证案例、相同 sampler 和目标；该独立测速不保存 checkpoint、不替代短程筛查终点。第二轮计时如下，原始记录见[`完整 epoch 测速`](../../outputs/research/mrgr_multiscale_runtime_v1/summary.json)：

| 项目 | 原 MRGR | 多尺度 MRGR |
|---|---:|---:|
| 训练每次更新 | 0.37335 秒 | 0.39137 秒 |
| 完整 epoch（含 36 案例验证） | 33.4808 秒 | 34.7329 秒 |
| 300 epoch 线性外推 | 2.790 小时 | 2.894 小时 |

该次相同设备完整 epoch 的新增耗时约 **3.74%**，计算预算具备进一步验证的可行性。外推只有两个早期 epoch 的依据，未包含启动、checkpoint 写盘，也未覆盖后续 epoch 的几何/视角组合变化和未来 GPU 竞争；**不是未来独占训练的耗时保证**。

注册和 smoke 的 GT64/GT128、禁用 NPY、外部 checkpoint 读取尝试均为 **0**。训练全过程启用 `GTReadGuard`，仅允许 Pool A train/val 投影读取。

另有主 agent 的[数据内容身份事后核查](../../outputs/research/optimization_screening_20261016/data_identity_postflight.json)：205 个数据/元数据文件的 SHA256 与先前 A、B 正式运行冻结记录全部一致，未读取体 GT 或 test。它是独立的事后核查，不是补写本实验预注册。

## 结果

### 预注册主要终点：固定第 20 轮

**以下 H/observed 均为投影测量误差，不能与历史模型的三维相对 L1 混用。** 两臂都完成 20 个完整 epoch、1,660 次更新；短程最佳 H 都发生在第 20 轮，未挑选不同训练长度。

| 指标 | 原 MRGR 对照 | 多尺度 MRGR |
|---|---:|---:|
| 训练 H 相对 L1 | 10.3547% | **9.4423%** |
| Val-A H 相对 L1（主要终点） | 9.7430% | **9.0474%** |
| Val-A H 相对 L1 P90 | 15.6033% | **14.2190%** |
| 训练总 objective | 0.0366507 | **0.0325264** |
| Val-A 总 objective | 0.0333714 | **0.0300712** |
| 短程最佳 H epoch | 20 | 20 |
| 本次进程耗时（不同共享 GPU） | 24 分 27 秒 | 29 分 44 秒 |
| 本次训练峰值 allocated | 7.665 GiB | 7.732 GiB |

主要终点相对下降 **7.139%**，36 案例中 **29 胜、7 负**。处理减对照的配对平均差为 **−0.00695588**，即约 **−0.696 个百分点**；按案例进行 20,000 次 bootstrap 得到 95% 区间 **[−0.00988706, −0.00415245]**，未跨 0。这是当前已知验证案例上的积极短程信号，区间不包含训练种子变化的不确定性。

两臂并行，训练评估墙钟约 29 分 45 秒；上表各臂耗时受各卡其他用户负载影响，不能将约 21.6% 的进程耗时差归为模块计算开销。受控一些的独立 GPU2 完整 epoch 检查见前文，增加约 3.74%。

原始结果：[`comparison.json`](../../outputs/research/mrgr_multiscale_v1/comparison.json)、[对照摘要](../../outputs/research/mrgr_multiscale_v1/reference/summary.json)、[处理摘要](../../outputs/research/mrgr_multiscale_v1/multiscale/summary.json)。

![20 轮训练与验证曲线](../../outputs/research/mrgr_multiscale_v1/curves.png)

### 追加探索：聚簇困难布局是否恶化

这项面板是在短程训练末期另行登记的**事后探索测量检查**，不改变原主要终点或 checkpoint selector。两臂均使用固定末第 20 轮 `last.pt`，在同一 GPU2 上推理；历史 clustered-V4 的 draw0–4 各 36 案例，输入四站及独立 H4 严格沿用已有 A/B 布局注册与无 GT 的预测 manifest。H 不输入重建网络，不保存或评分短程模型的 GT。

先将每个案例的五个 draw 误差平均，再在 36 案例上统计，结果如下：

| 五 draw 案例内平均后的指标 | 原 MRGR 对照 | 多尺度 MRGR |
|---|---:|---:|
| H 相对 L1 均值 | 53.1611% | **46.1124%** |
| H 相对 L1 P90 | 67.9084% | **58.6808%** |
| observed 相对 L1 均值 | 39.3430% | **36.0443%** |
| observed 相对 L1 P90 | 49.9370% | **45.1591%** |

H 均值相对下降 **13.259%**，**30 胜、6 负**；配对差 **−0.07048715**，案例 bootstrap 95% 区间 **[−0.09309403, −0.04779310]**。五个 draw 各自的 H 均值都改善，幅度约 **12.05%–15.46%**。Observed 均值相对下降 **8.384%**，22 胜、14 负。

该面板共两模型 × 五 draw × 36 案例，耗时 **12.02 秒**，全部禁止读取计数为 0。它说明本次普通测量收益没有伴随所测困难面板的测量恶化，但困难 H 绝对误差仍然较高，且不能据此声称三维困难几何已解决。记录见[面板注册](../../outputs/research/mrgr_multiscale_v1/hard_panel/registration.json)、[面板结果](../../outputs/research/mrgr_multiscale_v1/hard_panel/summary.json)，脚本为 [`probe_mrgr_multiscale_hard_panel_v1.py`](../../scripts/probe_mrgr_multiscale_hard_panel_v1.py)。

### 最终一致性检查

全部 20 轮的 train/val 案例与站点摘要、样本数、loader RNG、LR 序列均逐轮匹配；初始共享模型张量和完整前向严格相同。最终所有模型张量有限，新增 context 的 **18/18 个张量确实更新**，输出卷积权重范数为 0.4426；不是一个始终关闭的分支。独立按案例重算的 H 均值与第 20 轮验证日志一致。注册的全部数值源 SHA256 保持不变；训练、smoke、困难面板和独立核查均未读取体 GT 或外部 checkpoint。详见[`postflight_invariants.json`](../../outputs/research/mrgr_multiscale_v1/postflight_invariants.json)。

资源修订的执行入口为 [`run_mrgr_multiscale_budget_v1.py`](../../scripts/run_mrgr_multiscale_budget_v1.py)。加载处理 checkpoint 应通过原实验脚本的 `model_for(config, 'multiscale', device)` 构建，再加载 `model` state；标准历史 factory 不包含新增分支。

## 解释边界

这是一项单种子、已知开发验证集上的短程 measurement-only 筛查。早期 H 改善并不保证 300 轮最终改善，也不能推断三维 L1 或跨种子表现。新增参数较少，但没有等参数单尺度分支对照，故不能完全分离空间组织与新增容量的作用。不同物理 GPU 的浮点差异也未作为额外随机重复排除。

建议将该版本作为候选开展一次冻结协议的 300 轮测量自监督验证：保留既有 MRGR 输入、损失、采样和 checkpoint 选择规则，冻结训练源与测量普通/困难面板；训练结束并冻结 checkpoint 后，再独立评估 GT64 的普通与多布局困难三维误差。若要声称相对历史模型取得最终三维收益，需要正式结果及后续种子重复；目前不应根据这个短程结果直接替换历史模型。本轮未自行启动新的 300 轮训练。
