# 可学习几何尺度融合 v2：设计与预注册

本线路在上一版 OCMSF 固定嵌入失败后，检验“几何数值条件是否应由训练学习其幅度与方向”。不改变测量监督目标、案例划分、站位采样、投影算子、归一化或原始 OARI。

新增 logits 路径为 `Conv3d(5,3,1)`，权重及 bias 均初始化为零。它接在原内容 logits 之后、三尺度 softmax 之前；原零初始化残差输出层仍保留。D 的 5 通道输入为既有 OARI descriptor，C 的输入为内容隐藏激活的前 5 通道。因此 C/D 参数量及层数匹配，D 的额外几何投影归零时可精确退化到基本内容融合。此设计没有声称 C/D 函数类完全一样；新增 18 参数使可训练张量数量匹配。特别是 C 的额外线性 head 可代数并入原内容 logits，主要改变参数化/优化，而 D 引入新的几何特征，故不能声称有效函数容量严格匹配。

| 三臂 | 参数 | 额外 logits 输入 |
|---|---:|---|
| B_multiscale | 344,678 | 无新增融合 |
| C_content | 345,435 | 内容激活前 5 通道 |
| D_geometry | 345,435 | 原 OARI 五维描述符 |

三臂共享基础/多尺度初态；C/D 新分支初态完全相同，完整初始模型输出逐位一致。原 OARI descriptor 每次计算一次，由 hook 捕获并复用 C0/C1，退出包括异常均清理缓存，无新 eigensolve。可见性 mask 的位置保持一致，未观测区域 logits 归零且分支输出归零。

每臂完整 20 轮（166 训练、36 验证，每轮 83 更新），全都在物理 GPU1 顺序执行。首种子 20261018，若 D 同时通过两个主对照，第二种子 20261019 完整重复三臂。固定第 20 轮末权重，原 300 轮 cosine 前缀，其余优化设置保持已有正式配置。

两个主对照为 D/C（额外几何条件）和 D/B（整体结构）。各自要求普通留出投影相对 L1 均值至少下降 1%，普通 P90 与五布局聚簇均值恶化不超过 2%，完整轮耗时比 ≤ 1.20，300 轮线性外推 ≤ 5 小时；正确性与输出非坍缩通过。36 案例配对 bootstrap 20,000 次，只表示案例不确定性，不包括训练种子不确定性。

同权重干预预先固定首 2 个验证案例，D 第 20 轮末 checkpoint，分别开启/关闭新增 logits，而保留原 OARI 和可见性 mask；量化 alpha、输出场差与 H 指标。它不替代重训消融，不据此改变 gate。

代码与 manifest 已在训练之前冻结，所有结果独立新目录，不覆盖旧 OCMSF。短筛完全测量域、GT guard 开启；不读取最终 35 案例。正式 300 轮与冻结三维评价由主 agent 综合多路线结果后决定。

- 数值模块：`scripts/mrgr_adaptive_geometry_v2.py`
- 运行器：`scripts/probe_mrgr_adaptive_geometry_v2.py`
- CPU 语义测试：`tests/test_mrgr_adaptive_geometry_v2.py`
- 首种子注册：`outputs/research/mrgr_adaptive_geometry_v2_seed20261018/protocol.json`
- 首种子结果报告：`docs/reports/mrgr_adaptive_geometry_v2_seed20261018_zh.md`
