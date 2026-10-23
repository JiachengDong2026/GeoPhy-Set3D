# 三种深度学习基线：共同协议与训练前公平性审计

本轮先冻结最终 Plain 模型，再并行实现三种独立模型。统一 `model(batch) -> FP32 [B,1,64³]` CH4 质量分数，观测数可变，输出坐标/物理量/范围与主模型一致。新增独立 dispatcher 和 CLI 位于 `scripts/deep_baselines_v1/`，不改原模型注册表、自监督损失、数据集、物理投影或历史训练入口。

## 几何信息公平性（正式训练前冻结）

GeometryLifting 本身没有可训练参数，使用非持久 XYZ32³ buffer、站位相对坐标/角度、`grid_sample` 与可见性掩码，属于固定几何采样。然而其输入的 2D encoder 特征、后续 decoder 是主模型学习模块。历史 GeoLift-DeepSets 共享这两部分，306625 参数，属于受控表示实验，不能作为本轮标准 DeepSets 或最新 Plain 对照的正式成绩。

| 模型 | 合法信息 | 内部表示 | 使用 GeoLift/主模型学习模块 |
|---|---|---|---|
| Plain | 同一观测积分、射线/站位几何与有效掩码 | 2D encoder + 固定 lifting + 集合交互 + MRGR/PRF | 自身结构 |
| Ray-DeepSets-v1 | 同一合法观测，不含 H 输入 | 逐射线203维 phi、masked mean、坐标 Fourier query MLP | 否 |
| Ray-Senseiver-v1 | 同上 | 203维 token、64×64潜变量交叉/自注意力、分块坐标 query | 否 |
| Ray-GNOT-v1 | 同上 | 测量203维/几何202维异构线性算子注意力、全局query自注意力、坐标门控专家 | 否 |

共享的是无参数 `ray_tokens`：归一化积分1维、entry/exit XYZ6维、方向3维、路径长度1维及 entry/exit Fourier192维；保留有效的零积分射线，排除无效 NaN/padding。积分不是点浓度。三个新模型各自学习表示，不访问额外观测或体积标签。掩码和几何已具有足够信息，不强迫共享主模型专属的学习编码器。

## 训练、试运行与选模

正式配置从冻结 Plain 配置复制；除模型身份、输出目录、device 和两个种子 `20261011/20261019` 外逐字段一致。300 epoch/24900 updates、batch2、FP32、AdamW、cosine、clip、loss、A/B/H 采样和 H4 selector 全部原样复用。共享模块初始化不能逐参数对齐，因为架构和随机数调用顺序不同；三基线独立同种子初始化，无 warm start、无 GT 拟合输出先验。

前置工程顺序：CPU 单元测试/主模型回归 → 独占 GPU 真实 batch2 完整 objective 与 backward/optimizer/checkpoint 往返 → 固定4train/2val、5 epoch/10update 工程 pilot（原 cosine T_max300，不缩短正式预算）。pilot 输出不能标为正式结果，2 val 不能证明科学有效性。架构/预算不根据最终测试误差调整。

正式训练 gate：CPU/GPU/接口及边界通过，pilot 有限且无明显数值故障、显存可控，并结合历史失败证据判断；工程通过不自动推翻历史科学 NO-GO。Senseiver 原自监督路径及后续有限修复已有 `REPAIR_FAILED_NO_GO`，本轮仅作固定统一接口 pilot，禁止自动重跑相同300轮。其他模型若工程稳定，可在资源允许时排队执行已冻结双种子300轮，不进行额外宽度/loss/初始化搜索。

主训练入口复用原 `paper_final_v1/train.py`，仅当前进程注入独立工厂。每 epoch 的 last.pt 是恢复权威，含 model/best/AdamW/cosine/所有 RNG/loader/metrics；中断 epoch 重播，不能改变预算。独立 run.lock、全局 GPU UUID flock、间隔15秒双空闲快照、锁 FD 继承、资源快照/冲突检测；冲突只停止本任务子进程，保留 checkpoint。每 GPU 优先独占，不抢占其他用户任务。

## 评价与边界

开发评价复用36 val案例、G1–G5各5draw的900条固定清单。先固定 checkpoint selector 记录和 SHA，保存所有预测/测量指标并 seal 后才进入独立 GT64-val CPU 评分；禁止 GT128、test、Pool B。每案例/几何先 draw 平均，再案例宏平均，独立报告各 seed；G5−G1、grand-mean G5/G1、P90/P95/max/最差案例及10000次案例 bootstrap（seed2026101801）。训练成本和推理成本单独报告，不能用原 OARI 的 Ours 预测充当当前 Plain。

正式测试仅在三者配置、selector、seed/预算、评价回归和已选 checkpoint 均冻结后考虑；历史35例仍是探索性，独立测试缺失不能由重命名消除。未运行指标留空，不以 pilot、监督旧模型或失败修复替代正式自监督成绩。
