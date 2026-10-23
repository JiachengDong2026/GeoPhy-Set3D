# Ray-Senseiver-V1：独立接口、来源审计与历史科学门槛

本轮完成可运行的独立 Senseiver 接口及 CPU 验证，**没有新增 Senseiver 科学训练、没有读取 GT、没有修改原 Senseiver 或最终主模型**。工程测试通过不改变历史纯自监督 `REPAIR_FAILED_NO_GO`。应把本实现列为“工程可运行、正式双种子训练受历史科学门槛限制”，不能当作新成功基线。本文记录模型子任务；统一 loss、单 batch GPU smoke、资源锁、小 pilot 及最终执行状态由总集成报告记录。

## 实现与来源

入口为 `geophy_set3d.models.baseline_senseiver_v1:RaySenseiverV1`，构造参数只有 `query_chunk_size=16384, decoder_checkpoint=True`，配置见 `configs/baselines/senseiver/model_v1.json`。模型直接继承既有 `RaySenseiverSS`：254,913 个可训练参数，FP32 权重约 0.9724 MiB，不含 checkpoint 元数据/优化器。默认 seed 相同条件下，原模型与新包装所有 state tensor 及完整输出逐位相等；新增包装只校验显存控制选项、关闭外层 autocast 并显式要求 FP32 参数。没有读入历史或监督权重，没有输出尺度重初始化、latent 残差或训练配置搜索。

原论文：[Development of the Senseiver for efficient field reconstruction from sparse observations](https://www.nature.com/articles/s42256-023-00746-x)。公开实现：[OrchardLANL/Senseiver](https://github.com/OrchardLANL/Senseiver)，固定 commit `e443eb0ca188b5e83a0e6ce5afb8db7d90393de1`，不是未经核查地跟随上游最新版本。`third_party/senseiver/` 保留 BSD-3-Clause 许可及 LANL 原 notice；既有 `senseiver_core.py` 保留同一完整 notice。

2026-10-23 本地重新计算 `model.py`、`network_light.py`、`positional.py`、`s_parser.py`、`LICENSE.md` 五文件 SHA256，全部与既有 `docs/reports/artifacts/ray_senseiver_selfsup_v1/official_source.json` 相符。另实际在线读取上述固定 commit 的 `LICENSE.md`，SHA256=`62dd42b1e724f7bdc862906a5949772da7ad2a52133038414806e17206d80ceb`，与本地和来源登记相同；随后在线复核 `model.py` 的请求超时，该文件本轮仅完成本地历史 pin 核查及官方 core 数值回归。许可允许适配及再分发，保留通知/条款，不主张原作者背书。

原论文是稀疏**点值**观测与有监督场重建；本任务是射线积分输入、自监督物理测量目标。这里明确命名为 **Ray-Senseiver 任务适配**，不称原论文精度复现，也不采用 dynamic sensor placement。

## 信息与结构

每条有效射线一个 203 维 token：积分值/固定 `Y_SCALE=32284.894580078173`（1维）、域内进入和离开位置（各3维，映射到[-1,1]）、单位方向（3维）、长度/(√3×0.6m)（1维）、进入/离开位置的16频带 Fourier 特征（各96维）。使用原 `ray_tokens`；全部 valid rays 保留，包括精确零测量；invalid ray 和 padded view 不进入观测集合。空集合使用原零 sentinel，避免全 masked attention 产生 NaN，该 sentinel 不是新增物理观测。

网络保留官方 Perceiver 风格结构：203→64 token 投影；64×64 learned latent；latent 对完整观测集合做 cross attention，随后每个 block 三层 latent self attention，共三个 block，其中首 block 独立、后一个 block 共享执行两次。所有 attention 为4头、dropout=0。输出使用完整 262,144 个 XYZ 坐标查询（16频带、96维），与 learned output embedding 拼接后由原 query cross attention 和线性层、sigmoid 给出质量分数。网格 XYZ 顺序、端点 [0,0.6]m、分辨率64³与最终主模型一致，输出 FP32 `[B,1,64,64,64]`，范围[0,1]。

只访问 batch 的 `input_projection_raw`、`input_station_origin_m`、`input_ray_direction`、`input_t_near`、`input_path_length_m`、`input_valid_ray`、`view_mask`。没有 GeometryLifting、GeoPhy encoder、Plain Set Interaction、MRGR、PRF 或额外标签；H 测量不构造成输入 token。既有 `assert_no_gt` 拒绝 GT 字段。与主模型共享的是合法信息和输出物理定义，不是主模型专属的可训练几何表示。

编码 attention 规模为 O(N_ray×64) 和 O(64²)，不构造 N_ray² 的自注意力矩阵；完整 decoder 的 attention 计算为 O(262144×64)。按最多16,384 query 分块，训练时可对各块 checkpoint，不删 ray、不抽样 query、不减输出分辨率。`need_weights=False` 保留既有 PyTorch attention 路径；实际 kernel、GPU 峰值和耗时须由统一 smoke/benchmark 实测。仅完整 query Fourier 非持久 buffer 就有96 MiB，坐标 buffer另3 MiB，不能把参数大小当作总运行显存。

## 历史结果与不可混用边界

| 已有实验 | 实际结果 | 本轮处理 |
|---|---|---|
| 原纯自监督 Ray-Senseiver-SS | 113 epoch、9,379 updates 后在保存边界停止；Val H L1=1.00000095，`FAILED_LEARNING / NEAR_ZERO_COLLAPSE / EARLY_TERMINATED` | 复用失败证据，不把已停止 run 续成300轮 |
| 输出尺度初始化两臂 | 各35 epoch、140 updates，均 `FAILED_TO_LEARN`；Treatment H L1=1.0055393 | 该先验已测试且未解决学习问题，不再以“新最小修复”重做 |
| relative-L1 开/关 | `PARTIAL_LEARNING_EVIDENCE`，正式 `NO-GO` | 不把局部指标改善解释为有效重建 |
| consistency 开/关 | 共同35轮，Treatment H L1=1.0605742，测量依赖达标0例，正式 `NO-GO` | 不为了改善表现另改本轮共同 loss |
| 唯一 global-latent 残差修复 | Control35轮/140更新，Treatment30轮/120更新；共同30轮H L1=1.0606595/1.0352792，两臂 `FAILED_TO_LEARN`，最终 `REPAIR_FAILED_NO_GO` | 原修复及失败记录冻结，不把其结构移入新模型 |
| 有监督 Ray-Senseiver | GT64 参与梯度，187完整轮并下一轮59更新，2小时上限停止；有测量依赖，900条已评价 | 监督条件不同，仅作历史旁证；不加载其权重、不计入纯自监督榜单 |

证据对应 `docs/reports/ray_senseiver_init_probe_v1_zh.md`、`ray_senseiver_relative_probe_v1_zh.md`、`ray_senseiver_consistency_probe_v1_zh.md`、`ray_senseiver_quick_repair_v1_zh.md` 和 `ray_senseiver_supervised_v1_zh.md`。本轮没有读历史 checkpoint 或重新评价历史三维 GT。监督参考能学不等于本轮自监督路线通过；历史失败也不等于原 Senseiver 在其原任务上的能力上限。

新用户任务授权重新审计和独立小 pilot，但没有理由把原结构、同一测量监督的短暂 loss 下降直接升级为两种子300轮。可进行独立极小统一目标 pilot 验证完整 loss、更新、保存和有限数值；它属于工程复核。若无避免趋零、有效测量依赖及 H 误差持续优于零场的科学证据，正式训练应保持阻塞，记录未运行。不得自动追加初始化/relative/consistency/latent/宽度搜索，也不把历史失败全量预算再次无条件提交。统一控制器应明确保存该 gate 的决定与依据。

## 已执行 CPU 验证

命令：

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python -m pytest -q \
  tests/test_baseline_senseiver_v1.py tests/test_ray_senseiver.py
```

结果：**10 passed in 6.02s**，包括新增6项与旧模型4项。第一次启动漏设 `PYTHONPATH=src`，collection 阶段即 `ModuleNotFoundError`、未运行测试；补齐仓库既有 src 路径后执行上述命令通过，没有修改模型来规避失败。

验证覆盖：参数量/默认 state 精确一致、完整64³前向、嵌套 CPU autocast 下仍FP32输出；batch1/2、观测1/2/3、有效零/部分 mask/空集合；观测重排与 masked 内容不影响 latent；同一状态129条查询整块与19条 checkpoint 分块的预测、坐标梯度及全部模型参数梯度一致，encoder/latent/decoder梯度有限非零；state 保存恢复、GT 字段拒绝；非法分块选项拒绝。原回归另验证 pinned 官方 core 的 forward、共享参数执行语义及真实 measurement-only 几何。反向测试使用129查询降低 CPU 开销，不冒称已完成本子任务的完整64³ GPU loss smoke；该项由集成单独执行。

随后新增跨模型合成契约测试 `tests/test_baseline_contract_v1.py`，单独执行 **3 passed, 3 subtests passed in 9.04s**。该测试让三模型实际输出完整64³，再调用原 `selfsup.objective`、原六路 K96 物理投影及完整 loss backward；射线图仍为64×64，每幅只4条合成有效射线，以控制 CPU 成本，没有 mock loss/projector 或读任何真实数据。验证所有参数梯度有限、A/B/H两两不交、网络只见A/B、H目标和几何不获梯度、GT字段/错误case/重叠站位在模型调用前拒绝、selector仅由H relative L1决定。GNOT仅在此CPU fixture缩为width8/layer1、两异质源/全局self算子/两专家门控均保留，正式配置不变；Senseiver使用原完整结构。首次测试有指标转标量的 autograd warning，测试在只读评分处使用 `no_grad` 后单次重验通过，没有改旧评分函数或模型源码。

本实现不需要新增第三方运行时依赖。没有使用 GPU，没有科学 optimizer 更新，没有独立/历史正式测试读取，没有改写既有文件。正式指标、两个训练 seed 成绩和实测 GPU 成本本子任务均为**未运行**，不得填写估计精度。
