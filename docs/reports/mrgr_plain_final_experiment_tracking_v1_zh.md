# 普通集合交互最终模型实验追踪表

更新时间：2026-10-23T06:55:16.183730+08:00。本轮仅登记六个消融，结果为既有35案例上的探索性分析。

| 实验 | 种子 | 状态 | epoch / 更新 | 参数 | best SHA256 |
|---|---:|---|---|---:|---|
| uniform_plain_set_interaction_seed20261011 | 20261011 | 已核验，复用 | 300 / 24900 | 342209 | `0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c` |
| uniform_plain_set_interaction_seed20261019 | 20261019 | 已核验，复用 | 300 / 24900 | 342209 | `fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d` |
| uniform_plain_set_interaction_no_geometry_lifting_seed20261011 | 20261011 | complete | 300 / 24900 | 342209 | 9c1edf3945fbb10a096bab08369379b4c0d40075805dbb8508ac35d8297eca83 |
| uniform_plain_set_interaction_no_geometry_lifting_seed20261019 | 20261019 | complete | 300 / 24900 | 342209 | 7c29f08bff10297fde9b566f805b0cb42ec339ca83f9e6a557b00ef7d03781f8 |
| uniform_plain_set_interaction_single_scale_mrgr_seed20261011 | 20261011 | complete | 300 / 24900 | 342209 | 0658ac3211e8739db4329f8377b3dde4e818633638806b696702867d7af846fc |
| uniform_plain_set_interaction_single_scale_mrgr_seed20261019 | 20261019 | complete | 300 / 24900 | 342209 | b48a1fa67da0e33d516653024c1e522845cb8ab4aebf55d56ac6ca5387853d38 |
| uniform_plain_set_interaction_no_prf_seed20261011 | 20261011 | complete | 300 / 24900 | 329697 | 873dd93fa8161adbaf6ec1e7155c103728bfe756399390c17dda1de027704266 |
| uniform_plain_set_interaction_no_prf_seed20261019 | 20261019 | complete | 300 / 24900 | 329697 | 7ed0fcff103141ff5ddd9340b75e0c2989ca4fbf3f5f34956285e607558a533a |

完整状态：`/scratch/djc/GeoPhy-Set3D/mrgr_plain_core_ablations_v1/status.json`。日志、GPU快照和异常记录：`/scratch/djc/GeoPhy-Set3D/mrgr_plain_core_ablations_v1/scheduling/`。

完整模型不重训。OARI历史消融只复用定义，不能代替普通交互消融。所有六项300轮及全局预测封存完成后才允许本轮GT64评分，GT128禁止。

## 2026-10-24 追加：独立深度学习对照准备

上表消融结论及权重身份不变。最终Plain再次冻结并通过集成前/后逐位CPU回归；未重训主模型。新增Ray-DeepSets-v1（104577参数）、Ray-Senseiver-v1（254913）、Ray-GNOT-v1（527109）的独立实现与配置，CPU/GPU/5epoch工程pilot和四方法Val mini评价均通过。

DeepSets/GNOT各seed20261011/20261019，4项正式300epoch/24900updates已登记并进入GPU持久队列；当前有效输出为 `/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/training_registered_v2/`。Senseiver保留历史科学NO-GO，未自动重复300轮；所有未运行正式指标留空。mini训练量不对等，不构成正式精度比较。未启动新test/GT128或Pool B访问。

具体状态、哈希和完整边界见[执行报告](deep_learning_baselines_status_zh.md)、[正式状态CSV](artifacts/deep_baselines_v1/formal_comparison_status.csv)及实时 `deep_baselines_v1/queue_status.json`，不将队列中任务视为已完成。

## 2026-10-24 追加：暂停训练及两个seed20261011完整预算模型评价

用户要求暂停训练，队列及本任务训练进程组已于16:05:23+08停止，未干预其他用户任务。DeepSets seed20261019保留epoch240/19920updates；GNOT seed20261019无完整epoch提交；后续训练不自动恢复。

| 独立对照 | 种子 | 完整预算 | best epoch | 当前评价结论 | checkpoint SHA256 |
|---|---:|---|---:|---|---|
| Ray-DeepSets-v1 | 20261011 | 300 / 24900 | 291 | 35案例/17设置已完成，近零塌缩 | `f42ce9f3a1ca2093ccfcb1452314113c3922b02649ad11ddf540a5ea411a53e7` |
| Ray-GNOT-v1 | 20261011 | 300 / 24900 | 300 | 35案例/17设置已完成，近零塌缩 | `c86c90cd5b2fdbefe389d6253a31b843f45967ae5c5d763b60041d9077d709c7` |

主模型595份C1预测复用，两基线新生成1190份，共1785份全部封存后评分35个授权GT64；未读GT128/Pool B，未使用test选模，Plain双种子权重未改。结果仍是历史案例上的单训练种子探索性比较，不能替代独立测试或完整三算法双种子比较。科学失败如实保留，不自动做测试后修复。

[本次正式中文报告](deep_baselines_seed20261011_test35_v1_zh.md) · [当前追踪CSV](artifacts/deep_baselines_test35_v1/experiment_tracking.csv) · [验收](artifacts/deep_baselines_test35_v1/acceptance.json)。此前队列描述为历史启动状态，以此条和暂停记录为准。

## 2026-10-24 追加：塌缩审计与修复pilot最终NO-GO

完成独立物理/尺度/掩码、损失梯度、原优化器/恢复审计，以及每模型两个有依据的修复候选。v2仅修改末层初始化，尺度来自全166个train案例合法A/B测量；v3保持相同初态，仅将独立基线consistency改为齐次比值。这是新的loss协议，原selfsup.py和冻结Plain均未修改。

| 修复pilot | 种子 | epoch / 更新 | 初始完整val H L1 | 最终完整val H L1 | 零场 | 测量干预通过案例 | 正式训练资格 |
|---|---:|---|---:|---:|---:|---:|---|
| DeepSets v2 | 20261011 | 20 / 320 | 1.938704265 | 1.000001102 | 1.0 | 0/4 | NO-GO，近零塌缩 |
| DeepSets v3 | 20261011 | 30 / 480 | 1.938704265 | 1.000662862 | 1.0 | 0/4 | NO-GO，pilot预算耗尽 |
| GNOT v2 | 20261011 | 10 / 160 | 1.933147622 | 1.000190454 | 1.0 | 0/4 | NO-GO，无足够科学趋势 |
| GNOT v3 | 20261011 | 10 / 160 | 1.933147622 | 1.000256099 | 1.0 | 0/4 | NO-GO，无足够科学趋势 |

均使用固定32train/完整36val、64³、FP32、全部有效射线与原测量信息边界；未降低<=.99科学门槛。共1120次pilot更新，另4次独立smoke更新；新增正式训练更新为0。没有修复成功结论，不自动开展更多搜索、第二种子或Senseiver；原队列继续暂停。源码/数据身份、30个共同epoch的站位与LR、全部GO条件已独立复核，主模型最终CPU逐位回归通过。原失败和双种子Plain权重未覆盖，无GT数组/GT128/test/Pool B访问。

[根因与修复报告](deep_baselines_collapse_root_cause_repair_zh.md) · [完整pilot追踪及checkpoint哈希](artifacts/deep_baselines_repair_v2/pilot_summary.csv) · [独立复核](artifacts/deep_baselines_repair_v2/independent_pilot_verification.json) · [最终验收](artifacts/deep_baselines_repair_v2/acceptance.json)。权重保留在独立scratch目录，Git同步代码、配置、诊断日志与报告；这些pilot不进入正式精度排名。

## 2026-10-24 追加：条件学习能力 A/B/C 独立诊断完成

本轮从全新 seed20261011 测量尺度初态开始，H 完全排除于训练目标，训练仅为 `0.5 cross + 0.1 observed + 0.05 consistency`。真实 train 前四例按嵌套 1/2/4 例构成 A；四个预登记解析羽流构成 B；只读历史权重的五种测量干预与逐层梯度构成 C。禁止将此轮诊断与原正式训练或前轮修复 pilot 混算。

| 诊断 | 运行数 / 有效更新 | 主要结果 | 科学结论 |
|---|---|---|---|
| A：DeepSets v2，1/2/4 例 | 3 / 600 | 训练 AB L1 为 1.000001–1.000117 | 三项 NO-GO，未优于零场 |
| A：GNOT v2，1/2/4 例 | 3 / 600 | 训练 AB L1 为 1.000925–1.001471 | 三项 NO-GO，未优于零场 |
| B：DeepSets v2/v3，四例 | 2 / 400 | H L1 均约 0.9530；三维 L1 均约 1.0429 | 两项 NO-GO，逐例连续 H 及响应门槛失败 |
| B：GNOT v2/v3，四例 | 2 / 400 | H L1 为 0.6621/0.6422；三维 L1 为 0.8113/0.7851 | 两项 NO-GO，测量响应仍不足 |
| C：14 个模型状态、各两例 | 14 状态 / 0 | 编码器与潜在表示有响应，历史失败末态 sigmoid 深饱和 | 无训练 GO；fresh GNOT v1 有响应但重建误差极大 |

合计恰好 2000 次训练更新，未补步或扩大搜索。GNOT 合成两条件均有 3/4 案例三维误差优于零场，不能再笼统称为“无法表达非均匀场”；但几何捷径、共同模板及优化匹配问题尚未分离，200 次预算不能证明架构普遍不可学。全部 10 项均未通过完整条件学习门槛。

独立 NumPy 复核 2960 个指标字段、所有 AdamW 步数及初态身份；零更新补齐并独立重算全部 30 个 case 的敏感性条件。H 放大 1000 倍不改变训练 loss/参数梯度，网络只见 A/B。15 项 CPU 测试及 3 个 subtests 通过，冻结 Plain 的相同 train 输入 C0/C1 逐位回归通过。一次 C 输入梯度 OOM、一次模块导入错误和一次训练完成后的状态写入错误均有保留日志及有界修复，未丢弃或重放更新。历史权重/配置/结果未覆盖，无真实 GT 数组、GT128、val/test 数组或 Pool B 使用。

仅使用 GPU0 既有资源锁与双快照；GPU0 已释放，其他用户 GPU1–3 未干预。原正式队列仍暂停；无新正式 300 epoch、第二种子或 Senseiver 训练。唯一后续建议是另行授权“相同几何、不同解析羽流及物理合法测量交换”实验，本轮未执行。

[完整中文报告](deep_baselines_identifiability_v1_zh.md) · [十项配置/结果/checkpoint SHA](artifacts/deep_baselines_identifiability_v1/experiment_summary.csv) · [独立状态](artifacts/deep_baselines_identifiability_v1/final_status.json) · [交付复核](artifacts/deep_baselines_identifiability_v1/delivery_verification.json)。原始 checkpoint 和完整数组保留于 `/scratch/djc/GeoPhy-Set3D/deep_baselines_identifiability_v1/`，Git 仅同步代码及小体积证据。
