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
