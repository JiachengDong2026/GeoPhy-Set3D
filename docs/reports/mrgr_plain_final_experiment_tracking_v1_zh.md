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
