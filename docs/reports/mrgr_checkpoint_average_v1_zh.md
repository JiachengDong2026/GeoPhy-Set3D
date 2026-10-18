# 多尺度 MRGR 后期 checkpoint 固定平均：测量验证

本实验是 **posthoc checkpoint averaging**，没有实施 SWA 优化过程或 EMA。冻结正式多尺度模型的 epoch 250、275、300 等权平均；窗口提前固定，未搜索或根据结果挑选。对照为原训练选出的 best epoch 278。

仅使用训练池的 36 个验证案例投影。普通指标逐案例平均 A→H、B→H；聚簇沿用冻结的 5 组布局，先案例内平均再对 36 案例等权平均。以下不是三维浓度场误差，最终 35 条测试集没有使用。

| 留出投影相对 L1 | best278 | 固定平均 | 相对变化 |
|---|---:|---:|---:|
| 普通均值 | 3.109370% | 3.124849% | +0.4978% |
| 聚簇五布局均值 | 16.739194% | 16.602809% | -0.8148% |
| 普通 P90 | 5.091837% | 5.113502% | +0.4255% |

晋级结果：**未通过**。预注册条件：普通均值至少改善 1%，普通 P90 和聚簇五布局均值均不得恶化超过 2%。检查：`{"ordinary_mean": false, "ordinary_p90": true, "cluster5_mean": true}`。没有自动启动训练或三维 GT 评价。

| 配对案例统计（候选－对照） | 均值差（百分点） | 95% bootstrap 区间（百分点） | 改善/退化/持平 |
|---|---:|---:|---:|
| 普通 | +0.015479 | [-0.002249, +0.040366] | 16/20/0 |
| 聚簇五布局 | -0.136385 | [-0.280813, +0.002222] | 21/15/0 |

bootstrap 按 36 个独立案例成对重采样，20,000 次、固定种子 20261017。聚簇的 180 次重建不作为 180 个独立样本。区间仅反映此开发验证面板，不代表多种子稳定性。

所有命名浮点参数以 float64 累加再转换回原 dtype；非浮点与 buffer 要求完全一致后保留，不平均几何 buffer。每个源 checkpoint 的单模型平均均通过逐张量 bitwise identity 检查；形状、类型、有限值检查通过。

实测总耗时 **14.36 秒**；评估 best278 为 6.95 秒，平均候选为 6.40 秒。GPU1，FP32/TF32 off；没有训练更新，部署仍为单模型，参数量 344,678。

GTReadGuard 审计：`{"gt_file_opened": false, "gt_open_attempts": 0, "forbidden_npy_attempts": 0, "supervised_checkpoint_open_attempts": 0, "projection_open_count": 1296, "unique_pool_a_projection_files": 36, "policy": "Only Pool A train/val projection NPY; GT64/GT128 and foreign checkpoints denied", "audit_hook_inherited_by_workers": true}`。原源文件及权重 hash 事后复核一致。

证据：

- [注册协议](../../outputs/research/mrgr_checkpoint_average_v1/protocol.json)
- [完整汇总](../../outputs/research/mrgr_checkpoint_average_v1/summary.json)
- [逐案例原始指标](../../outputs/research/mrgr_checkpoint_average_v1/per_case_metrics.jsonl)

原始列：`arm, panel, case_id, input_ids, heldout_ids, relative_l1`；普通还记录 `A/B.relative_l1/relative_l2`，聚簇记录 `draw, relative_l2`。

| 源文件 | SHA256 |
|---|---|
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/protocol.json` | `461aa845efe1c37d0c2d5e90786e5587ead8a2d2d236f9f9797158be708c568c` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/summary.json` | `c711323497b38de3acc80d565e7c281531dfe761b1d7a14df8951f220850e72a` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json` | `78f29155fd1b86887aa9b90a36247bd741e3036f1d6eb38123797bcfce18d440` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_v1/station_epoch0.json` | `4c9b7ce5f860dd4d2d8006d68fc33e3e2c59f48306acd7f241621b3d41c8c69a` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/epoch_250.pt` | `575bdcbe6b360f8bdb0ba7a9c4e3609b8808a608683e2d4b382a499bd2ce3d9a` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/epoch_275.pt` | `57dd83d91067a06d8089828244a0d1486a6cafe3653b1c5261d8710f7d78f80d` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/epoch_300.pt` | `aad8c127ffc5a810ca8adb549b79d662f35e8d23dec2dc98c49d2a71d3c98675` |
| `/home/djc/projects/GeoPhy-Set3D/outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/best.pt` | `650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7` |

平均权重 SHA256：`f1936489835da5f5c28bc5132f292153068c5a61c86fccc249716481ded0faa1`。
注册 SHA256：`60eace06b269a4709f7cb2e508cf9ff18c800019a1ab9a007265b5ae005ab87f`；原始指标 SHA256：`99b3e47ec02b7ea730e94164a9d2e496f65bde3e42c4ffe4c802da5369e09430`。

独立复核：原 best278 普通 H 为 0.031093697580039333，本次重算为 0.031093697580039326，绝对差 6.94e-18；432 行键唯一、逐案例聚合与普通 A/B 均值复核通过。详情见 [independent_verification.json](../../outputs/research/mrgr_checkpoint_average_v1/independent_verification.json)。

解释：固定平均没有改善普通均值（反而增加约 0.50%），聚簇均值小幅下降约 0.81%，两个均值差的配对区间都跨零。聚簇 P90 为 24.077055% → 24.565767%（增加约 2.03%，此项为诊断而非预注册门槛）。本窗口没有替换原 best278 的证据，不能由此否定在训练中使用 EMA 的全部可能性。

几何实现核查：`lifting.voxel_xyz` 是 `persistent=False` 的 buffer（32³×3），由冻结模型构造器重新生成，不在 checkpoint 的 state_dict 内，因此本次 state_dict 中非参数 buffer 列表为空；该几何网格从未参与权重平均。
