# 多尺度 MRGR 参数优化 v2

预注册两个独立候选：学习率 2e-4→4e-4；consistency 权重 0.05→0.01。每种子新控制及候选均为同一 GPU2、相同初始权重、站位/打乱、166训练/36验证、固定20轮/1,660更新、300轮余弦前缀。每候选只改变一个变量。

原 consistency 是 A/B 三维场归一化 L1，可能与细节/测量拟合竞争；降低权重测试该竞争。双倍学习率测试短前缀是否受优化速度限制，但早期优势不代表300轮最终精度。没有PDE或三维GT监督。

## 种子 20261018

| 模型 | 普通 H相对L1 | P90 | 聚簇5均值 | 每轮s |
|---|---:|---:|---:|---:|
| control | 9.2070% | 16.3226% | 47.8787% | 33.05 |
| lr4e4 | 8.3480% | 13.1409% | 45.4275% | 33.29 |
| consistency001 | 8.8792% | 15.2476% | 49.0586% | 33.84 |

- lr4e4: 普通均值变化 -9.33%，P90 -19.49%，聚簇 -5.12%；门槛 通过；检查 `{"ordinary": true, "p90": true, "cluster": true, "runtime": true, "five_hour": true, "smoke": true}`。配对均值差95%区间 `[-0.012752642785040832, -0.00503398584220327]`（比例单位），胜/负 32/4。
- consistency001: 普通均值变化 -3.56%，P90 -6.59%，聚簇 +2.46%；门槛 失败；检查 `{"ordinary": true, "p90": true, "cluster": false, "runtime": true, "five_hour": true, "smoke": true}`。配对均值差95%区间 `[-0.006706158982273623, 4.209399985970785e-05]`（比例单位），胜/负 26/10。

[协议](../../outputs/research/mrgr_parameter_optimization_v2/seed20261018/protocol.json)；[结果](../../outputs/research/mrgr_parameter_optimization_v2/seed20261018/comparison.json)。

## 种子 20261019

| 模型 | 普通 H相对L1 | P90 | 聚簇5均值 | 每轮s |
|---|---:|---:|---:|---:|
| control | 11.2839% | 16.8051% | 48.9909% | 33.96 |
| lr4e4 | 9.7006% | 14.4037% | 45.3266% | 34.04 |

- lr4e4: 普通均值变化 -14.03%，P90 -14.29%，聚簇 -7.48%；门槛 通过；检查 `{"ordinary": true, "p90": true, "cluster": true, "runtime": true, "five_hour": true, "smoke": true}`。配对均值差95%区间 `[-0.021663438512764383, -0.010785358986735535]`（比例单位），胜/负 35/1。

[协议](../../outputs/research/mrgr_parameter_optimization_v2/seed20261019/protocol.json)；[结果](../../outputs/research/mrgr_parameter_optimization_v2/seed20261019/comparison.json)。

## 解释边界

所有数字仅是已反复使用开发验证集的投影误差，不是三维重建误差。没有读取三维GT或最终35例测试。病例bootstrap不包含训练数值轨迹与重复种子不确定性。正式训练由主agent综合决定，本脚本不启动300轮。真实分量loss和最大梯度范数逐轮记录；固定训练前2例的初始/末期全参数 consistency 梯度贡献记录在 smoke 和 late_gradient_probe。运行时含训练和验证，不含每轮checkpoint/曲线导出，外推可能受GPU负载影响。

第二种子中 `training.seed` 同时改变初始化与普通验证站位；每个种子内部严格配对。聚簇五布局跨种子固定。第二次结果是新初始化及新普通布局的匹配复核，不能解释为仅初始化重复，不将两次36例合并成72个独立病例。

原协议结论：LR4e-4 两次匹配筛查均通过，可交主agent开展正式训练验证；降低 consistency 到.01因首种子聚簇均值恶化2.464%超过2%上限，未晋级，不做第二种子。LR当前收益可解释为早期优化加快，不能据20轮预测300轮三维终态收益。

## 损失和梯度诊断

`standard_total` 始终按原 `.05` 一致性系数重组，实际 `total` 才使用处理组系数；不同系数的实际 total 不可直接比较优劣。下表梯度是每种子固定前两训练案例上的全参数梯度，末期不进行更新。

| 种子/组 | 第20轮一致性原.05梯度/原总梯度 | 实际系数梯度/原总梯度 | 一致性与原总梯度余弦 |
|---|---:|---:|---:|
| 20261018/control | 25.82% | 25.82% | 0.7174 |
| 20261018/lr4e4 | 41.99% | 41.99% | 0.7079 |
| 20261018/consistency001 | 34.41% | 6.88% | 0.5317 |
| 20261019/control | 25.13% | 25.13% | 0.6706 |
| 20261019/lr4e4 | 23.78% | 23.78% | 0.6035 |

初始梯度占比很小、末期变大反映测量项下降后目标构成发生变化；方向与总梯度同向不能证明和测量项无竞争，也不能据其大小直接判定该正则有害。固定两例探针并非全训练分布的平均结论。

![seed20261018 训练曲线](../../outputs/research/mrgr_parameter_optimization_v2/seed20261018/comparison_curves.png)

独立 CPU 审计：[seed20261018](../../outputs/research/mrgr_parameter_optimization_v2/seed20261018/independent_verification.json)，状态 已通过。

![seed20261019 训练曲线](../../outputs/research/mrgr_parameter_optimization_v2/seed20261019/comparison_curves.png)

独立 CPU 审计：[seed20261019](../../outputs/research/mrgr_parameter_optimization_v2/seed20261019/independent_verification.json)，状态 已通过。


## 投影相对L2补充指标

相对L2没有用于晋级筛选，以下仍是投影指标；不能将L1改善当作所有范数都同步改善。

| 种子/组 | 普通H投影相对L2均值 |
|---|---:|
| 20261018/control | 9.5052% |
| 20261018/lr4e4 | 9.6793% |
| 20261018/consistency001 | 9.1308% |
| 20261019/control | 11.4488% |
| 20261019/lr4e4 | 10.7146% |

第20轮逐轮平均诊断中，降低一致性系数的组，其训练 raw consistency .18693 对照 .15308（+22.1%），验证 .17670 对照 .14159（+24.8%）；实际 total 从 .03247 降到 .02664，但按原权重重组 standard_total 升到 .03412。因此较低实际 total 包含目标权衡的变化，不能直接当作整体拟合更佳的证据。LR组的一致性与控制接近。

## 统一普通布局补充审计

此项在第二种子LR结果产生前独立注册，四组固定epoch20权重哈希全部封存后，统一回放首种子普通A/B/H布局。只读测量、无GT、不影响原门槛或正式晋级。

| 训练种子 | 控制 H相对L1 | LR4e-4 | 相对变化 | 配对均值差95%CI（比例） |
|---|---:|---:|---:|---|
| 20261018 | 9.2070% | 8.3480% | -9.33% | [-0.012752642785040832, -0.00503398584220327] |
| 20261019 | 10.4429% | 9.0856% | -13.00% | [-0.01852364577419748, -0.008946412462046008] |

仅作初始化/训练变化在相同普通布局下的诊断，不用于挽救失败候选，不把144条记录视为144个独立病例。证据：[注册](../../outputs/research/mrgr_parameter_optimization_v2/common_layout_secondary/registration.json)、[完整输出](../../outputs/research/mrgr_parameter_optimization_v2/common_layout_secondary/summary.json)。

## 正式训练后续

主 agent 已按原两次筛查和独立核验封存晋级决定，新的 300 轮训练使用种子 20261011，保留原多尺度结构和原一致性系数，仅初始学习率改为 4e-4。训练不据早期曲线调整参数。

[正式晋级决定](../../outputs/research/mrgr_autonomous_optimization_v2/lr4e4_formal_decision.json)、[正式协议](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/protocol.json)、[相同初始化与数值配置审计](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/matched_initialization_audit.json)、[运行状态](../../outputs/research/mrgr_optimization_v2_lr4e4_seed20261011/status.json)。最终三维评价由主 agent 在训练完成后统一封存。
