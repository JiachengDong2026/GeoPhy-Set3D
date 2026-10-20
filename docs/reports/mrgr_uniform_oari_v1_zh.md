# 均匀融合＋反馈OARI：独立研究

Control为均匀融合MRGR；Treatment只在残差适配器中间加入独立OARI，描述符、损失、多尺度与数据流不变。新增10980参数，总参数353705。两个新种子各先20轮，以验证投影H4及工程/成本门槛判断，再按原cosine300、完整断点状态继续到300轮；最终仅按全部300轮验证H4选轮。未使用测试GT晋级。

本轮20轮门槛只排除明显工程失败：H4不超过匹配Control的1.10倍、成本不超过2倍、峰值显存不超过20GiB，完整检查通过。它不要求短程三维指标改善，也不沿用历史GCA支线的三维筛选条件。两个独立研究任务的规则与授权边界分别保留。

| 模型 | 普通V4 L1 | 聚簇V4 L1 | 聚簇L2 | 聚簇质心mm | 聚簇GT峰处比 | 聚簇峰比绝对误差 | 聚簇Dice50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_seed20261020 | 8.5957% | 30.5898% | 46.9908% | 9.6076 | 0.3292 | 0.2164 | 0.3728 |
| uniform_feedback_oari_seed20261020 | 8.6214% | 31.3760% | 46.8076% | 11.8589 | 0.2870 | 0.2707 | 0.3659 |
| uniform_seed20261021 | 8.8406% | 33.1448% | 49.2777% | 10.9030 | 0.2880 | 0.2292 | 0.3476 |
| uniform_feedback_oari_seed20261021 | 8.7756% | 34.5743% | 51.3018% | 14.4250 | 0.1519 | 0.3481 | 0.2552 |

## C0到C1的误差及峰值轨迹

C0/C1使用同一训练模型的共享解码器，属于阶段诊断。峰比绝对误差为abs(max(C)/max(GT)−1)，与GT峰坐标处的浓度比不同。

| 模型 | 布局 | C0 L1 | C1 L1 | C0峰比绝对误差 | C1峰比绝对误差 |
|---|---|---:|---:|---:|---:|
| uniform_seed20261020 | Val-A-V4 | 11.3132% | 8.5957% | 0.180483 | 0.195839 |
| uniform_seed20261020 | clustered-V4 | 35.5304% | 30.5898% | 0.246392 | 0.216357 |
| uniform_feedback_oari_seed20261020 | Val-A-V4 | 11.4967% | 8.6214% | 0.201366 | 0.224265 |
| uniform_feedback_oari_seed20261020 | clustered-V4 | 41.5781% | 31.3760% | 0.274851 | 0.270739 |
| uniform_seed20261021 | Val-A-V4 | 11.7628% | 8.8406% | 0.201296 | 0.202823 |
| uniform_seed20261021 | clustered-V4 | 39.8036% | 33.1448% | 0.270600 | 0.229201 |
| uniform_feedback_oari_seed20261021 | Val-A-V4 | 11.8931% | 8.7756% | 0.201608 | 0.209123 |
| uniform_feedback_oari_seed20261021 | clustered-V4 | 44.6549% | 34.5743% | 0.366237 | 0.348138 |

| 比较 | 布局 | 指标 | 差值 | 未校正95%案例区间 |
|---|---|---|---:|---|
| feedbackOARI2seed-minus-matchedUniform2seed | Val-A-V4 | volume.relative_l1 | -0.000196 | [-0.001513, +0.001018] |
| feedbackOARI2seed-minus-matchedUniform2seed | Val-A-V4 | volume.relative_l2 | +0.000523 | [-0.001604, +0.002289] |
| feedbackOARI2seed-minus-matchedUniform2seed | Val-A-V4 | morphology.absolute_peak_ratio_error | +0.017363 | [+0.008475, +0.025815] |
| feedbackOARI2seed-minus-matchedUniform2seed | Val-A-V4 | morphology.centroid_distance_mm | -0.145209 | [-0.274333, -0.029489] |
| feedbackOARI2seed-minus-matchedUniform2seed | Val-A-V4 | morphology.dice_0.5 | -0.000713 | [-0.009345, +0.006942] |
| feedbackOARI-minus-uniform_seed20261020 | Val-A-V4 | volume.relative_l1 | +0.000258 | [-0.001304, +0.001741] |
| feedbackOARI-minus-uniform_seed20261020 | Val-A-V4 | volume.relative_l2 | +0.000066 | [-0.004266, +0.004678] |
| feedbackOARI-minus-uniform_seed20261020 | Val-A-V4 | morphology.absolute_peak_ratio_error | +0.028426 | [+0.012563, +0.043455] |
| feedbackOARI-minus-uniform_seed20261020 | Val-A-V4 | morphology.centroid_distance_mm | -0.066300 | [-0.263436, +0.108305] |
| feedbackOARI-minus-uniform_seed20261020 | Val-A-V4 | morphology.dice_0.5 | +0.008981 | [-0.010302, +0.024436] |
| feedbackOARI-minus-uniform_seed20261021 | Val-A-V4 | volume.relative_l1 | -0.000650 | [-0.002374, +0.000991] |
| feedbackOARI-minus-uniform_seed20261021 | Val-A-V4 | volume.relative_l2 | +0.000980 | [-0.002034, +0.003743] |
| feedbackOARI-minus-uniform_seed20261021 | Val-A-V4 | morphology.absolute_peak_ratio_error | +0.006300 | [-0.008300, +0.020125] |
| feedbackOARI-minus-uniform_seed20261021 | Val-A-V4 | morphology.centroid_distance_mm | -0.224119 | [-0.398949, -0.072350] |
| feedbackOARI-minus-uniform_seed20261021 | Val-A-V4 | morphology.dice_0.5 | -0.010408 | [-0.023772, +0.000547] |
| feedbackOARI2seed-minus-matchedUniform2seed | clustered-V4 | volume.relative_l1 | +0.011079 | [+0.005249, +0.017329] |
| feedbackOARI2seed-minus-matchedUniform2seed | clustered-V4 | volume.relative_l2 | +0.009204 | [+0.001547, +0.017325] |
| feedbackOARI2seed-minus-matchedUniform2seed | clustered-V4 | morphology.absolute_peak_ratio_error | +0.086660 | [+0.049957, +0.127194] |
| feedbackOARI2seed-minus-matchedUniform2seed | clustered-V4 | morphology.centroid_distance_mm | +2.886649 | [+1.497389, +4.611756] |
| feedbackOARI2seed-minus-matchedUniform2seed | clustered-V4 | morphology.dice_0.5 | -0.049685 | [-0.071151, -0.029215] |
| feedbackOARI-minus-uniform_seed20261020 | clustered-V4 | volume.relative_l1 | +0.007862 | [+0.001390, +0.014964] |
| feedbackOARI-minus-uniform_seed20261020 | clustered-V4 | volume.relative_l2 | -0.001833 | [-0.012989, +0.009160] |
| feedbackOARI-minus-uniform_seed20261020 | clustered-V4 | morphology.absolute_peak_ratio_error | +0.054383 | [+0.009650, +0.096844] |
| feedbackOARI-minus-uniform_seed20261020 | clustered-V4 | morphology.centroid_distance_mm | +2.251329 | [+0.713565, +3.890659] |
| feedbackOARI-minus-uniform_seed20261020 | clustered-V4 | morphology.dice_0.5 | -0.006911 | [-0.041182, +0.026982] |
| feedbackOARI-minus-uniform_seed20261021 | clustered-V4 | volume.relative_l1 | +0.014296 | [+0.005687, +0.023641] |
| feedbackOARI-minus-uniform_seed20261021 | clustered-V4 | volume.relative_l2 | +0.020241 | [+0.009180, +0.031689] |
| feedbackOARI-minus-uniform_seed20261021 | clustered-V4 | morphology.absolute_peak_ratio_error | +0.118937 | [+0.067880, +0.176394] |
| feedbackOARI-minus-uniform_seed20261021 | clustered-V4 | morphology.centroid_distance_mm | +3.521969 | [+0.835908, +6.426080] |
| feedbackOARI-minus-uniform_seed20261021 | clustered-V4 | morphology.dice_0.5 | -0.092460 | [-0.127254, -0.058897] |

表中L1/L2差值使用0–1比例，质心为mm，其他指标为无量纲；完整主要终点同时区间见总报告。

[筛查门槛与逐种子结果](artifacts/mrgr_uniform_final_v1/training_complete.json)；[训练身份](artifacts/mrgr_uniform_final_v1/training_audit.json)；[同设备推理成本](artifacts/mrgr_uniform_final_v1/benchmark.json)。

两个种子、两个关键布局的L1方向不完全一致，当前收益证据混合；不据此扩大无约束搜索或晋级主模型，未来验证需预登记新独立案例。聚簇峰比绝对误差的两种子平均差为+0.086660，存在峰值恢复退化取舍。

没有额外独立案例，两个训练种子和案例区间仍不足以证明训练随机性下普遍稳定优势；新增容量的非注意力匹配对照仍缺失。所有不利指标保留。

本阶段按用户要求，先完成两种子均匀融合与两种子反馈OARI的三维配对评价；7个尚未启动的骨干消融暂缓，3个第二种子此前取消。暂缓任务不自动恢复，反馈OARI是否作为最终模型等待用户审阅，本报告不作自动晋级。
