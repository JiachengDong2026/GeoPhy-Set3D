# 新增反馈OARI、均匀融合与描述符结构删除：补充35例测试

完成统计：2026-10-20T13:05:16.920533+08:00；所有新增模型保持开发阶段冻结的最佳验证H4检查点，未重新训练、选轮或调参。

**本轮为用户明确要求的补充测试。35例与此前正式测试相同，已使用过；本结果不能称作全新独立保留集确认。既有35例主报告与主模型不修改。**

7个新模型各35例×17布局＝4165次重建、8330个C0/C1场，全部预测封存后才读取本轮test GT64评分；另核验复用原始MRGR两种子和均匀融合两种子的既有35例预测。总计11个模型、13090个场独立重算。GT128始终禁止。

均匀融合两种子的35例测试早已包含在原正式测试；此处复用其完全相同的检查点、595条预测清单及文件SHA，不重复执行。新增的是反馈OARI及其匹配baseline/零残差对照、结构删除两个种子和双重简化两个种子。

## C1三维重建结果

聚簇V4先将5个draw在案例内平均，再跨35例汇总。下表逐种子，不能把不同seed的模型做不配对排名。

| 模型 | 种子 | 普通V4 L1 | 聚簇V4 L1 | 聚簇V4 L2 | 聚簇质心mm | 聚簇峰比绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| pruned_observability | 20261011 | 8.4965% | 33.5305% | 48.6068% | 11.2262 | 0.2077 |
| uniform_pruned_observability | 20261011 | 8.7227% | 29.7149% | 46.0687% | 8.6973 | 0.1986 |
| pruned_observability | 20261019 | 8.9012% | 33.1103% | 47.8382% | 12.1714 | 0.2604 |
| uniform_pruned_observability | 20261019 | 8.7217% | 28.9359% | 41.4762% | 8.6650 | 0.1945 |
| feedback_oari | 20261011 | 8.3140% | 29.7930% | 44.4351% | 9.3615 | 0.2265 |
| baseline | 20261011 | 8.5089% | 31.3236% | 48.2761% | 11.1656 | 0.2038 |
| feedback_oari_zero_residual | 20261011 | 11.7767% | 35.4046% | 51.1844% | 13.1245 | 0.2869 |
| MRGR | 20261011 | 8.5159% | 31.6583% | 47.9077% | 10.6727 | 0.2000 |
| multiscale | 20261019 | 8.6411% | 35.7544% | 50.2474% | 13.3968 | 0.2161 |
| uniform_fusion | 20261011 | 8.5928% | 28.0246% | 45.0057% | 8.9787 | 0.1752 |
| uniform_fusion | 20261019 | 8.5527% | 29.9165% | 44.1267% | 10.1516 | 0.2145 |

## 配对差值

比较栏按所写表达式计算，L1负值表示下降；95%区间为未做多重校正的探索性案例bootstrap20000次（seed20261020）。两种子均值只作描述，区间不覆盖训练种子总体方差。

| 比较 | 种子 | 布局 | L1差百分点 | 95%案例区间 | 降/平/升 |
|---|---|---|---:|---|---:|
| feedback-baseline | 20261011 | Val-A-V4 | -0.1949 | [-0.3816, -0.0048] | 25/0/10 |
| feedback-zero_residual | 20261011 | Val-A-V4 | -3.4626 | [-4.0358, -2.9413] | 35/0/0 |
| B-A | two_seed_case_mean | Val-A-V4 | -0.0058 | [-0.1635, +0.1363] | 14/0/21 |
| C-A | two_seed_case_mean | Val-A-V4 | +0.1204 | [-0.0325, +0.2803] | 13/0/22 |
| D-B | two_seed_case_mean | Val-A-V4 | +0.1495 | [+0.0346, +0.2808] | 14/0/21 |
| D-C | two_seed_case_mean | Val-A-V4 | +0.0234 | [-0.0931, +0.1570] | 16/0/19 |
| (D-C)-(B-A) | two_seed_case_mean | Val-A-V4 | +0.0291 | [-0.1320, +0.2113] | 19/0/16 |
| feedback-baseline | 20261011 | clustered-V4 | -1.5306 | [-2.6892, -0.1897] | 27/0/8 |
| feedback-zero_residual | 20261011 | clustered-V4 | -5.6116 | [-7.1100, -4.1177] | 32/0/3 |
| B-A | two_seed_case_mean | clustered-V4 | -4.7358 | [-5.5307, -3.9563] | 35/0/0 |
| C-A | two_seed_case_mean | clustered-V4 | -0.3860 | [-1.0258, +0.2618] | 19/0/16 |
| D-B | two_seed_case_mean | clustered-V4 | +0.3548 | [-0.4659, +1.3219] | 18/0/17 |
| D-C | two_seed_case_mean | clustered-V4 | -3.9950 | [-5.2056, -2.7613] | 33/0/2 |
| (D-C)-(B-A) | two_seed_case_mean | clustered-V4 | +0.7408 | [-0.3449, +2.0070] | 15/0/20 |

## 解读与边界

Val-A-V4：反馈OARI相对本轮匹配baseline的三维L1变化-0.1949个百分点，峰比绝对误差变化+0.0054，质心误差变化-0.0138 mm。

clustered-V4：反馈OARI相对本轮匹配baseline的三维L1变化-1.5306个百分点，峰比绝对误差变化+0.0227，质心误差变化-1.8041 mm。

OARI只有一个训练种子；描述符删除和双重简化各两个种子。保留全部正负结果，不以本次测试选择新主模型。历史冻结MRGR与本轮续训匹配baseline不同，明确分开列出。计时来自不同GPU及运行负载，不能据此作严格速度排名。

[完整统计](artifacts/mrgr_extension_test35_v1/summary.json)；[逐案例CSV](artifacts/mrgr_extension_test35_v1/per_case_metrics.csv)；[独立复核](artifacts/mrgr_extension_test35_v1/independent_verification.json)；[冻结清单](artifacts/mrgr_extension_test35_v1/final_lock.json)；[访问边界](artifacts/mrgr_extension_test35_v1/boundary_checks.json)。

![预登记固定案例0：共享asinh色标](artifacts/mrgr_extension_test35_v1/fixed_case_0.png)

![预登记固定案例17：共享asinh色标](artifacts/mrgr_extension_test35_v1/fixed_case_17.png)

![预登记固定案例34：共享asinh色标](artifacts/mrgr_extension_test35_v1/fixed_case_34.png)

图固定显示seed20261011及原始冻结MRGR；切面在GT峰值y处，不按模型效果挑图。聚簇图仅draw0，数值表使用全部5draw。共享asinh色标增强低浓度可见性，超过GT峰值部分饱和。

运行产物：`/scratch/djc/GeoPhy-Set3D/mrgr_extension_test35_v1`；代码：`scripts/mrgr_extension_test35_v1/`。大数组/权重仅留scratch，仓库保留小型标量、CSV、图表和验收证据。
