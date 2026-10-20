# 均匀融合 MRGR：正式骨干消融补充评价

本阶段沿用已使用的35例、17布局，是补充配对评价，不是独立测试确认。GT128禁止读取。
主模型固定为MRGR-UniformFusion-v1，seed20261011、第285轮、342725参数，checkpoint SHA256：
`64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。消融结果不用于重选模型或轮次。
反馈OARI研究停止；仅保留历史封存指标，以维持原14项比较族，不新增其训练或推理。

## 范围、复用与协议

复用已核验的均匀融合seed20261011/20261019对照、描述符结构删除双种子、其他历史参照及已完成的seed20261020/20261021统计。
全面搜索scratch、历史研究目录和各工作树输出后，七个目标任务只有登记、没有正式checkpoint，因此本阶段补跑七项；短程工程检查和原GCA骨干消融均不替代正式任务。
每项300轮、24900更新，FP32/TF32关闭、batch2、AdamW lr2e-4/wd1e-4、cosine300、clip1。仅按完整训练期间验证投影H4严格最小值选轮；训练禁止GT64/GT128和测试投影。
全部七项训练完成、checkpoint冻结、7×595组新增C0/C1预测统一封存后，才读取本阶段授权GT64。
无反馈删除编码器/适配器并跳过反馈投影，减少12512参数；零残差保留名义参数但残差置零且跳过反馈投影，梯度/计算路径不同；不能把两者等同为纯容量匹配消融。
无射线关系只清零前三关系通道、保留描述符与维度；无一致性仅将权重0.05改为0；单尺度沿原登记空间拓扑，保持宽度/参数，感受野和计算不同；描述符删除为真实结构删除384参数。

## 逐种子完整结果

普通为固定Val-A-V4，聚簇先对每案例五draw平均，再对35案例汇总。L1/L2为百分比，定位为mm，其他量无量纲。

| 模型 | 布局 | C0 L1% | C1 L1% | C1 L2% | GT峰坐标比 | 峰比绝对误差 | 质心mm | Dice50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_no_feedback_seed20261011 | Val-A-V4 | 12.1526 | 12.1526 | 18.6457 | 0.9146 | 0.1957 | 2.5551 | 0.7981 |
| uniform_no_feedback_seed20261011 | clustered-V4 | 39.6169 | 39.6169 | 55.1454 | 0.1075 | 0.4073 | 17.7581 | 0.1894 |
| uniform_no_feedback_seed20261019 | Val-A-V4 | 12.5515 | 12.5515 | 18.6013 | 0.8972 | 0.1802 | 2.2745 | 0.7966 |
| uniform_no_feedback_seed20261019 | clustered-V4 | 38.9903 | 38.9903 | 54.2649 | 0.0907 | 0.4949 | 15.8928 | 0.1209 |
| uniform_zero_residual_seed20261011 | Val-A-V4 | 12.1546 | 11.9823 | 18.7740 | 0.9043 | 0.1893 | 2.3139 | 0.7984 |
| uniform_zero_residual_seed20261011 | clustered-V4 | 41.0576 | 37.3773 | 54.0834 | 0.1618 | 0.3171 | 12.7344 | 0.2565 |
| uniform_zero_residual_seed20261019 | Val-A-V4 | 12.7796 | 12.4813 | 18.5289 | 0.8972 | 0.2006 | 2.4224 | 0.7989 |
| uniform_zero_residual_seed20261019 | clustered-V4 | 39.8858 | 38.2061 | 53.7042 | 0.1328 | 0.4067 | 13.6363 | 0.2231 |
| uniform_no_ray_relations_seed20261011 | Val-A-V4 | 11.3255 | 8.7395 | 16.0114 | 1.0155 | 0.1984 | 1.6406 | 0.8232 |
| uniform_no_ray_relations_seed20261011 | clustered-V4 | 40.9789 | 30.6718 | 46.6314 | 0.2811 | 0.2180 | 10.6181 | 0.3672 |
| uniform_no_consistency_seed20261011 | Val-A-V4 | 11.4970 | 8.8173 | 16.8896 | 1.0395 | 0.2466 | 1.6928 | 0.8175 |
| uniform_no_consistency_seed20261011 | clustered-V4 | 40.9395 | 34.5926 | 50.4926 | 0.2100 | 0.3312 | 12.7078 | 0.2374 |
| uniform_single_scale_seed20261011 | Val-A-V4 | 11.1418 | 8.5331 | 15.8232 | 1.0092 | 0.1803 | 1.8306 | 0.8216 |
| uniform_single_scale_seed20261011 | clustered-V4 | 40.2132 | 31.9171 | 46.9544 | 0.3307 | 0.2428 | 10.6009 | 0.3863 |
| uniform_fusion_seed20261011 | Val-A-V4 | 11.3762 | 8.5928 | 15.7314 | 1.0076 | 0.2013 | 1.6174 | 0.8203 |
| uniform_fusion_seed20261011 | clustered-V4 | 33.0008 | 28.0246 | 45.0057 | 0.3322 | 0.1752 | 8.9787 | 0.4136 |
| uniform_fusion_seed20261019 | Val-A-V4 | 11.5614 | 8.5527 | 16.0457 | 0.9973 | 0.2030 | 1.6150 | 0.8166 |
| uniform_fusion_seed20261019 | clustered-V4 | 35.5517 | 29.9165 | 44.1267 | 0.3716 | 0.2145 | 10.1516 | 0.4104 |
| uniform_pruned_observability_seed20261011 | Val-A-V4 | 11.2388 | 8.7227 | 16.2261 | 1.0051 | 0.1929 | 1.6483 | 0.8140 |
| uniform_pruned_observability_seed20261011 | clustered-V4 | 36.4996 | 29.7149 | 46.0687 | 0.3414 | 0.1986 | 8.6973 | 0.4174 |
| uniform_pruned_observability_seed20261019 | Val-A-V4 | 11.3209 | 8.7217 | 15.9174 | 1.0433 | 0.2151 | 1.6672 | 0.8243 |
| uniform_pruned_observability_seed20261019 | clustered-V4 | 35.4943 | 28.9359 | 41.4762 | 0.4137 | 0.1945 | 8.6650 | 0.4850 |

## 配对差值与预登记校正

差值方向为“消融−同种子均匀融合”，L1正值表示完整模型误差更低。以案例为单位bootstrap20000次，种子20261020。
原14项Bonferroni分母保持不变。按此前测试前的种子缩减登记，无反馈、零残差双种子汇总补齐；无射线关系、无一致性和单尺度各仅seed20261011，95%区间属于单种子探索性证据，不替代原三个双种子主要比较。
原14项中现可计算11项，3项双种子比较仍不执行；这是原种子范围变更的规定，不新增种子、不缩小校正分母。案例区间不估计训练种子总体不确定性。

| 比较 | 布局 | L1差百分点 | 95%案例区间 | 原14项同时区间（仅主要终点） |
|---|---|---:|---|---|
| no_feedback-minus-uniform2seed | Val-A-V4 | 3.7793 | [+3.2462, +4.3476] | [+3.0021, +4.6290] |
| no_feedback-minus-uniform_seed20261011 | Val-A-V4 | 3.5598 | [+3.0503, +4.0964] | 探索性，未作主要终点校正 |
| no_feedback-minus-uniform_seed20261019 | Val-A-V4 | 3.9988 | [+3.4192, +4.6229] | 探索性，未作主要终点校正 |
| zero_residual-minus-uniform2seed | Val-A-V4 | 3.6590 | [+3.1313, +4.2120] | [+2.9001, +4.4926] |
| zero_residual-minus-uniform_seed20261011 | Val-A-V4 | 3.3895 | [+2.8594, +3.9614] | 探索性，未作主要终点校正 |
| zero_residual-minus-uniform_seed20261019 | Val-A-V4 | 3.9286 | [+3.3718, +4.5238] | 探索性，未作主要终点校正 |
| no_ray_relations-minus-uniform_seed20261011 | Val-A-V4 | 0.1467 | [-0.0855, +0.4535] | 探索性，未作主要终点校正 |
| no_consistency-minus-uniform_seed20261011 | Val-A-V4 | 0.2245 | [-0.0042, +0.4795] | 探索性，未作主要终点校正 |
| single_scale-minus-uniform_seed20261011 | Val-A-V4 | -0.0597 | [-0.1980, +0.0858] | 探索性，未作主要终点校正 |
| pruned_descriptor-minus-uniform2seed | Val-A-V4 | 0.1495 | [+0.0346, +0.2808] | [-0.0083, +0.3634] |
| pruned_descriptor-minus-uniform_seed20261011 | Val-A-V4 | 0.1299 | [+0.0046, +0.2673] | 探索性，未作主要终点校正 |
| pruned_descriptor-minus-uniform_seed20261019 | Val-A-V4 | 0.1691 | [-0.0061, +0.3664] | 探索性，未作主要终点校正 |
| no_feedback-minus-uniform2seed | clustered-V4 | 10.3331 | [+9.6203, +11.0204] | 探索性，未作主要终点校正 |
| no_feedback-minus-uniform_seed20261011 | clustered-V4 | 11.5923 | [+10.5105, +12.7363] | 探索性，未作主要终点校正 |
| no_feedback-minus-uniform_seed20261019 | clustered-V4 | 9.0739 | [+8.0296, +10.0908] | 探索性，未作主要终点校正 |
| zero_residual-minus-uniform2seed | clustered-V4 | 8.8211 | [+7.8136, +9.7075] | 探索性，未作主要终点校正 |
| zero_residual-minus-uniform_seed20261011 | clustered-V4 | 9.3526 | [+8.0822, +10.6495] | 探索性，未作主要终点校正 |
| zero_residual-minus-uniform_seed20261019 | clustered-V4 | 8.2896 | [+7.1501, +9.3491] | 探索性，未作主要终点校正 |
| no_ray_relations-minus-uniform_seed20261011 | clustered-V4 | 2.6472 | [+1.6811, +3.7045] | 探索性，未作主要终点校正 |
| no_consistency-minus-uniform_seed20261011 | clustered-V4 | 6.5680 | [+5.6957, +7.3696] | 探索性，未作主要终点校正 |
| single_scale-minus-uniform_seed20261011 | clustered-V4 | 3.8925 | [+3.2396, +4.5314] | 探索性，未作主要终点校正 |
| pruned_descriptor-minus-uniform2seed | clustered-V4 | 0.3548 | [-0.4659, +1.3219] | 探索性，未作主要终点校正 |
| pruned_descriptor-minus-uniform_seed20261011 | clustered-V4 | 1.6902 | [+0.5996, +2.9528] | 探索性，未作主要终点校正 |
| pruned_descriptor-minus-uniform_seed20261019 | clustered-V4 | -0.9806 | [-2.0194, +0.0569] | 探索性，未作主要终点校正 |

## 各因素的贡献与取舍

no_feedback：普通V4差+3.7793个百分点，完整模块误差较低；聚簇V4差+10.3331个百分点，方向一致。
Val-A-V4：峰比绝对误差差-0.0143、95%区间[-0.0386, +0.0088]；定位mm差0.7986、95%区间[+0.4305, +1.2295]；Dice50差-0.0211、95%区间[-0.0577, +0.0026]。
clustered-V4：峰比绝对误差差0.2563、95%区间[+0.2219, +0.2880]；定位mm差7.2603、95%区间[+5.1241, +9.3670]；Dice50差-0.2568、95%区间[-0.2885, -0.2224]。
逐种子普通V4差：20261011 +3.5598个百分点；20261019 +3.9988个百分点。两个固定种子的案例均值不能估计训练种子总体区间。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

zero_residual：普通V4差+3.6590个百分点，完整模块误差较低；聚簇V4差+8.8211个百分点，方向一致。
Val-A-V4：峰比绝对误差差-0.0072、95%区间[-0.0255, +0.0106]；定位mm差0.7519、95%区间[+0.4097, +1.1360]；Dice50差-0.0198、95%区间[-0.0374, -0.0050]。
clustered-V4：峰比绝对误差差0.1671、95%区间[+0.1388, +0.1951]；定位mm差3.6203、95%区间[+2.3537, +5.0886]；Dice50差-0.1722、95%区间[-0.1967, -0.1472]。
逐种子普通V4差：20261011 +3.3895个百分点；20261019 +3.9286个百分点。两个固定种子的案例均值不能估计训练种子总体区间。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

no_ray_relations：普通V4差+0.1467个百分点，区间跨零，主要终点收益证据不足；聚簇V4差+2.6472个百分点，方向一致。
Val-A-V4：峰比绝对误差差-0.0030、95%区间[-0.0208, +0.0153]；定位mm差0.0232、95%区间[-0.1906, +0.2131]；Dice50差0.0029、95%区间[-0.0066, +0.0118]。
clustered-V4：峰比绝对误差差0.0428、95%区间[+0.0145, +0.0708]；定位mm差1.6394、95%区间[-0.2802, +3.7003]；Dice50差-0.0464、95%区间[-0.0719, -0.0206]。
逐种子普通V4差：20261011 +0.1467个百分点。仅单种子，不能推断跨种子稳定性。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

no_consistency：普通V4差+0.2245个百分点，区间跨零，主要终点收益证据不足；聚簇V4差+6.5680个百分点，方向一致。
Val-A-V4：峰比绝对误差差0.0453、95%区间[+0.0241, +0.0673]；定位mm差0.0754、95%区间[-0.0895, +0.2491]；Dice50差-0.0029、95%区间[-0.0160, +0.0086]。
clustered-V4：峰比绝对误差差0.1561、95%区间[+0.0965, +0.2154]；定位mm差3.7291、95%区间[+1.8960, +5.6669]；Dice50差-0.1761、95%区间[-0.2118, -0.1402]。
逐种子普通V4差：20261011 +0.2245个百分点。仅单种子，不能推断跨种子稳定性。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

single_scale：普通V4差-0.0597个百分点，区间跨零，主要终点收益证据不足；聚簇V4差+3.8925个百分点，方向不一致，存在布局依赖。
Val-A-V4：峰比绝对误差差-0.0211、95%区间[-0.0379, -0.0036]；定位mm差0.2132、95%区间[+0.0706, +0.3583]；Dice50差0.0013、95%区间[-0.0124, +0.0135]。
clustered-V4：峰比绝对误差差0.0676、95%区间[+0.0293, +0.1086]；定位mm差1.6222、95%区间[+0.1663, +2.9377]；Dice50差-0.0273、95%区间[-0.0741, +0.0173]。
逐种子普通V4差：20261011 -0.0597个百分点。仅单种子，不能推断跨种子稳定性。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

pruned_descriptor：普通V4差+0.1495个百分点，区间跨零，主要终点收益证据不足；聚簇V4差+0.3548个百分点，方向一致。
Val-A-V4：峰比绝对误差差0.0019、95%区间[-0.0129, +0.0179]；定位mm差0.0415、95%区间[-0.1088, +0.2041]；Dice50差0.0007、95%区间[-0.0102, +0.0096]。
clustered-V4：峰比绝对误差差0.0018、95%区间[-0.0216, +0.0262]；定位mm差-0.8840、95%区间[-2.0834, +0.2759]；Dice50差0.0392、95%区间[+0.0144, +0.0646]。
逐种子普通V4差：20261011 +0.1299个百分点；20261019 +0.1691个百分点。两个固定种子的案例均值不能估计训练种子总体区间。
机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。

## 状态、成本与完整性

| 新任务 | 轮次/更新 | 选轮 | 参数 | checkpoint SHA256 |
|---|---|---:|---:|---|
| uniform_no_feedback_seed20261011 | 300 / 24900 | 293 | 330213 | 6d5bba73fb05c773d8a96506a6af6f1626568fc1f9717de837dcb106763a0575 |
| uniform_no_feedback_seed20261019 | 300 / 24900 | 288 | 330213 | e255b35ea161a9bc417e5e6f60a7b816f8c7b20b6b38e67f2f393cb246f9fefd |
| uniform_zero_residual_seed20261011 | 300 / 24900 | 290 | 342725 | 2e3b8ee23b1cdd0e2882078c1363b8cc423ec0044df37ea7e185754a3cccdd25 |
| uniform_zero_residual_seed20261019 | 300 / 24900 | 287 | 342725 | 371a1510e2f807856aa4383819c8bdbd82cd084bf7c9c77426c75dcd7165eeb0 |
| uniform_no_ray_relations_seed20261011 | 300 / 24900 | 283 | 342725 | f6afaa13bdcb851ee5e7ebee6086116be482157a3a12546c1164a22ff6e207f4 |
| uniform_no_consistency_seed20261011 | 300 / 24900 | 296 | 342725 | f5eb47fa75cdf7966bdc070ad1daa28745607ebf7af07cc99c0608c87436ba11 |
| uniform_single_scale_seed20261011 | 300 / 24900 | 298 | 342725 | c0cd20470043f013394e6bd0a34b317c7d96c054e012a0e59292fc8c8f0f2958 |

同一GPU、同一开发案例、batch1/V4/FP32，3次预热、10次同步C0+C1前向；不包含载入/I/O，不作跨设备训练速度排名。

| 方法 | 参数 | 中位前向ms | 峰值已分配MiB |
|---|---:|---:|---:|
| MRGR | 344678 | 23.918 | 1103.41 |
| uniform_fusion_seed20261011 | 342725 | 22.337 | 1103.15 |
| uniform_pruned_observability_seed20261011 | 342341 | 20.222 | 272.31 |
| uniform_no_feedback_seed20261011 | 330213 | 13.028 | 1103.10 |
| uniform_zero_residual_seed20261011 | 342725 | 20.458 | 1103.15 |
| uniform_no_ray_relations_seed20261011 | 342725 | 22.379 | 1103.15 |
| uniform_no_consistency_seed20261011 | 342725 | 22.421 | 1103.15 |
| uniform_single_scale_seed20261011 | 342725 | 22.702 | 1103.15 |

数值失败、零输出、预算耗尽和高误差案例全部保留。预定L1≥50%和≥100%阈值只用于描述高误差率，不排除案例。

| 模型 | C1重建数 | L1≥50% | L1≥100% | 数值失败 |
|---|---:|---:|---:|---:|
| uniform_no_feedback_seed20261011 | 595 | 42 | 0 | 0 |
| uniform_no_feedback_seed20261019 | 595 | 34 | 0 | 0 |
| uniform_zero_residual_seed20261011 | 595 | 30 | 0 | 0 |
| uniform_zero_residual_seed20261019 | 595 | 37 | 0 | 0 |
| uniform_no_ray_relations_seed20261011 | 595 | 10 | 0 | 0 |
| uniform_no_consistency_seed20261011 | 595 | 20 | 0 | 0 |
| uniform_single_scale_seed20261011 | 595 | 13 | 0 | 0 |
| MRGR | 595 | 18 | 0 | 0 |
| LR4E4 | 595 | 13 | 0 | 0 |
| multiscale_seed20261019 | 595 | 32 | 0 | 0 |
| uniform_fusion_seed20261011 | 595 | 11 | 0 | 0 |
| uniform_fusion_seed20261019 | 595 | 14 | 0 | 0 |
| NNLS-NL | 595 | 357 | 198 | 0 |
| Tikhonov | 595 | 286 | 168 | 0 |
| TV | 595 | 354 | 199 | 0 |
| SART | 595 | 286 | 188 | 0 |
| uniform_pruned_observability_seed20261011 | 595 | 10 | 0 | 0 |
| uniform_pruned_observability_seed20261019 | 595 | 12 | 0 | 0 |
| uniform_seed20261020 | 595 | 12 | 0 | 0 |
| uniform_feedback_oari_seed20261020 | 595 | 20 | 0 | 0 |
| uniform_seed20261021 | 595 | 14 | 0 | 0 |
| uniform_feedback_oari_seed20261021 | 595 | 24 | 0 | 0 |

| 任务 | GPU UUID | PID | 开始（北京时间） | 结束 | 状态 |
|---|---|---:|---|---|---|
| benchmark | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3921401 | 2026-10-21T01:16:45.461075+08:00 | 2026-10-21T01:17:00.710150+08:00 | complete |
| predict_uniform_no_consistency_seed20261011 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3924425 | 2026-10-21T01:24:55.969144+08:00 | 2026-10-21T01:26:11.911254+08:00 | complete |
| predict_uniform_no_feedback_seed20261011 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3921625 | 2026-10-21T01:17:18.150216+08:00 | 2026-10-21T01:18:34.125998+08:00 | complete |
| predict_uniform_no_feedback_seed20261019 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3922181 | 2026-10-21T01:18:49.750301+08:00 | 2026-10-21T01:20:05.791303+08:00 | complete |
| predict_uniform_no_ray_relations_seed20261011 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3923863 | 2026-10-21T01:23:24.415958+08:00 | 2026-10-21T01:24:40.357014+08:00 | complete |
| predict_uniform_single_scale_seed20261011 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3924982 | 2026-10-21T01:26:27.477721+08:00 | 2026-10-21T01:27:43.456866+08:00 | complete |
| predict_uniform_zero_residual_seed20261011 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3922740 | 2026-10-21T01:20:21.343516+08:00 | 2026-10-21T01:21:37.281165+08:00 | complete |
| predict_uniform_zero_residual_seed20261019 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3923292 | 2026-10-21T01:21:52.883454+08:00 | 2026-10-21T01:23:08.826265+08:00 | complete |
| seal_score | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3925543 | 2026-10-21T01:27:59.051979+08:00 | 2026-10-21T01:33:02.905186+08:00 | complete |
| uniform_no_consistency_seed20261011_full300 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3851763 | 2026-10-20T22:15:09.472348+08:00 | 2026-10-21T00:59:00.509805+08:00 | complete |
| uniform_no_feedback_seed20261011_full300 | GPU-37424437-e208-62d1-e617-53c421c94a49 | 3757244 | 2026-10-20T20:23:12.257533+08:00 | 2026-10-20T22:14:53.729932+08:00 | complete |
| uniform_no_feedback_seed20261019_full300 | GPU-d3da1ea7-9229-8f8a-5054-854962d4b6b7 | 3757258 | 2026-10-20T20:23:12.492997+08:00 | 2026-10-20T22:13:21.943705+08:00 | complete |
| uniform_no_ray_relations_seed20261011_full300 | GPU-d3da1ea7-9229-8f8a-5054-854962d4b6b7 | 3850988 | 2026-10-20T22:13:37.670050+08:00 | 2026-10-21T00:52:39.930821+08:00 | complete |
| uniform_single_scale_seed20261011_full300 | GPU-fdd41417-bc2c-971b-b84d-6464cf7c5749 | 3855880 | 2026-10-20T22:24:49.125368+08:00 | 2026-10-21T01:16:29.583591+08:00 | complete |
| uniform_zero_residual_seed20261011_full300 | GPU-fdd41417-bc2c-971b-b84d-6464cf7c5749 | 3758873 | 2026-10-20T20:24:29.227358+08:00 | 2026-10-20T22:24:33.414060+08:00 | complete |
| uniform_zero_residual_seed20261019_full300 | GPU-cd80a78f-7d90-c291-4680-4895db9aac8b | 3782303 | 2026-10-20T20:45:51.279626+08:00 | 2026-10-20T22:43:22.216037+08:00 | complete |

[复用审计](artifacts/mrgr_uniform_ablations_v1/reuse_audit.json)；[阶段登记](artifacts/mrgr_uniform_ablations_v1/stage_registration.json)；[训练审计](artifacts/mrgr_uniform_ablations_v1/training_audit.json)；[全案例指标](artifacts/mrgr_uniform_ablations_v1/per_case_metrics.csv)；[完整配对统计](artifacts/mrgr_uniform_ablations_v1/paired_statistics.jsonl.gz)；[最差案例](artifacts/mrgr_uniform_ablations_v1/worst_cases.csv)；[独立重算](artifacts/mrgr_uniform_ablations_v1/independent_verification.json)；[成本](artifacts/mrgr_uniform_ablations_v1/benchmark.json)；[资源](artifacts/mrgr_uniform_ablations_v1/resource_tasks.json)。

原始大文件、checkpoint、预测及资源快照保留在 `/scratch/djc/GeoPhy-Set3D/mrgr_uniform_ablations_v1`。可复现命令：
```bash
/scratch/djc/envs/gas-torch/bin/python /home/djc/projects/GeoPhy-Set3D/scripts/uniform_ablations_v1/stage.py controller
```
已有接收产物禁止覆盖；恢复前检查失败记录、任务锁、完整模型/优化器/调度器/RNG/步数。

所有次要指标、C0/C1、17布局逐draw和案例内五draw均值均随CSV/原始指标导出。单种子有利结果或个别次要指标不能支持模块普遍有效的论文表述。
