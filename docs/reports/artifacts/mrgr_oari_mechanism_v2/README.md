# OARI mechanism v2 evidence

阅读顺序：

1. `mrgr_oari_mechanism_v2_final_zh.md`：最终中文报告，包含完整逐布局机制解释。
2. `acceptance_final.json`：最终报告验收，链接原始数值验收和报告补充登记。
3. `protocol.json`、`checkpoint_audit.json`、`historical_baseline_chain_check.json`：冻结协议、官方模型与封存预测证据链。
4. `independent_verification.json`、`forward_execution_audit.json`、`layout_completion_*.json`：独立重算、真实调用次数和分布局完成核验。
5. CSV/CSV.GZ、`raw_group_records.jsonl.gz`、每布局 probe ZIP：全部案例、head、有效视角数量、原始记录与预定少量体素证据。

完整对照始终为 **Multi-scale MRGR + Uniform Fusion**，seed 20261011，epoch285，342725参数。
本轮没有训练或GT64/GT128读取，只诊断已冻结函数的敏感性。

`acceptance.json` 和 `mrgr_oari_mechanism_v2_zh.md` 保留原数值管线及自动生成报告作为审计证据。
最终报告只修正并补充解释，没有改变原协议、原数值源码、干预、统计或预测。
特别是：本轮五布局draw0中，历史P1的实际无效来源污染率为0；不能将源码在其他mask条件下的潜在风险当成当前数据的实测污染。
详见 `report_supplement_registration.json`、`interpretation_evidence.json` 和 `preflight_notes.json`。

P1*是逐体素共同重排有效视角索引的空间排列场，保持每体素矩阵约束，
但不表示空间上统一、物理可实现的新采集几何。它与历史全局排列的扰动分布不同，
两者的响应差不能当作mask的因果贡献。V3的8个场不是8种独立局部排列。

大型预测和完整排列数组留在独立scratch目录，由清单中的SHA256链接；不会重复推理已封存基准。
仓库的probe压缩包不是全体素attention缓存。权重NPZ仅用于独立代数复核，不是新训练检查点。
大CSV采用gzip；所有统计样本量以35案例为准，不以体素、head或排列数量充当样本量。

最省资源的后续优先级见 `next_step_zh.md`：先复用现有Uniform Fusion无射线模型做极小冻结路径探针，
暂不新增训练；正式报告中的方位角训练对照只是针对“重训练增量价值”问题的条件性备选。
