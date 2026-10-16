**目标适配性与非 NeRF 方法调研资料**

对应报告：[GeoPhy-Set3D：目标适配性、非 NeRF 方法比较与研究建议](../../goal_architecture_assessment_non_nerf_zh.md)。获取日期为 2026-10-15（北京时间）。

本目录归档定向文献检索和原文响应，未运行重建实验、模型训练、checkpoint 读取或新的 GT 评价。原始网页可能含作者的结果表和参考文献；报告只引用实际核对且与本任务相关的内容。

| 文件 | 说明 |
| --- | --- |
| `selfsup_sparse_recent.*` | 自监督/无监督稀疏视图与有限角层析，按首发时间排序的定向检索 |
| `multi_operator.*` | 多算子下不完整测量学习 |
| `geometry_generalization.*` | 几何条件化层析/算子方法 |
| `ambient_diffusion.*` | Ambient Diffusion 候选；含不相关命中，未全部纳入报告 |
| `gaussian_corrected.*` | 宽化检索后核对高斯层析原文身份 |
| `sparse2inverse.*` | Sparse2Inverse 精确标题检索 |
| `core_metadata.html` | arXiv API 返回的 XML，后缀沿用统一响应命名；传统方法及 Ambient Diffusion 的元数据/摘要 |
| `*_full.html` / `*_full.txt` | 获取的 HTML 全文与供阅读的纯文本提取；成功获取不等于逐式全文审阅 |
| `*_access.json` | 全文/API 获取 URL、访问时间及成功响应 SHA256 |
| `retrieval_corrections.json` | 错误候选编号和空查询的纠正说明 |
| `local_evidence_manifest.json` | 本次所依据源码和历史报告的哈希快照 |
| `source_manifest.json` | 报告引用的 16 篇原论文、阅读层级和对应归档 |
| `document_validation.json` | 文档链接、来源身份、哈希核验；不是模型效果测试 |

报告重点核对了 11 篇论文正文中的相关方法、理论条件或实验设置：Sparse2Inverse、SPLIT、多算子无监督学习、Equivariant Splitting、ELIPPS、2026 拆分设计比较、几何感知扩散、NIO、GAIA、R²-Gaussian 和高斯射线追踪。其余列为摘要/已有档案支持的方法概述；没有将它们描述为本次复现。

来源纠正：最初猜测的 R²-Gaussian 编号 `2409.03593` 实际返回另一篇电力系统论文，已在标题核对中拒绝；`r2_gaussian_full.*` 仅是这次无关响应的保留记录，不作为科学证据。正确编号为 `2405.20693`，使用 `r2_gaussian_correct_full.*`。`core_metadata.html` 中同一错误编号的条目也不参与引用。

`gaussian_tomography`、`gas_selfsup` 及最初 `senseiver` 等精确检索返回空集；这只说明这些检索式未命中，不能说明不存在相关研究。宽化 Gaussian 检索获得了正确原文。Google 的两次初步连接失败，未用于文献证据。

2025—2026 年新工作按原论文和作者报告解释，不预设已经独立复现或具备适用于本项目的理论保证。检索未穷尽全部数据库，也没有用不同任务、不同标签或不同推理预算的数值建立统一排行榜。
