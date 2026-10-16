# 架构分析与定向文献调研资料

对应报告：[整体架构、性能瓶颈与改进方案](../../geophy_set3d_architecture_bottlenecks_and_research_zh.md)。资料获取日期为 2026-10-14 UTC。

这是文档与只读研究归档，不是训练实验。没有新的模型效果、GT 评价、checkpoint 访问、GPU 前向或 optimizer 更新。

| 文件 | 含义 |
| --- | --- |
| `sources.json` | 10 篇实际成功访问的 arXiv 原论文摘要页元数据、摘要、URL、时间和页面 SHA256 |
| 10 个以论文简称命名的 `.html` | 对应上述摘要页的原始响应 |
| `fulltext_reading.json` | 三篇实际读取的完整 HTML 正文的来源、标题、文件身份及校验信息 |
| `learned_primal_dual_fulltext.html` | Learned Primal-Dual 正文，arXiv HTML 返回 v3 |
| `naf_fulltext.html` | NAF 正文 |
| `equivariant_imaging_fulltext.html` | Equivariant Imaging 正文 |
| `fulltext_manifest.json` | PDF 获取尝试记录，全部失败；不把它作为全文已读取的凭据 |
| `additional_retrieval.json` / `extra_fulltext_retrieval.json` | HTML 正文与 CVF 来源的追加访问成功/失败记录 |
| `static_architecture.json` | 当前 Ours 的 CPU 静态参数统计，不调用 forward、不加载权重或数据 |
| `local_evidence_manifest.json` | 分析所依据的本地模型/训练/物理源码和历史报告 SHA256 快照 |
| `document_validation.json` | 参数加和、相对链接、来源数量与本地快照校验结果；不是科学效果测试 |

阅读范围：10 篇摘要，另读 Learned Primal-Dual、NAF、Equivariant Imaging 正文。报告明确区分全文机制与仅由摘要支持的概述。PDF 网络连接失败、部分 HTML 404/超时均保留，没有用未读取内容补造引用。

调研围绕本仓库的物理反馈、空间融合和无三维标签目标展开，覆盖代表性方法，不声称穷尽截至 2026 年的最新工作。原论文中的精度、运行速度和理论假设不直接转化为四站 TDLAS 的性能承诺。

本地历史数值只引用原报告，没有复跑历史训练/推理。并行演进的 Ray-Senseiver 训练记录不纳入本目录的固定数值证据清单；主报告对其状态的描述只对应阅读时点。
