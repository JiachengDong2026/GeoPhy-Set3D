# OARI 四项定向实验：实现设计与差异清单

本阶段新增三项300轮/24900更新训练，以及正式seed20261019模型的70个draw0冻结obs_token探针。全部结果属于同一已使用35案例上的探索性比较，不是独立测试确认。不得覆盖历史模型或协议。

## 单因素实现

- 以官方Uniform MRGR工厂构造新鲜完整模型，保持Encoder、GeometryLifting、原始4维几何描述符、多尺度decoder、Uniform Fusion和Physics Residual Feedback的同种子初始化逐张量相同。
- 新增独立PlainStage0与MRGR子类，使用Python MRO使原始MRGR.forward的super()进入PlainStage0；反馈forward原代码不变。PlainStage0只将ObservabilityAwareRayInteraction替换为GeometryConditionedSetInteraction，不计算RayGeometryDescriptor、LocalObservabilityDescriptor、5维token或pairwise bias。
- 普通交互输入为原features/geometry/visibility/view_mask。只从同种子新鲜初始化的OARI模块复制形状兼容的共享子层初始化；48→32 token_projection独立初始化，不加载56→32权重；构造时恢复RNG，避免影响训练采样流。
- return_attention返回context、矩阵、visibility与融合权重；不包含伪造pairwise_bias。原MRGR只读取context_features与visibility，接口兼容。
- no_ray_relations完全调用原官方工厂，其pairwise输入前三列置零定义不变；仍保留可观测性与原参数量。
- 冻结探针严格复用OARI v2 Intervention实现，保留learned bias/SiLU，不训练。

## 输入、统计与访问边界

三项训练使用正式300轮/24900更新、AdamW2e-4/wd1e-4、cosine300、batch2、FP32/禁TF32、完整166训练/36验证案例，按验证投影H4严格最小选轮。测试仅普通Val-A-V4 draw0和clustered-V4 draw0..4，每模型210份；聚簇先在案例内平均五draw，再做35案例配对bootstrap。另单列draw0与冻结探针对齐，不能混合。冻结探针仅两布局draw0共70份。复用完整模型两种子和no-ray种子11的匹配历史预测。

在所有三项训练完成、checkpoint冻结、全部700份新增测试预测封存前，不读取本阶段GT64。用户本轮明确授权在封存后按既有35案例范围评分；GT128始终禁止。统计预登记：差=处理−匹配完整模型；20000次案例bootstrap，seed20261026，95%百分位均值区间、均值/中位数/SD、误差方向计数(阈值1e-8)；辅助指标与布局对比均探索性，不将案例区间解释为训练种子不确定性。

## 工程门槛与调度

CPU检查共享初始化、特征提升相等、mask/padding/排列等变/全无效安全、接口与有限合法输出；单batchGPU完整正式目标反传与优化smoke在独立新鲜模型进行，不进入正式训练状态。通过后启动三项全预算训练。复用现有UUID锁、15秒双快照、进程/显存/利用率检查与子进程继承锁FD；每项独占GPU，忙卡排队，不干预其他用户。

审计时GPU1/2被用户yl占用，GPU0/3空闲；运行前重新检查，不假定四卡仍空闲。
