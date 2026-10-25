# Senseiver 自监督决策审计 v1：零更新复核预登记

本轮依据最新 main `542f7abfd152354400c13d310afea128f6bd39c3` 开始审计。历史正式自监督实验已有 113 完整 epoch、9,379 更新并由用户明确停止；未完成 300 轮。不得把本文件描述为新的训练 pilot 协议。

## 决策与适用范围

先检查历史配置、源码、初始状态、数据元信息、checkpoint、日志和实际更新。如果与请求的原始适配 pilot 同条件，且已有失败证据，则不重复训练，不改名规避 NO-GO。当前 V1 wrapper 声明只封装原始 RaySenseiverSS，必须核验其 fresh 状态逐位一致。

本轮新增计算仅为末态只读复核：不创建 optimizer、不反传、不更新任何权重，不访问三维 GT/GT128、Pool B 或最终 test；不生成论文 900 条推理/GT 排名。原始训练、监督结果、其他模型和旧队列保持原状。

## 固定只读计算

- checkpoint：历史 `ray_senseiver_selfsup_v1/last.pt`；SHA256 `567ec1f88bdaa1e78cd14665727ee353add0cd1ce457256ff9bb963b95e367b9`；epoch112（零起点），9,379 更新。best SHA `c2a748696a806aba28f9483679edead0a69bb986b8e04ce7e123f2985a12ad91`。
- 配置：原始 `configs/train_ray_senseiver_selfsup_v1.yaml`；模型 254,913 参数、64 latent、原始注意力与 sigmoid、FP32、全部有效射线、完整 64³ 查询，query chunk=16,384。
- 固定诊断：train 索引 0/1/2，sampler epoch0；错配 donor 分别 1/2/3。A/B 两支分别执行 normal、zero、wrong_case，共18个完整场。错配 donor 在 recipient 的站位/射线路径取值，仅替换 raw/normalized measurement；其他对象保持一致。
- H 仅在网络输出后计算官方 Pdiff/K96 测量指标，不进入模型。每案例 A/B 指标等权平均。这里 H 是训练案例的未输入站位，不是独立测试集。
- 额外按原 `training.selfsup.run_epoch(..., optimizer=None)` 在完整36例 Val 重放固定 selector 与 loss；只将 loader 改为 workers0 以审计文件访问，不改模型/目标/站位/归约。不复制 loss。
- 保存完整场、logit、H投影/reference/mask、身份、标量与前后权重 hash，供第二个实现独立重算干预指标。结果仅写 `/scratch/djc/GeoPhy-Set3D/senseiver_selfsup_decision_v1`，Git保存小体积曲线/身份/指标和报告。

## 固定判据及不可补证事项

沿用请求的必要门槛：至少2/3案例 normal-zero relative MA >=1e-3；至少2/3案例错配使 H relative L1 增大；正常 full-Val H relative L1<1；相对真正 epoch0 至少改善5%；工程有限且有效更新、无持续近零/恒定/饱和；最后连续两次检查均通过才可 GO。relative MA 定义 `sum(abs(C_normal-C_zero))/max(sum(abs(C_normal)),1e-30)`，保持历史定义。

本轮只检查既有证据，不补写不存在的 epoch0 full-Val，也不把 epoch1当epoch0。单个末态复核不能补成两次成功。已有8例修复协议中 H<0.90/最后三次检查等更严门槛适用于不同初始化/loss对照，不静默替换本轮原始适配门槛；若发现真正适用于同一实验的冲突则停止。任一必要门槛明确失败即可 NO-GO，不因其他缺失判据而宣布 GO。此判断不评价三维准确度或原论文网络能力上限。

## 资源与失败处理

CPU审计先行。GPU0 UUID `GPU-37424437-e208-62d1-e617-53c421c94a49` 采用既有全局 UUID 锁及间隔15秒双空闲快照，最多等待60秒；独立调度目录，监控外来进程，冲突只停止自己的子进程。OOM/身份冲突/访问越界安全退出并保留日志，不转为训练，不恢复旧队列。

## 交付与后续

交付历史更新审计、原始配置/源码/初始权重/checkpoint SHA、113轮曲线及前30轮标记、最终逐案例干预、Val重放、独立复核和中文决策。若同协议失败确认，结论为 `NO_GO_EXISTING_SAME_PROTOCOL_FAILED`，不登记无新假设的 fresh pilot、不启动正式训练。只提出一个结构不同的机制假设候选，不实现或扩展搜索。
