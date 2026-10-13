# Geometry Augmentation formal v1：一次匹配干预预注册

本协议继承 frozen SelfSup v1、五档 sensitivity 和显式 augmentation preparation。历史 original v1 和 same-q 300 epoch NO-GO 全部保留；本轮仅测试已冻结显式 sampler，不实现经典算法或新网络。适用本次单 seed、Val Pool A 的匹配确认实验，不是最终 test 结论。

## 前置结果与证据

Task A：独立新模型、`model.train()`、batch2/accum1、FP32、原 objective、AdamW(2e-4,1e-4)、clip1；Control/Treatment 各重复一次 fresh 首步，共4次真实 optimizer step。有限输出/损失/非零梯度、参数变化、optimizer state 建立与禁读均检查。首步 loss 两次完全一致，Control/Treatment 参数最大绝对差分别约3.28e-7/2.24e-7；optimizer tensor差约6.98e-10/4.66e-10，tensor hash不一致。接近零参数的相对差可大于1，不能只报告 loss 相等就称逐位复现。独立原训练函数与新标量记录函数的10步比较、所有100步 replay 明细见归档。

优先保留历史 `seed_everything` 的 TF32=false、cudnn benchmark=false、FP32 配置。历史未强制 deterministic algorithms/cudnn deterministic；CUDA 2D/3D grid_sample backward 使用非确定累加，强制 deterministic 会使该路径不可用。两臂使用相同设置；本轮推断限于一次 matched seed，不能排除随机数值路径的影响。严格审计 tolerance 在执行 smoke 前登记：abs1e-7、rel1e-6；超过时如实记录，不追认 bitwise equivalence。首轮共享 GPU2 显存被其他用户占用导致 smoke OOM，未执行 optimizer；保留失败，改用GPU0/1，不修改 batch或数值超参。

Task B：核对全部300 case orders、49,800个A/B/H计划、V、batch composition、worker0/shuffle=false、初始 full state、原 optimizer/scheduler/loss/归一化与36例自然 Val selector。实际 fresh replay100步，保存1/10/100参数与optimizer hash、loss parts、LR、clip前后 norm；另运行原函数与独立标量记录函数各10步。旧日志只有epoch汇总，缺少历史逐步参数/optimizer、clip后梯度、初始独立param/buffer hash和完整CUDA运行flags，不能证明完整数值等价：**CONTROL_REUSE = REJECTED**。重新 fresh train两臂，不复用旧结果拼接新Treatment。

Task C：新seed2026101601，V4每档10个，共50个确认layout；按照station ID排序后的组合排除历史25个，不以顺序或新ID冒充新组合。沿用冻结q05/q40/q60/q80：0.0849504590903776、0.3527876964960668、0.41610931419256153、0.45343856393747445。G1≥q80；G2[q60,q80)；G3[q40,q60)；G4[q05,q40)；G5<q05且circular span≤5°。训练hard≤10°与确认G5≤5°保持区别。实际射线方向、粗16³ coverage、站位距离/高度为描述，不称可观测性。几何选取只读Pool A geometry/source index，不读取measurement、GT或模型。每case/draw五档共用H4，排除该draw五档输入并集；H seed2026101602。

## 唯一干预、训练与选择

配置入口 `configs/geometry_augmentation_formal_v1.yaml`，所有部署路径可通过`--config`配置。独立输出`geometry_augmentation_formal_v1`。Control原DisjointABHSampler；Treatment完整复用冻结GeometryAugmentationABHSampler计划，natural/moderate/hard=.50/.25/.25，moderate边界与V5 population完全继承preparation，不重估。A/B mode先独立抽样，disjoint rejection仅重抽站位，mode/V不重抽；H2在补集均匀抽样。station marginal shift为干预机制的一部分，不能临时平衡。

两臂同seed20261011、同fresh初值、同300epochs、166train case/epoch、83step/epoch、24,900step/arm、相同case order/V schedule/A/B/H数量、原GeoLift-OARI-MRGR-Res3D332070、原 loss hold+0.5cross+0.1observed+0.05consistency、AdamW lr2e-4 wd1e-4、cosine T_max300、epoch结束Val之后scheduler.step、clip1、effectivebatch2。无AMP、无warm start、无early stop。只允许本臂last续跑，恢复model/optimizer/scheduler/RNG，已完成不重算；不读取任何外部checkpoint训练。

每epoch保存total/hold/cross/observed/consistency、原测量指标、LR、每stepclip前后梯度、时间/显存、selector、best状态、case/station digest。实际digest必须逐epoch与登记计划一致；站位分mode频率、inclusion/exclusion、A/B与joint3×3、V、拒绝总量/最大次数全程归档，不事后重平衡。

selector完全沿用原自然Val双输入A4/B4/H4 mean held-out measurement relative L1，严格`score<best`才更新，300epochs全部执行。训练/Val/选择读保护拒绝GT64/GT128、Pool B、final test、监督/foreign checkpoint及数据写入。只在两臂300完成、best SHA冻结后读取本臂best进行无GT测量评价与原selector replay；随后独立进程只读saved predictions及36例Val GT64做3D指标。GT不参与选epoch/布局/阈值/比例。

## 冻结评价与统计

每臂36case×(历史自然V4五draw+已有25layout+确认50layout)=2,880forward，总5,760。三部分单独报告，确认集为主终点；historical natural沿用历史输入/H，development仅诊断。每个case/geometry两臂输入与H相同。保存预测后才可进入独立GT64流程，H仅评价，不输入network。

10指标：3D relativeL1/L2、MAE/RMSE、peak absolute error（质量分数，无量纲）、integrated concentration relative proxy error（冻结voxel sum比值，无量纲）、input/H relativeL1/L2。前向计时CUDA synchronize，记录显存；共享GPU计时仅描述，不当独占硬件benchmark。XYZ、64³、冻结P_diff及原指标完全不改。

36例为统计单位，先同case/同group平均layouts，再配对；bootstrap seed2026101603、10,000次、percentile95%CI，两臂共用case重采样索引。报告mean/median delta、T/C及ratio CI、wins/losses/exactties、Wilcoxon。主终点不做多重校正；其余组别/指标及退化检验统一Holm校正，并明确描述性解释。worst-case同时给case-group均值的mean/median/p90/p95/max/worstID和单layout最坏记录，避免混淆。

主终点confirmatory G5 relativeL1：delta=T−C，CI upper<0。natural非劣界限是确认G1–G4逐case等权macro，grand mean T/C≤1.03。遵照本轮指定采用**point ratio规则**，同时报告bootstrap ratio CI；不能宣称CI非劣证明。

精确决策：

1. **GO**：G5 delta CI upper<0且natural ratio≤1.03。
2. **PARTIAL**：GO失败、natural ratio≤1.10，且（G5 CI upper<0，或G5改善≥10%且delta CI跨/接触0）。不称正式成功。
3. **NO-GO**：其余所有结果，包括明显natural损害。阈值冻结后不得改变。

实际改善1−T/C解释：<5%、5–10%、10–15%、≥15%、≥25%；负值是恶化。标签不改GO。次终点：每臂G5/G1、G5−G1及配对变化（先case组均值、再grand mean ratio；CI重采样case），历史3.37仅背景，不作阈值。

## 完整性、归档与停止

正式训练前提交本协议、源码/config哈希、前置结果和几何清单。冻结全部训练/评价/统计源码；报告文案可在结果后新增。历史标量文件哈希/大文件metadata、dataset内容清单验证保护旧产物；不为审计读取GT128内容。Git不提交checkpoint、GT、dataset或大tensor。产物归档源码/config/协议、geometry IDs/H、seeds、sampler与access审计、训练标量、逐case/draw CSV、统计JSON、图、hash、中文报告、完整regression结果。

最终报告逐项回答指定15问题。不管GO/NO-GO均仅建议下一阶段单独预注册model-side geometry awareness；不实施新模块、ratio/threshold/curriculum/loss搜索或新训练。本轮完成并同步GitHub后停止。
