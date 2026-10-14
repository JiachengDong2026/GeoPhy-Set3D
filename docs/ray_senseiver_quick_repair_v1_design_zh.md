# 唯一修复登记：全局latent解码残差

阶段A（登记commit86c2293）已完成，last35 encoder只提取16次，未更新。固定四站[26,27,45,42]，0–7拟合、8–15独立检查；fit-only centered latent RMS=.00339497965664、目标mean=.0360886491958/std=.0191742192557。岭回归check MSE=1.316291064e−8，query头=2.388596479e−5，常量=4.969343502e−4；两者均有效。query更新200次，best来自fit step200，没有用check选权重。原query结构在标准化latent/标量任务上能学习，不能断言它损坏；读出结果也不证明三维重建可学。

结合旧同几何干预中latent有响应而最终输出几乎不变，本轮只选择decoder方向的一项最小适配，不改输入或loss。设原query cross层（含其residual/MLP）输出为F(q,L)，新增：

`F_new(q,L) = F(q,L) + W_cond × mean_tokens(L)`

随后仍用原postproc线性层和sigmoid，保留完整64³坐标查询、原encoder/attention/latent数量。W_cond为64×64、无bias、**全零初始化**；新增4096参数，原共享参数逐元素保持。因此epoch0输出函数与Control等价；新增权重从第一步获得梯度，不加载A读出权重，不使用训练/检查集统计到重建模型。没有新增gate、LayerNorm、损失、输入增益或第二版本。

预期：给query特征一个不经过decoder KV LayerNorm/attention的全局latent条件路径，便于利用全局幅度/内容变化；它不绕过encoder。新增路径只含token均值与小线性映射，原chunk每次约64×64=4096 MAC，16块共65536 MAC/样本，加上每query64个相加（完整场约16777216次）；不改变原主要attention复杂度，新增权重FP32约16KiB。实际参数总数在initialization登记精确记录。

不确定性：只用均值可能遗漏token间差异；归档的FD局限不能证明LayerNorm或attention是根因；A使用标准化latent和标量目标，不同于三维训练。这个残差可能只改变整体浓度、甚至加快错误的幅度下降。必须按相同测量L1/L2、空间/case差异及normal-zero依赖判定，不能以total下降或有新通路宣布修复成功。

接下来只执行预登记80epoch/320step上限与原科学提前退出的fresh两臂。若失败，归档此唯一修复，不尝试第二结构或loss；若GO，仅冻结候选，不启动完整训练。
