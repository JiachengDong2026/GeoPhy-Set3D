# 共享权重双步MRGR反馈 v2：独立设计与预注册

动机：原MRGR只执行一次可微观测残差反馈。这里的物理约束指测量线积分正演一致性，不是流动/扩散PDE约束。若残差编码器学到的是局部修正规律，可否不增加参数，在第一次重建上再次应用同一修正，以减少仍未解释的观测残差？

实现保留原多尺度核心完整不变。第一步返回C0、C1、context1、visibility_base；第二步对C1使用相同Q96真实物理投影，重新计算带符号观测残差，复用原residual_encoder、lifting、residual_adapter、fusion和decoder，得到context2与C2。第二步依赖C1的路径不detach。对外C1表示最终场，C_feedback1保留原第一步场。

两臂wrapper均持有core.*权重，参数总数344,678完全相同，初态张量逐位相同。B_one只调用原核心，T_two多一次反馈；原残差adapter最后层零初始化，故零初态下context2=context1，完整初始输出需通过真实GPU逐位检查。CPU已完成参数一致与代数/梯度语义测试；GPU待前一线路结束后顺序运行。

**此路线预注册独立计算例外：完整轮时间比≤1.50、300轮外推≤5小时。** 这里有意增加一次物理迭代，和新增轻量模块1.20成本门槛不同；该门槛在GPU尝试前注册，未改变旧方案门槛。质量条件仍为普通H均值至少改善1%、普通P90及困难聚簇5布局均值恶化不超过2%，正确性与输出非坍缩通过。

同GPU1两臂顺序20轮；seed20261018，固定第20轮末权重。166train/36val、原目标、AdamW2e-4/wd1e-4、300轮cosine前缀、batch2、FP32、Q96、归一化与站位/loader规则不变。若首种子通过，重复seed20261019；不由子agent自行启动正式300轮。GT guard开启，不读三维GT或最终35测试案例。

额外记录首2验证案例、固定T_two末轮权重的steps1/2比较：确认同权重steps1输出等于C_feedback1，量化最终场变化、观测残差变化、留出H变化。真实smoke记录全部共享参数梯度与C_feedback1中间梯度，确认新增迭代不是无效分支。重复迭代不保证收敛，也可能放大噪声/过修正；投影改善不代表三维改善。即使两步胜出，收益也可能部分来自有效深度或反馈增益变化；当前两臂不能单独证明重新计算物理残差是唯一原因。若成为主线，后续可对照重复旧残差或两倍单步增益。

- `scripts/mrgr_two_step_v2.py`
- `scripts/probe_mrgr_two_step_v2.py`
- `tests/test_mrgr_two_step_v2.py`
- `scripts/verify_mrgr_two_step_v2_intervention.py`
- `outputs/research/mrgr_two_step_v2_seed20261018/protocol.json`
