# 中断pilot的只读复核登记

DeepONet工程smoke已通过；pilot已保存6完整epoch/96更新，epoch7在下一次AdamW step前因FP32梯度norm==0断言停止。前6轮完整Val H已到1.0、输出均值7.176e-22，明显不满足预登记非塌缩/优于零场条件。不得恢复该训练或启动正式300轮。

新增零更新诊断仅加载原last.pt（第6轮）及其SHA，不是中断瞬间内存态。复用原run_epoch重放36例Val，train0/1/2和Val0/1/2/3 normal/zero/wrong-case干预及封存，沿用既定数值门槛；不创建optimizer，不做参数更新。另对固定train0/1的原objective仅backward一次，以FP64累积梯度norm核查FP32范数下溢的机制可能；这不是恢复训练，不据此改loss/初始化/验收标准。旧模型和既有新pilot代码清单保持原样。新诊断代码和协议单独登记，GPU0锁与冲突保护同前。

实际步数的证据限制：last checkpoint保证96次，epoch7至多已执行15次额外更新后触发断言，精确进程总数介于96–111。观察器的逐step trace计划在pilot完成后持久化，但异常时丢失内存trace。这是实现的日志缺陷，必须报告；不将96冒称全部真实更新，不凭重跑猜数。现阶段保留所有原始日志/checkpoint，不覆盖、不重试训练。

FNO独立pilot仍按原封存配置/门槛执行；不能因DeepONet失败自动判FNO。若FNO也出现同类中断，本脚本仅在相应last完整epoch上执行同样零更新审计，按实际epoch与日志形成独立结论。新审计不增加第三候选，不进行任何超参搜索。独立CPU NumPy复算随后核查全部保存场/测量/条件性和最终判定。
