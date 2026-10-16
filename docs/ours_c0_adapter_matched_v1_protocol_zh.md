# Ours C0 局部方向描述适配：固定输入有限配对实验

本轮检验一个假设：原 OARI 的局部方向描述经集合交互及池化后，直接再次提供给 C0 解码侧，能否改善固定困难布局下的三维精度。原模型已经利用这些描述；本实验不是首次加入几何，也不声称描述等于可辨识性或 Fisher 信息。

新增唯一模块为 `concat(fused32, existing_descriptor5) → Conv1³(37,16) → SiLU → Conv1³(16,32)` 残差，末层零初始化，共 1,152 参数，仅作用于 C0，局部全不可见区域修正严格为零。C1 的融合/解码和原 signed residual 路径保持原数学定义。新模型隔离实现，不修改旧模型、loss、采样器或 projector。

Control 原 v1 与 Treatment 新模型均从 SHA256=`1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026` 的无三维标签 v1 best 开始，共享参数逐项一致，并核验零初始化的实际 C0/C1 输出一致。不是 fresh 训练，也不与原 300 轮预算冒称等预算；本轮两臂各增加20轮/1,660次更新，AdamW lr2e−5/wd1e−4，cosine20，FP32、batch2、clip1。只重置本轮 optimizer/scheduler，epoch0无更新模型纳入 measurement-only best 选择。

训练复用原166 train案例、原 natural DisjointABHSampler 的 epoch0–19，预存完全相同的逐例 A/B/H station IDs 和 case order。两臂站位、观测数量、所有测量/几何/mask、顺序、loss及预算相同；没有视角优化、增强、重采样搜索或额外输入。原36 Val自然 A4/B4/H4唯一选择指标为 mean held-out relative L1；H只作目标，不进入网络。训练/选择禁止GT64/128、监督权重、Pool B及test。

完整训练后冻结两臂best SHA，重放selector，再按历史900条固定V4清单推理并保存 C0/C1 及输入hash。全部输出封存后，独立 saved-output evaluator 才读取36 Val GT64；不加载checkpoint或重新推理。与原reference逐条验证站位顺序、measurement/geometry/mask hash。与原冻结v1和同期Control分别比较，不把重复layout、射线或体素当独立样本。

主终点：G5三维L1；普通布局G1–G4 macro为保护终点。先每case/group平均五layout，再以36case配对bootstrap10,000次，固定seed2026102001。成功仅指此次有限探索：Treatment相对Control与原v1均至少改善G5 5%，对应差值95%CI上界<0；普通macro L1和G5 H L1相对两参考均不超过1.03。L2/peak/积分/最坏case同时报告，不能以测量拟合替代三维精度。失败/不确定不更改阈值、不搜索第二模块或超参。

单seed、小额继续训练及已经研究过的36 Val限制结论；本轮不是论文最终性能确认，不是新增三维标签预算曲线。归档协议、源码hash、初态、训练访问审计、配对输入核验和冻结输出。用户已授权本项优化尝试，本轮实施与有限对照均在该范围内。
