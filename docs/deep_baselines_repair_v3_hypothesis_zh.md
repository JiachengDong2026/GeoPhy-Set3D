# 第二个候选：仅修复consistency的共同幅度缩小奖励

在第二候选pilot前登记。DeepSets v2在32train/36val完成20轮/320更新后NO_GO_NEAR_ZERO_COLLAPSE；H4=1.0000011023，mean=8.67e−9，测量置零干预通过0/4。此时weighted consistency梯度0.00484475，是weighted hold梯度2.28e−5的约212倍。仅初始化未阻止失败，不能据此启动正式训练。

原公式对于固定A/B场的共同幅度alpha为 alpha*N/(alpha*D+1e−6)，其幅度导数为N*1e−6/(alpha*D+1e−6)^2>0。即保持相对形状误差不变，仅把两个场同时缩小也能降低consistency。独立合成场AD/FD与解析式核对了这一因果性质；极小alpha时应比较对数幅度导数，绝对dL/dalpha含1/alpha级相消，首版探针的绝对误差断言失败原日志保留，未据此改变pilot门槛。

v3保持v2全部初态参数、架构、sigmoid、输入、投影器、四分项loss权重和优化器，只把独立基线consistency改为 mean|A−B|/[.5(mean|A|+mean|B|)]，零/零定义0，FP64归约后返回FP32。新的loss有齐次尺度不变性，去除共同幅度的人工奖励；不是删除一致性项，不改变任何measurement loss。通过独立文件和仅v3进程内adapter接入，原selfsup.py以及主模型loss不变。

这是**新loss协议**，不能声称与旧Plain训练loss逐公式完全相同。预期至少去除缩小幅度的梯度来源，但它不解决有限token条件信息、坐标MLP先验、射线逆问题欠定等其它困难；只有相同fresh初态/案例/预算的pilot和原科学门槛才可判定是否足够。若仍失败，不再增加第三候选或宽度/LR/seed搜索。

先对已失败的DeepSets登记；GNOT仅在v2独立失败且证据支持时采用同一第二候选。两个模型分别GO，不因另一个改善而自动准入。所有GT、测试测量/标签及第二种子均禁止。

GNOT v2于10轮/160更新结束：完整val H4=1.00019045368876，mean=1.35e−6，测量置零通过0/4；e5→e10改善小于.005，无继续pilot的科学趋势，独立判NO-GO。其原失败checkpoint的consistency加权梯度已超过hold约5.47倍，且同一解析幅度奖励适用，因此登记GNOT的第二候选v3，初始化/样本/有效射线/优化预算与v2匹配。不追加第三候选。
