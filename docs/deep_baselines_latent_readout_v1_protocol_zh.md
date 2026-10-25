# 冻结内部表示与线性可读性诊断 v1：运行前协议

只读取固定几何 v2 的两个 checkpoint_200、登记的 AB/geometry 和解析合成 truth；不读取真实数据。eval + inference_mode、requires_grad=False；神经网络更新为0，剩余100次预算不动，旧队列不恢复。所有新数组写入独立 scratch 命名空间。

## 提取与复现

逐模型、逐案例、逐 A/B 分支先无 hook 复现 v2 全64³场，要求逐位相等，之后开启只读 hook/wrapper 再要求逐位相等。DeepSets 原始有效测量/掩码、phi token、mean pooled latent（从decoder首层输入读取）、decoder首层输出、末层Linear前128维及logit/浓度。GNOT measurement_encoder token、两块cross测量源key/context统计、两块before_self/after_self全空间状态及global self key/context统计、head末层前64维及logit/浓度。GNOT没有单一 pooled latent，用实际全局算子统计与末块空间状态描述，绝不新增网络。

normal 8例全量保存；所有zero输入在同分支逐位相同，保存1份并登记逐例别名。复用v2全部28个无向配对和56个有向交换/分支。交换同时复制donor raw与归一化伴随字段，保持其余字段；全输入逐位等于donor后复用已验证的donor表示，不重复计算完全相同的推理。每模型/分支额外实跑一个交换并核验各层逐位相等。逐层统计及L2/余弦使用全特征，norm定义 ||x-y||/max(||x||,||y||,1e-30)。另存RMS绝对差，禁止把FP32微差等同有用信息。

## 独立合成监督 ridge

两模型使用同一流程。A/B分别拟合，不混分支；native维度128/64不同，报告系数数目，非等参数竞赛。每模型/分支分别拟合normal和zero特征，共8个读出。逐体素的末层前特征，只做仿射标量线性回归，无神经层/优化器。标签仅seen0–5的FP64解析场。所有64³体素权重相同，FP64分块充分统计；feature中心/尺度仅来自每个fold的5例seen（最终重拟合6例），std下限1e-6，避免无限放大微弱通道。截距不惩罚，目标为mean squared error + lambda*||beta_standardized||²。

事先固定lambda=[1e-6,1e-4,1e-2,1]；leave-one-entire-plume-out六折，验证指标为完整场相对L1，六例等权；均值最小，1e-12内平局选较大lambda。不扩大搜索。最终六例重拟合并封存系数及SHA后，才读取unseen6/7标签。线性预测不裁剪，作为主要评分输出；负值/超1比例另报，质心用正部（零质量返回null）；质量误差用原始有符号积分。该读出是解析监督诊断，不能称原无监督模型成绩。

完整体积指标：相对L1、MAE、mean/std/max、非零比例(>1e-10)、正结构比例(>1e-6)、质心误差m、积分相对误差、Pearson相关。zero-field相关/质心未定义用null。同时报告normal ridge相对original、zero ridge、zero field的改善，以及unseen对(6,7)预测差分与真差分的cosine/relativeL1；零特征不能区分羽流，防止共同模板的虚假提升。

## 运行前判据（有界诊断）

表示meaningful响应：逐分支聚合/算子条件表示（DeepSets pooled_latent；GNOT末块measurement_operator）的28对归一化L2中位数>=1e-4；测量token另报但不单独据此宣称条件解码响应。低于只称微弱响应。B/GO需两个分支各两个unseen都满足normal ridge相对original、zero field、zero ridge的L1绝对改善>=0.01，并且unseen差分cosine>=0.1且相对差分误差<0.99；还要求相邻候选lambda的unseen结论不用于选择（不额外评价），用seen六折最佳lambda一致率>=3/6、求解相对残差<1e-6及有限系数评估稳定性。证据只覆盖2个unseen/单几何。

A/NO-GO：上游条件表示低于响应阈值且normal probe未通过上述条件收益门槛；只建议停止当前版本，不能归咎于输出头。C/NO-GO：响应达阈值但探针不达门槛，不能断言encoder失败，可讨论非线性组织、层选择、覆盖/可辨识性；没有明确B证据不授权新训练。数值不稳定、分支结论相反或未完成则D/INCONCLUSIVE。所有科学阈值在unseen标签评价前固定，不追加训练、seed、调参或读取真实GT。

独立脚本不导入执行期metrics/decision，重算保存的全体积指标、CV选择/划分、表示统计和分类；审查checkpoint、历史Git跟踪文件、队列及冻结主模型哈希。异常必须保留，恢复以新目录/新增脚本明确记录，不静默覆盖。
