# Ray-Senseiver-SS v1：训练前冻结协议

本轮只训练 Senseiver 的射线积分、自监督任务适配版本，冻结 Ours 与 DeepSets。官方源码固定 e443eb0ca188b5e83a0e6ce5afb8db7d90393de1，BSD-3 许可及 LANL notice 原样保留在 third_party/senseiver；所读 model.py/network_light.py/positional.py/s_parser.py hash 见 artifacts 官方来源。论文 https://www.nature.com/articles/s42256-023-00746-x 。原论文空间点传感器与场标签训练改为完整射线积分与 measurement-only objective；不称直接复现，不使用 dynamic sensor placement。

每条 valid ray（包括零测量）一个 token；invalid/padded view 排除，完整64×64扫描，不采样 ray。entry=origin+direction*t_near，exit=entry+direction*path_length。输入为固定尺度归一化积分，entry/exit 的 `2*x/.6-1`，unit direction，length/(sqrt(3)*.6)，以及 entry/exit 的16频带 sin/cos(pi*x*f)，f=linspace(1,32,16)。不学习 station/view 编号；积分不解释为局部浓度。全 invalid 使用零虚拟 token 作为先验，绝不 NaN。A/B 单独输入，H 不进网络。

沿用官方 Encoder/Decoder 的残差、LayerNorm、GELU、learned latent 与坐标 cross-attention。64 latents、64 channels/preproc，num_layers3，每 block3 self attention，所有 heads4，decoder/query64，learned output query size1，bands16、dropout0。layer_1 独立；layer_n 在第2/3次调用共享参数。没有 lifting/OARI/GCA/MRGR 或3D卷积。query 使用 endpoint-aligned [0,.6]^3 XYZ（y高度），官方三轴 sin 后三轴 cos 顺序；完整262144 queries，sigmoid FP32 [B,1,64,64,64]。首选 chunk16384，整个 decoder chunk non-reentrant activation checkpoint，encoder只执行一次；可修复显存配置，不改变科学参数。小样本验证分块预测/所有参数梯度一致及所有 query 梯度，真实最大 A5/B5/H2 forward/backward/step，置换/padding/zero-valid/all-invalid/GT保护验证。

166 train/36 Val，Pool A；原自然 A3–5/B3–5/H2 两两不交，Val固定A4/B4/H4 IDs，原归一化32284.894580078173 ppm·m。直接调用原 selfsup objective/project_input/run_epoch/selection_metric；完整 L_hold+.5L_cross+.1L_obs+.05L_cons（内部SmoothL1/relative/epsilon/mask/reduction不变），P_diff/K96/all valid rays。seed20261011，FP32 effective batch2（允许1/accum2），AdamW lr2e-4/wd1e-4，clip1，cosine T_max300，全300epochs/24900次更新，不早停/调参/追加轮数。smoke独立，正式fresh初始化；只允许本run last.pt恢复完整状态。natural Val双预测平均 held-out relative L1是唯一selector。允许新基线胜出。

正式训练前登记全部900输入、旧两模型标量/来源/checkpoint hash、源码及metadata。300完成和best SHA冻结后，独立measurement-only selector replay，保存全部900完整输出/IDs/hash，再独立saved-output Val GT64 evaluator；后者不读checkpoint/measurement。禁GT128/Pool B/final test。物理观测身份一致，Senseiver无需lifting mask；逐行完整measurement/geometry/path/view/ray/H tensor hash与已存DeepSets身份核对，旧 Ours/DeepSets900标量复用。

十指标全部报告：3D L1/L2、MAE/RMSE、peak absolute、integral proxy relative、observed/H4 L1/L2。每case/group先平均5draw，再36case paired bootstrap10000、seed2026101801、逐项95% percentile CI，tie1e-8；主 Ours−Senseiver（负有利Ours），辅助 Senseiver−DeepSets。不算p值，不作多重校正显著性宣称。报G1–G5、macroG1–G4、G5绝对、G5−G1、总体组均值之比G5/G1及P90/P95/max/worst ID。

同GPU batch1 FP32成本用case0/18/35 × draw0 × G1/G3/G5共9输入，warmup10/repeats30同步计时，三模型交替顺序；Ours计内部projector，Senseiver计encoder+完整64³ decoder。记录参数/FP32字节、训练实际墙钟/steps/峰值、chunk/checkpoint、median/P90/allocated峰值及共享GPU限制。固定图：三模型五档误差、主配对CI、训练measurement、9个GT及三模型重建（固定y32=.30476m与y均值、x/z单位m，共享每case色标）、精度成本图。

中文报告回答机制适配、输入/梯度、实际训练/塌缩、身份、各档胜负不确定、L2/H4一致性、成本、末期收敛迹象与一个下一阶段建议。单seed/历史参考/开发Val/固定预算，不外推原论文能力上限，不归因单模块。完成停止，不调Ours/新基线/多seed/最终test。

外部目录 outputs/comparisons/ray_senseiver_selfsup_v1 与 outputs/checkpoints/ray_senseiver_selfsup_v1；Git仅代码/配置/标量/图/审计，按协议→实现→结果实际顺序提交并经7897代理推送。注册清单见 docs/reports/artifacts/ray_senseiver_selfsup_v1/protocol.json。
