# GeoPhy-Set3D 最终模型冻结清单

本轮以 Git `ee6ac52e07034aee1b23b5779de54818d9b43d27` 已冻结的 **Plain Set Interaction** 模型为准。仓库配置中的历史 `model.name=geolift_oari_mrgr_res3d` 不能单独决定网络身份，必须经 `scripts/oari_targeted_v1/variants.py:model_for(..., 'uniform_plain_set_interaction', ...)` 构建。冻结记录、实际 checkpoint 与 dispatcher 已一致核验。

| 角色 | 模型 | 参数 | 最佳 epoch | checkpoint SHA-256 |
|---|---|---:|---:|---|
| 用户指定论文主模型 | uniform_plain_set_interaction_seed20261011 | 342209 | 298 | `0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c` |
| 独立种子参照 | uniform_plain_set_interaction_seed20261019 | 342209 | 279 | `fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d` |

权重位于 `/scratch/djc/GeoPhy-Set3D/mrgr_oari_targeted_v1/training/<模型名>/checkpoints/best.pt`。未重训、未微调、未覆盖权重。结构是 SharedEncoder2D、固定 GeometryLifting、普通集合交互、多尺度 MRGR、UniformFusion、PRF；本轮不再使用 OARI 作为主模型。

完整机器证据：`/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/main_anchor.json` 及 SHA 旁车；包含逐文件训练源码哈希、实际配置、训练/验证案例列表、原协议/数据索引/训练统计/布局清单哈希和两份权重身份。Git 中保留精简导出 `artifacts/deep_baselines_v1/main_anchor.json`。原冻结与经典算法证据仍在 `artifacts/mrgr_plain_final_v1/`，不覆盖。

输入是允许的 Pool A 射线积分与站位/射线几何；输出 C0/C1 为 XYZ 顺序的 `[B,1,64,64,64]` CH4 质量分数、sigmoid `[0,1]`，坐标节点覆盖 `[0,0.6]³ m`。代表性验证输入、输入哈希和输出张量哈希已记录；预测保存在 scratch `main_anchor_prediction.npz`。CPU 回归重新加载模型，C0/C1 与冻结锚点逐位一致；记录在 `checks/main_regression.json`。

冻结训练为 166 train / 36 val，300 epoch、24900 optimizer updates、batch 2、无梯度累积、FP32/TF32 关闭、梯度裁剪 1、AdamW lr 2e-4 / wd 1e-4、cosine T_max=300。数据 A/B 输入各随机 3–5 站、H 留出 2 站；验证 A/B 各 4 站、H4，三者不交叠。损失 hold/cross/observed/consistency 权重为 1/0.5/0.1/0.05，保留原投影 K96、归一化、SmoothL1/relative-L1 定义。唯一 selector 是最低验证 H4 measurement relative L1，训练及选模均不读体积 GT。

验证 G1–G5 的 36×5×5=900 几何清单与历史 35 案例正式评价是不同协议，不能混用。35 案例已用于开发探索，本轮不能宣称独立测试确认；独立测试状态沿用 `BLOCKED_NO_INDEPENDENT_DATA`。新三基线全部冻结与训练完成前不启动其正式测试，不读取 GT128 或 Pool B。

首次锚点工程检查因旧 guard 输出根目录指向历史 scratch 而拒绝读取本轮 NPZ，随后仅修正新脚本的输出 allowlist；失败证据 `freeze_precheck_failure.json` 和预检查 NPZ 保留。未修改模型、数据或原 guard。
