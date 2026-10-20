# MRGR-UniformFusion 正式候选身份与训练记录

主模型只替换GCA融合，保留描述符、射线关系、集合交互、多尺度、损失与原始测量残差反馈。部署标识为MRGR-UniformFusion-v1；固定seed20261011、第285轮，不根据本轮测试结果改选。全部参数可训练，逐张量名称/形状/参数数目见parameter_inventory。

有效视角权重为m/max(Σm,1)，m=visibility AND view_mask；无效特征先屏蔽，NaN无效视角不污染输出；全无效位置精确零。C0/C1共用一个decoder，两次前向而非两个独立模型；实际中间张量及decoder调用见同设备benchmark。

统一原训练配置：300轮/24900更新，FP32、TF32关闭、batch2、AdamW2e-4/wd1e-4、cosine300、梯度裁剪1，原A/B/H采样、loss和p99.5归一化不改。新增两种子从20轮完整断点续训，和OARI匹配恢复路径；前缀不另算预算。所有种子按全程验证投影H4最低值选轮，无体积GT。

| 种子 | 参数 | 最佳轮 | 检查点SHA256 | 来源 |
|---|---:|---:|---|---|
| 20261011 | 342725 | 285 | `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc` | 历史复用 |
| 20261019 | 342725 | 286 | `19bb5f698349c976358be782e9d43301afa9e1eec4a89ddb4e1ceeb8d9bd1424` | 历史复用 |
| 20261020 | 342725 | 292 | `581cca5ba4c36359dc7c9b0d13aa4ca54416f0b7358e21cd61d5c63f46014cd2` | 本轮新训练 |
| 20261021 | 342725 | 291 | `aadaa583b87e87dc942588b5493c4aae3bf7da23e407ef06c84cae22f6322bdc` | 本轮新训练 |

[正式配置](../../configs/mrgr_uniform_fusion_v1.json) · [固定身份](artifacts/mrgr_uniform_final_v1/model_identity.json) · [所有种子权重清单](artifacts/mrgr_uniform_final_v1/model_catalog.json) · [参数清单](artifacts/mrgr_uniform_final_v1/parameter_inventory.json) · [实际前向形状/成本](artifacts/mrgr_uniform_final_v1/benchmark.json) · [逐轮日志与选轮核验](artifacts/mrgr_uniform_final_v1/training_audit.json)。

大型权重保留model_catalog记录的服务器路径，Git不提交权重或预测数组。推理必须使用正式配置登记的uniform工厂；原config.model.name只是未改动的历史基础构造器，直接build_model会遗漏均匀融合替换。统一复现入口见scripts/uniform_final_v1/README.md。
