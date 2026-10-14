# 统一深度学习比较接口

输入使用原projection_pool_collate的完整measurement-only batch；`model(batch)`返回FP32 `[B,1,64,64,64]`质量分数。view mask表示真实station，valid-ray mask表示真实射线，两者不得混用。H不进入model；训练A/B分别forward，只有原projector读取另一输入或H的目标geometry。

`training.deep_comparison`提供独立baseline严格config校验、模型构建/身份、prediction检查、tensor/source SHA及协议验证。`models.build_model`只新增DeepSets注册，已有注册名行为不变。原SelfSup v1 validator仍只允许原MRGR。新的模型需独立校验，不能修改原保护或假造MRGR残差分支。

`training.selfsup`的make_dataset/make_loader/objective/run_epoch/selection_metric是原封不动复用的模型无关路径。只新增独立训练入口与更新计数/输入观察器；观察器只记录实际model输入，不改变数值函数。有效batch分组、case/station schedule、selector、checkpoint model/optimizer/scheduler/RNG及source hash均归档。恢复只允许本run last.pt。

评价分离为selector replay、measurement-only infer、saved-output GT64以及GT-free成本timing四个子命令。独立冻结身份必须先完成完整训练与best SHA。900清单包含case/index/group/draw/geometry ID/有序input/H station IDs；逐行校验tensor身份、view/ray/lifting masks与历史来源后保存预测、十指标、前向时间、显存和SHA。GT evaluator不能读checkpoint或measurements；inference不能读GT。两阶段共同禁止Pool B/test/GT128。

`evaluation.deep_comparison`按case/group/draw配对，拒绝缺行、重复、station顺序/identity/mask差异；先五draw平均再36case paired bootstrap，统一Δ=Ours−baseline。指标定义完全复用VolumeMetrics与relative_projection_metrics，不重新定义峰值或积分。训练、评价、归档可独立恢复，不重复已保存且hash一致的预测。

当前实现只支持预注册GeoLift-DeepSets-Sum训练；比较合同及评价清单可复用到后续Senseiver适配，但新增模型需另行声明结构、信息/标签、selector、预算和访问规则。本轮不实现Senseiver，不承诺将其他结构强行套入既有model-specific训练入口。

执行示例（先检查/登记/提交source manifest，再正式训练）：

```bash
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/train_deepsets_selfsup.py smoke
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/train_deepsets_selfsup.py train
# 完整300轮及best冻结后
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/evaluate_deep_comparison.py selector
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/evaluate_deep_comparison.py infer
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/evaluate_deep_comparison.py timing
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/evaluate_deep_comparison.py gt
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python scripts/report_deep_comparison.py
```
