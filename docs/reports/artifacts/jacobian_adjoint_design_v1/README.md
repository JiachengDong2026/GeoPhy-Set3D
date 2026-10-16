# Jacobian 伴随修正设计：合成数值核验

对应[具体实现设计](../../jacobian_adjoint_refinement_implementation_v1_zh.md)。日期为 2026-10-15（Asia/Shanghai）；JSON 使用 UTC 时间。

本目录只有合成算子核验，没有正式模型、真实数据、checkpoint、GT、训练或 optimizer。

- `verify_synthetic_operator.py`：3³ float64 场、3 条射线、K7；固定几何八邻点 gather/scatter 与仓库现有投影器比较，并检查一、二阶梯度。
- `synthetic_operator_check.json`：本次实际结果及已知范围限制。
- `provenance.json`：现有投影器和复核脚本的 SHA256，以及文档链接核验结果。

从仓库根目录复核：

```bash
PYTHONPATH=src /scratch/djc/envs/gas-torch/bin/python docs/reports/artifacts/jacobian_adjoint_design_v1/verify_synthetic_operator.py
```

脚本会重写本目录的合成结果 JSON。它只核验内部采样点和 (0,1) 浓度范围；完整网格、边界、mask/padding、CUDA/FP32 和性能尚未核验。这些小规模数值结果不能证明三维重建精度提升。
