# OARI几何利用与顺序不变性诊断进度

北京时间：2026-10-21T14:54:55.607801+08:00
状态：`complete`；完成布局：17/17。验收通过，GitHub 远端已核验提交 `68c57d3c0933adaf009b8e0e2c5144833f19f8eb`。

[正式报告](mrgr_geometry_utilization_diagnostic_v1_zh.md) · [结果解读与审计补充](mrgr_geometry_utilization_v1_interpretation_zh.md) · [验收记录](artifacts/mrgr_geometry_utilization_v1/acceptance.json)。

冻结UniformFusion seed20261011/epoch285，无训练、无权重更新、反馈OARI未重启。
原始A预测复用。P0/P1每输入10种非恒等置换，V3穷举仅5种并单独记录；P2移除首层仿射中的显式关系加性贡献。
全部新增预测封存后才读GT64；GT128禁用。重复置换不当作独立案例或种子。
运行目录：`/scratch/djc/GeoPhy-Set3D/mrgr_geometry_utilization_v1`。
