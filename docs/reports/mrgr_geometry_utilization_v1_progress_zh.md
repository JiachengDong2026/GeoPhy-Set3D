# OARI几何利用与顺序不变性诊断进度

北京时间：2026-10-21T14:54:35.854985+08:00
状态：`accepted_pending_sync`；完成布局：17/17。

冻结UniformFusion seed20261011/epoch285，无训练、无权重更新、反馈OARI未重启。
原始A预测复用。P0/P1每输入10种非恒等置换，V3穷举仅5种并单独记录；P2移除首层仿射中的显式关系加性贡献。
全部新增预测封存后才读GT64；GT128禁用。重复置换不当作独立案例或种子。
运行目录：`/scratch/djc/GeoPhy-Set3D/mrgr_geometry_utilization_v1`。
