# 数据管理

实际数据存放在 `/data/djc/GeoPhy-Set3D/data/`，项目通过符号链接访问：

- `raw/` → `/data/djc/GeoPhy-Set3D/data/raw/`
- `processed/` → `/data/djc/GeoPhy-Set3D/data/processed/`

Fluent 生成并导出的原始交付数据位于：
`raw/Fluent/static_final_frame_partial_v1/`。
包含观测数组、三维标签、逐样本元数据和索引，共 237 个样本。
详细说明见 `docs/data-layout.md`。

原始数据不直接覆盖；数据集说明指出部分数组使用硬链接，应只读使用。
处理结果写入 `processed/`，并记录输入版本、处理参数、单位和坐标定义。

实际数据与本机符号链接已被 Git 忽略。克隆项目到其他机器后，需要重新配置数据路径或创建链接。
