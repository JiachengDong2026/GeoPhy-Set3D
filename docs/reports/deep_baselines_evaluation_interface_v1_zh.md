# 深度学习基线统一验证评价入口 v1

入口 `scripts/deep_baselines_v1/evaluate.py` 使用已保存的 `docs/reports/artifacts/selfsup_v1_c0c1_geometry/station_subsets.json`：36 个 **val** 案例 × G1–G5 × 5 draw = 900 组，观测 V4 与 H4 不重叠，顺序完全沿用清单。输出同为 64³ CH4 质量分数。只允许 Val Pool A；GT128、Pool B 和 test 不在本入口许可范围。

该入口支持 `plain`、`deepsets`、`senseiver`、`gnot`，不训练模型、不启动后台进程、不管理 GPU。调用者须先取得仓库规定的 GPU 资源锁和双快照许可，再运行 GPU 推理。主模型只允许 `main_anchor.json` 对应的冻结最终 Plain checkpoint，最终单场为 C1；三个基线是各自的单场输出，不将其虚构为有 PRF 的 C0/C1。

## 分阶段边界

1. `register`：必须给定已选 checkpoint、SHA256 和预先存在的 JSON 选模记录；选模记录必须包含相同的 `checkpoint_sha256`（主模型为 `main_anchor.json` 中的 `model.checkpoint_sha256`）。冻结架构、源代码、固定 900 清单、数据来源索引、训练标尺、Val 案例表和 Pool A 几何哈希。输出目录必须为空。
2. `infer`：验证上述身份；使用 `DiagnosticReadGuard` 禁止 GT；读取合法测量完成逐组推理，记录输入/H 全 tensor 哈希、参数量、单次前向时间、前向峰值显存、输出范围及标准差、近常量标记、观测和留出测量误差，保存 NPZ 和每组 JSON，全部保存后 seal。已有完整 JSON+NPZ 经哈希验证后可复用；孤立预测文件不会被覆盖，需人工核查恢复身份。
3. `gt`：必须已存在完整 seal。只读取已保存预测和 Val GT64；GT 阶段不读取 checkpoint 或测量数组。使用原有 `VolumeMetrics`，保存每 draw 指标、访问审计和完成记录。主模型 GT64 的历史科学非独立性不因本次重新评分而改变。
4. `stats`：先对同一案例同一 G 档的 5 draw 求平均，再进行案例宏平均、median、P90、P95、最大值和最差案例报告。10000 次案例 bootstrap，seed `2026101801`。G5−G1 为案例配对误差差；G5/G1 为两组 grand mean 的比值，在每个 bootstrap 样本内重新求比，分母为零时输出 null。
5. `compare`：将一个基线 run 与冻结 Plain run 配对，严格核对每 draw 案例、站点顺序、输入和 H 哈希。Δ 定义为基线−Plain，每 seed 独立报告。若基线 seed19 对比固定 seed11 Plain，明确记录 `same_training_seed=false`，不误称同种子对照。区间为探索性边际 95% CI，不是多重比较校正后的确认性检验。

所有写入采用 exclusive 创建；已有结果和正式协议不得被覆盖。已冻结数值源若发生改变会停止，不静默沿用过期注册。注册应在本轮所有实现文件写定后执行。

## 小规模集成选项

`register --mini` 固定选择 `case_index=0,1`、每个 G 档 `draw=0`，共 10 组。标识为 `NONFORMAL_MINI_INTEGRATION`，不能称为正式 900 组比较；`formal=false` 的 pilot checkpoint 只允许该选项。此选项用于验证代码和数据边界，不能用于架构或超参数搜索。

```bash
PY=/scratch/djc/envs/gas-torch/bin/python
# 以下路径/哈希由实际已选 checkpoint 和选模记录填写。
$PY scripts/deep_baselines_v1/evaluate.py register --run RUN_DIR \
  --arm deepsets --seed 20261011 --checkpoint SELECTED_BEST_PT \
  --checkpoint-sha256 EXACT_SHA256 --selection-record SELECTED_JSON --mini
$PY scripts/deep_baselines_v1/evaluate.py infer --run RUN_DIR --device cuda:0
$PY scripts/deep_baselines_v1/evaluate.py gt --run RUN_DIR
$PY scripts/deep_baselines_v1/evaluate.py stats --run RUN_DIR
$PY scripts/deep_baselines_v1/evaluate.py compare --run RUN_DIR --reference-run PLAIN_RUN_DIR
```

`forward_seconds` 为同步的单次前向，排除额外 Pdiff 测量诊断和 I/O，主模型内部物理反馈仍计入自身前向；没有 warmup/repeats，不能把该字段冒充稳定性能基准。显存为模型前向结束前的 CUDA allocated 峰值，包含模型驻留张量。评价脚本保留每个训练 seed 的身份，不把多 seed 或多 draw 混作独立案例。

## 已执行与未执行

本子任务执行了 6 项 CPU 测试，均通过：冻结 900 清单结构；先 draw 后案例聚合；使用独立的 bootstrap 计数矩阵复核均值 CI；grand mean 几何退化比；缺失、重复、重叠 H、非有限值的拒绝；零分母的 null；输入/H 身份核对与配对差值统计。`--help` CLI 导入检查通过。

本子任务 **未执行**真实测量推理、GPU 操作或 GT64 读取。真实四模型 mini、GPU 锁调度、checkpoint provenance 及 read-guard 集成仍需父任务执行后才能称入口完成端到端验证；900 组评价须等正式 checkpoint 固定后运行。未完成的模型重建指标应保留空白。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  /scratch/djc/envs/gas-torch/bin/python -m unittest discover \
  -s tests -p test_deep_baselines_evaluate_v1.py -v
```
