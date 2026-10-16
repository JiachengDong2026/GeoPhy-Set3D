# MRGR 轻量多尺度版：300 轮正式训练启动记录

已于 **2026-10-16T04:37:08.805999+00:00（UTC）** 启动单种子完整训练，使用物理 **GPU0 / RTX 4090**，后台 PID **838310**。该卡启动时负载最低，但仍有其他用户进程，不视为独占资源。进程已脱离当前会话，关闭对话不会终止训练。

**配置：** 沿用通过20轮筛查的12通道多尺度上下文结构，参数344,678；种子20261011，从相同随机初始化重新开始，不复用筛查或历史权重。166训练/36验证案例，batch2，每轮83次更新，300轮共24,900次更新；FP32、关闭TF32，AdamW lr2e-4 / wd1e-4 / clip1，cosine T_max=300。原A/B/H采样和自监督损失不变，没有提前停止。

**选模：** 每轮固定Val-A留出投影相对L1，严格最小者保存为best.pt；不按三维GT或困难布局选模。训练不读取体积GT、最终test或历史权重。完成训练后需独立封存预测并开展普通/多布局验证三维评价，当前尚未进行该评价。

**启动核验：** 116个注册源文件和205个数据文件内容核对通过；首轮学习率、站点摘要及loader随机状态与已完成筛查一致。已读取检查点确认模型有限、优化器/调度器/RNG完整、最佳权重一致，训练访问审计禁止读取计数均为0。CPU持久worker恢复测试通过；并非完整GPU逐数值恢复比较。

**本记录生成时已完成 3/300 轮**（最新状态以status.json为准）。

| 轮数 | 训练+验证秒/轮 | 验证投影相对L1 |
|---|---:|---:|
| 1 | 34.32 | 159.8193% |
| 2 | 33.37 | 85.6970% |
| 3 | 33.65 | 46.7401% |

按当前早期完整轮次线性外推，300轮约 **2.79小时**，后续采样和共享GPU负载可能改变速度，且写盘/绘图也有额外开销。初始误差仍处于快速下降阶段，不作最终效果判断。

**文件：**

- [实时状态](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/status.json)；[逐轮指标](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/metrics.jsonl)；[控制台日志](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/console.log)。
- [训练曲线](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/curves.png)（首轮及每5轮更新）；[注册协议](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/protocol.json)；[启动检查](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/startup_verification.json)。
- [最佳模型](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/best.pt)；[最新可恢复检查点](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/checkpoints/last.pt)；每25轮另保存模型快照。
- [独立训练入口](../../scripts/train_mrgr_multiscale_formal_v1.py)；[前期筛查报告](mrgr_multiscale_v1_zh.md)。

**中断恢复：** 仅在确认原训练进程已退出后，在仓库根目录执行下列命令。进程锁阻止两个训练同时写同一目录；恢复以last.pt为已提交轮次，修复指标和best.pt，并恢复模型、优化器、调度器、随机状态及采样顺序。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /scratch/djc/envs/gas-torch/bin/python -u scripts/train_mrgr_multiscale_formal_v1.py train --resume
```
