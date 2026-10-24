"""Export immutable evaluation evidence and a Chinese report; no arrays/GT in Git."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as r
c = r.c
ROOT, PROJECT = r.ROOT, r.PROJECT
ART = PROJECT / 'docs/reports/artifacts/deep_baselines_test35_v1'
REPORT = PROJECT / 'docs/reports/deep_baselines_seed20261011_test35_v1_zh.md'


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def main():
    p = r.verify()
    acceptance = r.load('acceptance.json')
    assert acceptance['completed'] and acceptance['training_remains_paused']
    assert not ART.exists() and not REPORT.exists(), 'Preserve previous publication'
    ART.mkdir(parents=True)
    plain = c.verified(c.BASE / 'model_manifest.json')['models'][0]
    plain_summary = Path(plain['run']) / 'summary.json'
    assert c.sha(plain_summary) == plain['training_summary_sha256']
    p['models'][-1]['best_val_H'] = c.read(plain_summary)['best_val_H']
    shutil.copy2(plain_summary, ART / 'plain_training_summary.json')
    selected = ('protocol.json', 'cpu_checks.json', 'controller_launch.json', 'controller.log',
        'cpu_checks.log', 'engineering_regression.log', 'score.log', 'analysis.log', 'inference_manifest.jsonl',
        'prediction_manifest.json', 'all_predictions_sealed.json', 'GT64_access.json',
        'metrics.jsonl', 'per_case_metrics.csv', 'case_aggregated_metrics.csv',
        'summary.csv', 'paired_statistics.csv', 'statistics.json',
        'independent_statistics_verification.json', 'cost.csv', 'acceptance.json')
    for name in selected:
        shutil.copy2(ROOT / name, ART / name)
        checksum = ROOT / (name + '.sha256')
        if checksum.exists(): shutil.copy2(checksum, ART / checksum.name)
    for name in ('events.jsonl', 'resources.jsonl', 'seed11_test35_inference.json', 'seed11_test35_inference_attempt1.log'):
        (ART / 'scheduling').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scheduling' / name, ART / 'scheduling' / name)
    for arm in r.ARMS:
        source = r.TRAIN / f'{arm}_seed{r.SEED}'
        dest = ART / 'training' / source.name
        dest.mkdir(parents=True)
        for name in ('protocol.json', 'protocol.json.sha256', 'summary.json', 'status.json', 'metrics.jsonl', 'environment.json'):
            shutil.copy2(source / name, dest / name)
    for name in ('queue_status.json', 'user_pause_20261024.json'):
        shutil.copy2(r.b.ROOT / name, ART / name)
    paused = r.TRAIN / 'deepsets_seed20261019'
    for name in ('status.json', 'user_pause_20261024.json'):
        (ART / 'training/deepsets_seed20261019').mkdir(exist_ok=True)
        shutil.copy2(paused / name, ART / 'training/deepsets_seed20261019' / name)
    # An up-to-date status table; unrun Val G1--G5 values stay empty.
    tracking = []
    for m in p['models']:
        tracking.append(dict(method=m['method'], seed=r.SEED, parameters=m['parameters'],
            state='frozen_reused' if m['arm'] == 'plain' else 'completed_training_and_exploratory_test35_near_zero_collapse',
            committed_epochs=300, committed_updates=24900, best_epoch=m['best_epoch'],
            checkpoint_sha256=m['checkpoint_sha256'], evaluation='historical35cases_17settings_C1',
            independent_confirmation=False))
    for arm in ('deepsets', 'gnot'):
        tracking.append(dict(method=f'ray_{arm}_v1_seed20261019', seed=20261019,
            parameters=104577 if arm == 'deepsets' else 527109, state='paused_by_user',
            committed_epochs=240 if arm == 'deepsets' else 0,
            committed_updates=19920 if arm == 'deepsets' else 0, best_epoch=240 if arm == 'deepsets' else '',
            checkpoint_sha256='', evaluation='not_run', independent_confirmation=False))
    for seed in (20261011, 20261019):
        tracking.append(dict(method=f'ray_senseiver_v1_seed{seed}', seed=seed, parameters=254913,
            state='historical_scientific_NO_GO_no_formal_training', committed_epochs=0, committed_updates=0,
            best_epoch='', checkpoint_sha256='', evaluation='not_run', independent_confirmation=False))
    write_csv(ART / 'experiment_tracking.csv', tracking)
    costs = list(csv.DictReader((ROOT / 'cost.csv').open()))
    st = r.load('statistics.json')
    def stat(method, condition, metric):
        return next(x for x in st['summary'] if x['method'] == method and x['condition'] == condition and x['metric'] == metric)
    def paired(method, condition, metric='volume.relative_l1'):
        return next(x for x in st['paired'] if x['method'] == method and x['condition'] == condition and x['metric'] == metric)
    def fmt(v, precision=6):
        return '未定义' if v is None else f'{v:.{precision}f}'
    lines = [
        '# DeepSets / GNOT seed20261011：完成训练模型的探索性评价报告', '',
        f'生成时间：{c.now()}。执行前仓库 commit：`2dad11f220deec30e20806ffdac49490e17b4619`。', '',
        '**训练已按用户要求暂停，未自动恢复。两个 seed20261011 模型均完成300轮/24900次有效更新；本轮完成其既有35案例评价。两者输出均接近零，三维相对L1/L2约为1。工程运行完成不代表科学学习成功，不能将该结果解释为DeepSets/GNOT算法本身能力的普遍结论。**', '',
        '## 1. 暂停状态与固定模型', '',
        '2026-10-24 16:05:23（北京时间）停止本任务持久训练队列及其子训练进程组。DeepSets seed20261019权威checkpoint保留在epoch240/19920次有效更新，未完成300轮；GNOT seed20261019没有完整epoch提交。Senseiver沿用历史科学NO-GO，未启动新的正式训练。没有终止其他用户进程。', '',
        'seed20261011已完成模型的选择在本轮GT评分前固定：按全300轮验证留出H4 relative L1严格最小值选best，DeepSets为epoch291，GNOT为epoch300。二者训练/选模未访问体积GT；没有按test结果换epoch、seed、初始化、loss或架构。', '',
        '| 模型 | 参数 | 完整训练 | best epoch | 验证H4 L1 | best checkpoint SHA-256 |',
        '|---|---:|---|---:|---:|---|']
    for m in p['models']:
        lines.append(f"| {m['method']} | {m['parameters']} | 300 / 24900 | {m['best_epoch']} | {fmt(m.get('best_val_H'), 12)} | `{m['checkpoint_sha256']}` |")
    lines += ['', '主模型Plain保持冻结，无新增训练/微调/推理选模。代表性CPU C0/C1输出与冻结锚点逐位一致；主模型及第二种子参考权重哈希均通过核验。基线为独立射线适配版本，名称中的GNOT/DeepSets不是无修改的原论文任务：输入203维合法射线积分/几何token，输出完整FP32 64³质量分数，不共享主模型可训练表示。训练300轮、batch2、24900有效更新、AdamW、LR/cosine、loss权重、clip、A/B/H采样与冻结共同协议对齐。不同架构不宣称逐参数相同初始化。', '',
        '## 2. 评价范围、访问边界与统计方法', '',
        '沿用已冻结35案例、Pool A输入几何、预处理、input station IDs、heldout IDs和input tensor hashes。共17个设置：普通V4（`Val-A-V4`）和spread-V4各draw0；V3、V6、clustered-V4各draw0–4。每方法595份最终场。两基线共1190份新预测，Plain复用595份已有C1预测；总1785份全局封存后才由独立CPU评分进程读取35个授权GT64。', '',
        '仅单场基线最终输出与Plain C1比较；没有给基线虚构PRF前后C0/C1。Plain历史595份C1的全部正式标量指标重新计算并与原始结果核对。GT128和Pool B未读；评分阶段拒绝checkpoint、测量数组和几何数组，推理阶段拒绝GT。测试未用于checkpoint选择。预测XYZ、64³、FP32、有限数值及[0,1]范围均通过。', '',
        '先在每个案例/布局内对draw取算术均值，再对35案例做宏平均。95%区间使用20000次案例bootstrap、seed20261019；同案多个draw不作为独立案例。Δ=baseline−Plain，误差指标正值表示baseline更差，Dice负值表示baseline更差；近似不变阈值1e−8。区间未经多重比较校正，均为探索性；仅一个训练种子，不能估计跨种子不确定性。', '',
        '缺失/未定义指标不删除失败案例：逐案例CSV保留null与有效数；有任何案例未定义时，不生成伪装成完整35案例的CI。三维物理质心积分阈值沿用原正式实现，近零DeepSets可能触发未定义。相对自身极小峰值的Dice50/峰值坐标以及近零GNOT质心即使数学上可计算，也不代表恢复了有意义的浓度结构。', '',
        '**这些35案例已参与早期模型选择和诊断，仍为探索性评价，不是独立测试集确认。** 完整Val36×G1–G5×5draw=900清单与这里的35案例/17设置不同；本轮没有运行完整Val900，G5−G1/G5÷G1留空，不用V3/V6或布局类名替代几何档位。', '',
        '## 3. 主要重建结果', '',
        '下表为逐案例宏平均，数值为误差比值（0.1=10%），不是百分比数字。L1和L2越小越好，Dice50越大越好。', '',
        '| 布局 | 模型 | Relative L1 | Relative L2 | Dice50 | 物理质心有效案例 |',
        '|---|---|---:|---:|---:|---:|']
    for condition in c.CONDITIONS:
        for m in p['models'][::-1]:
            name = m['method']
            lines.append(f"| {condition} | {name} | {fmt(stat(name, condition, 'volume.relative_l1')['mean'], 9)} | {fmt(stat(name, condition, 'volume.relative_l2')['mean'], 9)} | {fmt(stat(name, condition, 'morphology.dice_0.5')['mean'])} | {stat(name, condition, 'volume.centroid_error_mm')['valid_cases']}/35 |")
    lines += ['', '配对L1差值如下（baseline−Plain）：', '',
        '| 布局 | baseline | Δ均值 | Δ中位数 | 案例95%CI | 增加/降低/基本不变 |',
        '|---|---|---:|---:|---|---|']
    for condition in c.CONDITIONS:
        for arm in r.ARMS:
            x = paired(r.method(arm), condition)
            interval = x['ci95']
            lines.append(f"| {condition} | {arm} | {fmt(x['mean'])} | {fmt(x['median'])} | [{fmt(interval[0])}, {fmt(interval[1])}] | {x['increased']}/{x['decreased']}/{x['unchanged']} |")
    lines += ['', '普通和聚簇V4的辅助指标：', '',
        '| 布局 | 模型 | 峰值相对幅度绝对误差 | 峰值绝对幅度误差（质量分数） | 峰值定位mm | 物理质心mm（有效数） | 观测L1 | 留出L1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition in ('Val-A-V4', 'clustered-V4'):
        for m in p['models'][::-1]:
            name = m['method']
            val = lambda metric: fmt(stat(name, condition, metric)['mean'])
            centroid = stat(name, condition, 'volume.centroid_error_mm')
            lines.append(f"| {condition} | {name} | {val('morphology.absolute_peak_ratio_error')} | {val('volume.peak_absolute_error')} | {val('volume.peak_location_error_mm')} | {fmt(centroid['mean'])} ({centroid['valid_cases']}/35) | {val('measurement.observed.relative_l1')} | {val('measurement.heldout.relative_l1')} |")
    lines += ['', '完整L1/L2、MAE、RMSE、峰值、两种既有质心定义、Dice、测量误差的均值/中位数/P90/P95/最大值/最差案例/配对CI见CSV。Dice的最差案例取最小值，其余误差取最大值。未定义量保留缺失数，不填0。', '',
        '## 4. 塌缩证据、成本与科学结论', '',
        '| baseline | 595份输出均值范围 | 输出最大值上界 | 完成训练耗时（小时） |',
        '|---|---|---:|---:|']
    pred = r.load('prediction_manifest.json')
    for m in p['models'][:2]:
        rr = [x for x in pred if x['method'] == m['method']]
        lines.append(f"| {m['arm']} | {min(x['output_mean'] for x in rr):.4e}–{max(x['output_mean'] for x in rr):.4e} | {max(x['output_max'] for x in rr):.4e} | {m['train_elapsed_seconds']/3600:.3f} |")
    lines += ['', '两种新基线都表现为接近零浓度场，验证H4误差约1与测试三维L1约1一致；这不是NaN/Inf数值崩溃。它们在当前固定自监督训练协议下没有形成有效重建，不能以“跑完300轮”称为成功基线，也不能将该负结果泛化为原始算法不适合所有射线重建任务。本文方法在本轮固定适配和协议下误差较低，但对两个塌缩基线的优势不等同于对充分调优、成功学习的原算法证明全面优越。', '',
        '目前只有seed20261011完成评价；不宣称跨种子稳定性。测试后未修复网络、缩短预算、增加搜索或换选模。最小后续研究应另行登记，仅用train/val诊断输出尺度、sigmoid饱和及测量条件利用；是否恢复剩余训练需遵循用户后续指示。本轮保持暂停。', '',
        '| 模型 | 参数变化（相对Plain） | 平均前向ms | 中位前向ms | P95 ms | CUDA峰值分配MiB | checkpoint字节 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in costs:
        lines.append(f"| {x['method']} | {x['delta_parameters']} | {float(x['mean_forward_ms']):.3f} | {float(x['median_forward_ms']):.3f} | {float(x['p95_forward_ms']):.3f} | {float(x['peak_allocated_MiB']):.2f} | {x['checkpoint_bytes']} |")
    lines += ['', '本轮仅GPU0/RTX4090执行推理，继承全局UUID flock、15秒双空闲快照、主机资源门槛和外来冲突监测；GPU1–3保留给其他用户任务。基线各有一次不计时warmup，595个布局/案例各做一次同步前向；计时不含诊断/I/O。Plain是历史硬件计时复用，冷启动/预热/缓存/负载条件未统一，因此此表不是受控延迟benchmark，不能凭此断言速度排名。峰值是模型前向阶段绝对CUDA allocated，非显卡总占用；checkpoint文件还含元数据，非纯权重字节。FLOPs/能耗未测。', '',
        '训练耗时取原引擎完成epoch累计秒数，不含中断epoch被丢弃计算及队列等待。24900指正式提交更新，不能声称物理尝试总次数恰为24900。历史GPU3任务间隙冲突、GPU0外来任务冲突及恢复记录全部保留，未给其他用户进程发信号。', '',
        '## 5. 验收、追溯与复算', '']
    independent = r.load('independent_statistics_verification.json')
    access = r.load('GT64_access.json')
    lines += [f"两基线完整预算及H4选模、主模型输出逐位回归、模型/config/source/input/checkpoint哈希、全部1785份预测及评分通过验收。595份Plain历史场重算的最大指标差为{access['maximum_old_metric_difference']:.3g}。独立数值算法从原始draw差值重建并通过multinomial计数权重及独立排序插值复核{independent['paired_vectors_checked']}个完整配对向量，最大差{independent['maximum_difference']:.3g}；未定义质心向量按协议不生成CI。", '',
        '本地原始输出：`/scratch/djc/GeoPhy-Set3D/deep_baselines_test35_v1/`；训练权重：`/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/training_registered_v2/<arm>_seed20261011/checkpoints/best.pt`。权重和1785份64³预测在本地保存，Git同步代码、冻结配置/协议、完整两模型训练指标日志、预测路径及SHA、逐draw/逐案例CSV、统计、GPU记录和报告，不上传GT数组。', '',
        '- [验收](artifacts/deep_baselines_test35_v1/acceptance.json) · [协议与源码/输入/模型身份](artifacts/deep_baselines_test35_v1/protocol.json)',
        '- [逐draw指标CSV](artifacts/deep_baselines_test35_v1/per_case_metrics.csv) · [案例聚合CSV](artifacts/deep_baselines_test35_v1/case_aggregated_metrics.csv)',
        '- [配对统计CSV](artifacts/deep_baselines_test35_v1/paired_statistics.csv) · [完整统计](artifacts/deep_baselines_test35_v1/statistics.json) · [独立统计复核](artifacts/deep_baselines_test35_v1/independent_statistics_verification.json)',
        '- [预测清单与SHA](artifacts/deep_baselines_test35_v1/prediction_manifest.json) · [全局封存](artifacts/deep_baselines_test35_v1/all_predictions_sealed.json) · [GT64访问审计](artifacts/deep_baselines_test35_v1/GT64_access.json)',
        '- [参数与计算成本](artifacts/deep_baselines_test35_v1/cost.csv) · [GPU调度记录](artifacts/deep_baselines_test35_v1/scheduling/events.jsonl) · [训练暂停记录](artifacts/deep_baselines_test35_v1/user_pause_20261024.json)',
        '- [更新后的实验追踪表](artifacts/deep_baselines_test35_v1/experiment_tracking.csv) · [DeepSets训练记录](artifacts/deep_baselines_test35_v1/training/deepsets_seed20261011/metrics.jsonl) · [GNOT训练记录](artifacts/deep_baselines_test35_v1/training/gnot_seed20261011/metrics.jsonl)', '',
        '复算入口：`scripts/deep_baselines_test35_v1/run.py`。register→cpu→launch/infer→score→analyze按独立进程分阶段，原始目录不可覆盖；若复现实验需更换独立输出目录并登记新的身份，不能将已封存目录重新执行。`deliver.py`仅从已验收证据生成报告/导出，不训练或更改选模。', '']
    REPORT.write_text('\n'.join(lines))
    # Explicitly update current tracker, while retaining old mini results and Val gaps.
    old_path = PROJECT / 'docs/reports/artifacts/deep_baselines_v1/formal_comparison_status.csv'
    old = list(csv.DictReader(old_path.open()))
    for row in old:
        arm, seed = row['method'], int(row['seed'])
        if arm in r.ARMS and seed == r.SEED:
            row.update(status='completed300_test35_near_zero_collapse', committed_epochs=300, committed_updates=24900,
                       evaluation='not_run_900_val; completed_exploratory_test35_17settings')
        elif arm in r.ARMS:
            row.update(status='paused_by_user', committed_epochs=240 if arm == 'deepsets' else 0,
                       committed_updates=19920 if arm == 'deepsets' else 0, evaluation='not_run_900_val; test_not_run')
        elif arm == 'Plain_Set_Interaction':
            row['evaluation'] = 'not_run_900_val; frozen_historical_test35_reused'
    write_csv(old_path, old)
    print(str(REPORT))


if __name__ == '__main__':
    main()
