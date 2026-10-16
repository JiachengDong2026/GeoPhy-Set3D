#!/usr/bin/env python3
"""Compare guarded Plan A speed reports and every numerical artifact element.

CPU-only reporting; no model, dataset, or GT is opened, and checkpoints are
never deserialized. Optional acceptance evidence verifies checkpoint byte hashes.
Tolerances are engineering FP32 checks, not scientific accuracy acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {
    'volumes': dict(rtol=1e-4, atol=1e-6),
    'heldout_predictions': dict(rtol=1e-4, atol=1e-6),
    'clipped_gradients': dict(rtol=1e-3, atol=1e-6),
    'post_update_state': dict(rtol=1e-4, atol=1e-6),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def timed_summary(rows):
    values = [row['seconds'] for row in rows]
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), repeats=len(values), values=values)


def load_report(path):
    report = json.loads(path.read_text())
    if report.get('protocol') != 'plan_a_full_update_speed_v2' or report.get('status') != 'passed':
        raise ValueError(f'Need a completed passed benchmark: {path}')
    if report.get('precision') != 'fp32' or report.get('tf32_matmul') or report.get('tf32_cudnn'):
        raise ValueError(f'FP32 with TF32 disabled required: {path}')
    artifact_path = Path(report['numerical_artifact']['path'])
    if not artifact_path.is_file():
        artifact_path = path.with_suffix('.numerics.pt')
    if sha(artifact_path) != report['numerical_artifact']['sha256']:
        raise ValueError(f'Numerical artifact hash mismatch: {artifact_path}')
    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
    return dict(path=str(path.resolve()), sha256=sha(path), raw=report, artifact=artifact,
                artifact_path=str(artifact_path.resolve()))


def canonical_branches(values):
    """Normalize [A1,B1,A2,B2] and [A12,B12] to equal case-major branches."""
    if len(values) % 2 or not values:
        raise ValueError('Expected alternating A/B tensors')
    return {branch: torch.cat(values[offset::2], dim=0)
            for branch, offset in (('A', 0), ('B', 1))}


def compare_tensors(reference, candidate, *, rtol, atol):
    keys_equal = set(reference) == set(candidate)
    rows = {}
    sum_square_diff = sum_square_reference = 0.
    total = failures = 0
    max_abs = 0.
    exact = True
    finite = True
    shapes_equal = True
    for key in sorted(set(reference) & set(candidate)):
        first, second = reference[key], candidate[key]
        if first.shape != second.shape or first.dtype != second.dtype:
            rows[key] = dict(shape_or_dtype_mismatch=True,
                             reference_shape=list(first.shape), candidate_shape=list(second.shape),
                             reference_dtype=str(first.dtype), candidate_dtype=str(second.dtype))
            shapes_equal = False
            exact = False
            continue
        first, second = first.double().reshape(-1), second.double().reshape(-1)
        difference = (first - second).abs()
        is_finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
        num_failed = int((difference > atol + rtol * first.abs()).sum())
        dsq, rsq = float(difference.square().sum()), float(first.square().sum())
        maximum = float(difference.max()) if difference.numel() else 0.
        is_exact = bool(torch.equal(first, second))
        rows[key] = dict(elements=first.numel(), max_abs=maximum,
                         relative_l2=math.sqrt(dsq / max(rsq, 1e-300)),
                         outside_tolerance=num_failed, exact=is_exact, finite=is_finite)
        if num_failed:
            indices = (difference > atol + rtol * first.abs()).nonzero().flatten()[:5]
            rows[key]['violation_examples'] = [dict(
                flat_index=int(index), reference=float(first[index]), candidate=float(second[index]),
                absolute_difference=float(difference[index]),
                allowed_difference=float(atol + rtol * first[index].abs())) for index in indices]
        sum_square_diff += dsq
        sum_square_reference += rsq
        total += first.numel()
        failures += num_failed
        max_abs = max(max_abs, maximum)
        exact &= is_exact
        finite &= is_finite
    passed = keys_equal and shapes_equal and finite and failures == 0
    return dict(passed=passed, exact=exact and keys_equal and shapes_equal,
                keys_equal=keys_equal, shapes_and_dtypes_equal=shapes_equal, finite=finite,
                reference_only_keys=sorted(set(reference) - set(candidate)),
                candidate_only_keys=sorted(set(candidate) - set(reference)),
                rtol=rtol, atol=atol, elements=total, max_abs=max_abs,
                difference_l2=math.sqrt(sum_square_diff), reference_l2=math.sqrt(sum_square_reference),
                relative_l2=math.sqrt(sum_square_diff / max(sum_square_reference, 1e-300)),
                outside_tolerance=failures, tensors=rows)


def gpu_summary(report):
    samples = report.get('gpu_load_samples', [])
    devices, process_sets, utilizations, memories = set(), set(), [], []
    selected_index = report.get('device', 'cuda:0').split(':')[-1]
    for sample in samples:
        target_uuid = None
        for line in sample.get('gpus', '').splitlines():
            columns = [column.strip() for column in line.split(',')]
            if len(columns) < 7 or columns[0] != selected_index:
                continue
            target_uuid = columns[1]
            devices.add(target_uuid)
            try:
                memories.append(float(columns[4]))
                utilizations.append(float(columns[5]))
            except ValueError:
                pass
        for line in sample.get('processes', '').splitlines():
            columns = [column.strip() for column in line.split(',')]
            if len(columns) >= 4 and columns[0] == target_uuid:
                process_sets.add((columns[1], columns[2]))
    return dict(samples=len(samples), device_uuids=sorted(devices),
                observed_processes=[dict(pid=pid, executable=exe) for pid, exe in sorted(process_sets)],
                utilization_percent_range=[min(utilizations), max(utilizations)] if utilizations else None,
                total_device_memory_mib_range=[min(memories), max(memories)] if memories else None,
                exclusive_gpu_established=False,
                limitation='nvidia-smi records concurrent processes. Logical CUDA index may differ from physical index when CUDA_VISIBLE_DEVICES remaps devices; exclusive throughput is not established.')


def public_report(loaded):
    raw = loaded['raw']
    audit = raw.get('gt_read_audit', {})
    return dict(path=loaded['path'], sha256=loaded['sha256'], artifact_path=loaded['artifact_path'],
                numerical_artifact_sha256=raw['numerical_artifact']['sha256'],
                train=timed_summary(raw['train']), validation=timed_summary(raw['validation']),
                train_warmup=raw.get('train_warmup'), validation_warmup=raw.get('validation_warmup'),
                peak_train_allocated_gib=max(row['memory'].get('peak_allocated_bytes', 0) for row in raw['train']) / 2**30,
                microbatch=raw['microbatch'], gradient_accumulation=raw['gradient_accumulation'],
                diagnostics=raw['diagnostics'], resolved_model=raw['resolved_model'],
                input_sha256=raw['input_sha256'], initial_weights_sha256=raw['initial_weights_sha256'],
                source_sha256=raw['source_sha256'], config_sha256=raw['config_sha256'],
                environment=raw['environment'], gt_read_audit=audit,
                gt_audit_passed=not any(audit.get(key, 1) for key in (
                    'gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')),
                gpu_load=gpu_summary(raw))


def comparison(baseline, candidate):
    first, second = baseline['raw'], candidate['raw']
    initial = compare_tensors(baseline['artifact']['initial_state'], candidate['artifact']['initial_state'], rtol=0., atol=0.)
    result = dict(candidate_path=candidate['path'],
                  identical_inputs=first['input_sha256'] == second['input_sha256'],
                  identical_data_hashes=first['data_file_sha256'] == second['data_file_sha256'],
                  identical_initial_weight_hash=first['initial_weights_sha256'] == second['initial_weights_sha256'],
                  initial_weights=initial, phases={})
    for phase in ('train', 'validation'):
        metrics = {}
        for field, tolerance in TOLERANCES.items():
            if phase == 'validation' and field in ('clipped_gradients', 'post_update_state'):
                continue
            before, after = baseline['artifact'][phase][field], candidate['artifact'][phase][field]
            if isinstance(before, list):
                before, after = canonical_branches(before), canonical_branches(after)
            metrics[field] = compare_tensors(before, after, **tolerance)
        before, after = timed_summary(first[phase]), timed_summary(second[phase])
        result['phases'][phase] = dict(
            numerical=metrics,
            baseline_loss=baseline['artifact'][phase]['loss'], candidate_loss=candidate['artifact'][phase]['loss'],
            baseline_seconds=before, candidate_seconds=after,
            median_speedup=before['median'] / after['median'],
            observed_extreme_ratio_range=[before['minimum'] / after['maximum'], before['maximum'] / after['minimum']],
            range_note='Extreme ratios combine observed min/max; this is not a confidence interval or paired measurement.')
    result['numerical_passed'] = (
        result['identical_inputs'] and result['identical_data_hashes'] and result['identical_initial_weight_hash']
        and initial['exact'] and all(metric['passed'] for phase in result['phases'].values()
                                           for metric in phase['numerical'].values()))
    return result


def epoch_result(path):
    raw = json.loads(path.read_text())
    if raw.get('protocol') != 'plan_a_cold_full_epoch_speed_v2':
        raise ValueError(f'Unrecognized full epoch benchmark: {path}')
    result = dict(path=str(path.resolve()), sha256=sha(path), status=raw.get('status'),
                  phases=raw.get('phases', {}), gradient_check=raw.get('gradient_check'),
                  gt_read_audit=raw.get('gt_read_audit'), gpu_load=gpu_summary(raw),
                  provenance=raw.get('provenance'), timing_policy=raw.get('timing_policy'),
                  os_file_cache_policy=raw.get('os_file_cache_policy'),
                  outer_operator_is_model_operator=raw.get('outer_operator_is_model_operator'),
                  resolved_config=raw.get('resolved_config'), elapsed_seconds=raw.get('elapsed_seconds'))
    train = raw.get('phases', {}).get('train', {})
    validation = raw.get('phases', {}).get('validation', {})
    full_membership = (train.get('metrics', {}).get('sample_count') == 166
                       and validation.get('metrics', {}).get('sample_count') == 36
                       and train.get('metrics', {}).get('optimizer_updates') == 83
                       and train.get('metrics', {}).get('complete_epoch')
                       and validation.get('metrics', {}).get('complete_epoch'))
    result['full_166_train_36_val_completed'] = bool(raw.get('status') == 'passed' and full_membership)
    if result['full_166_train_36_val_completed']:
        seconds = train['seconds'] + validation['seconds']
        result.update(train_plus_validation_seconds=seconds,
                      peak_allocated_gib=max(phase.get('cuda', {}).get('peak_allocated_bytes', 0)
                                             for phase in (train, validation)) / 2**30,
                      extrapolated_300_epochs_hours=seconds * 300 / 3600,
                      extrapolated_three_seeds_serial_hours=seconds * 900 / 3600,
                      extrapolation_limitation='A linear projection from one shared-GPU epoch with fresh weights/optimizer; not an exclusive-GPU result, confidence interval, convergence budget or observed 300-epoch run. Includes train/validation runner metrics, excludes setup and research checkpoint/freeze serialization; future sampling, cache misses and load can change throughput.')
    else:
        result['error'] = raw.get('error', 'Full 166/36 completed epoch evidence not available; no long-run projection made')
    return result


def run_evidence(path):
    """Read scalar run records and verify byte hashes, without loading weights."""
    names = ('training_summary.json', 'acceptance_result.json', 'gt_read_audit.json',
             'freeze.json', 'provenance.json', 'measurement_dependence.json', 'failure.json')
    loaded, files = {}, {}
    for name in names:
        file = path / name
        if file.is_file():
            loaded[name] = json.loads(file.read_text())
            files[name] = sha(file)
    result = dict(path=str(path.resolve()), record_file_sha256=files,
                  training_summary=loaded.get('training_summary.json'),
                  acceptance=loaded.get('acceptance_result.json'),
                  audit=loaded.get('gt_read_audit.json'),
                  measurement_dependence=loaded.get('measurement_dependence.json'),
                  failure=loaded.get('failure.json'))
    provenance = loaded.get('provenance.json', {})
    result['provenance_hashes'] = {key: provenance.get(key) for key in
                                 ('source_sha256', 'data_sha256', 'config_sha256')}
    code = provenance.get('code', {})
    changed = [name for name, expected in code.items()
               if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
    result['source_matches_current'] = bool(code) and not changed
    result['source_mismatch_paths'] = changed
    freeze = loaded.get('freeze.json', {})
    result['freeze'] = freeze
    checks = {}
    for name, field in (('best.pt', 'best_sha256'), ('last.pt', 'last_sha256')):
        checkpoint = path / 'checkpoints' / name
        expected = freeze.get(field)
        if expected:
            checks[name] = dict(expected_sha256=expected, exists=checkpoint.is_file(),
                                sha256=sha(checkpoint) if checkpoint.is_file() else None)
            checks[name]['matches'] = checks[name]['sha256'] == expected
    result['checkpoint_byte_hashes'] = checks
    result['completed'] = bool(result['training_summary'] and result['training_summary'].get('completed'))
    result['acceptance_passed'] = bool(result['acceptance'] and result['acceptance'].get('passed'))
    return result


def markdown(summary):
    base = summary['baseline']
    lines = ['# 方案 A 实现优化与速度实测', '',
        '本报告比较相同真实 Pool A 测量、相同初始权重下的完整 A/B/H 训练更新与验证计算。所有计算保持 FP32，并关闭 TF32。', '',
        '**设备为共享 RTX 4090，未取得独占卡吞吐结果。** 同期 `nvidia-smi` 负载与进程已保存在原始 JSON 中；不同时间的共享负载会影响速度比。', '',
        '## 测速协议与适用范围', '',
        '- A2、K=4、Q=96、lift32、64³ 输出、prior width32；有效 batch=2。',
        '- 每个训练计时包含完整双分支物理展开、A/B/H 外部损失、反向、实际案例梯度平均、clip 与 AdamW 更新。',
        '- CPU collation 与传输包含在计时内，磁盘读取、模型/优化器重置、数值证据拷贝排除。',
        '- `benchmark_plan_a_speed.py` 单独创建外部投影 operator；优化后的真实 runner 复用 `model.operator`。二者的几何缓存归属及命中情况有差异。',
        '- 快测速包含 objective 所请求的逐步 diagnostics，但不包含真实 runner 的指标汇总、标量记录、快照及日志工作；实际 166 案例 epoch 验收需补齐这些开销。',
        '- 至少一次预热与至少两次正式计时；报告中位数及观测最小/最大值。',
        '- 每次从相同初始权重、新 AdamW 开始；包含首次更新的 moment 初始化。几何缓存与 CUDA kernels 在预热后保持热状态。',
        '- 固定两个训练案例和两个验证案例，仅表示该输入下的工程吞吐；不能直接等同 166 案例 epoch、36 案例验证或后期权重下的训练耗时。',
        '- 本测速脚本自身不生成验收；真实 smoke/gate 复验另列。未开展 300 轮长训练或 GT 三维精度评价。', '',
        '## 实测速度', '',
        '| 配置 | 训练更新中位数（最小–最大）秒 | 两案例验证中位数（最小–最大）秒 | 训练/验证中位加速 | 峰值训练显存 GiB |',
        '|---|---:|---:|---:|---:|']

    def time_cell(row):
        return f"{row['median']:.3f}（{row['minimum']:.3f}–{row['maximum']:.3f}）"

    lines.append(f"| 基线 | {time_cell(base['train'])} | {time_cell(base['validation'])} | 1.00× / 1.00× | {base['peak_train_allocated_gib']:.2f} |")
    for name, candidate in summary['candidates'].items():
        comp = summary['comparisons'][name]
        lines.append(f"| {name} | {time_cell(candidate['train'])} | {time_cell(candidate['validation'])} | {comp['phases']['train']['median_speedup']:.2f}× / {comp['phases']['validation']['median_speedup']:.2f}× | {candidate['peak_train_allocated_gib']:.2f} |")
    lines += ['', '最小/最大极值速度比范围保存在 summary.json 中；它们不是置信区间，也不是配对速度统计。', '',
              '## FP32 数值一致性', '',
              '输入张量 hash、实际数据文件 hash 和初始参数必须完全一致；初始参数另做逐元素精确比较。容差在比较工具中固定：输出与 H 预测 `rtol=1e-4, atol=1e-6`，完整 clip 后梯度 `rtol=1e-3, atol=1e-6`，更新后参数 `rtol=1e-4, atol=1e-6`。逐元素条件为 `|candidate-reference| ≤ atol + rtol·|reference|`。', '',
              '以下 relative L2 使用每组全部张量元素，未抽样；完整梯度包含全部有梯度参数。它们衡量实现一致性，不能证明长训练收敛或科学精度等价。', '',
              '| 候选 | 阶段 / 张量 | relative L2 | 最大绝对差 | 超差元素 / 总元素 | 检查 |',
              '|---|---|---:|---:|---:|---|']
    for name, comp in summary['comparisons'].items():
        for phase_name, phase in comp['phases'].items():
            for field, metric in phase['numerical'].items():
                lines.append(f"| {name} | {phase_name} / {field} | {metric['relative_l2']:.3e} | {metric['max_abs']:.3e} | {metric['outside_tolerance']} / {metric['elements']} | {'通过' if metric['passed'] else '未通过'} |")
        lines += ['', f"{name}：输入与数据 hash {'一致' if comp['identical_inputs'] and comp['identical_data_hashes'] else '不一致'}；初始参数逐元素{'一致' if comp['initial_weights']['exact'] else '不一致'}；整体数值检查{'通过' if comp['numerical_passed'] else '未通过'}。", '']
    if summary.get('baseline_repeat'):
        repeat = summary['baseline_repeat']
        comp = summary['baseline_repeat_comparison']
        lines += ['## 基线复测', '',
                  f"基线复测训练中位数 {repeat['train']['median']:.3f} 秒，验证中位数 {repeat['validation']['median']:.3f} 秒；原基线/复测中位比为 {comp['phases']['train']['median_speedup']:.2f}× / {comp['phases']['validation']['median_speedup']:.2f}×，反映同期共享环境波动，不能全部归因于代码。", '']
    if summary.get('epoch_result'):
        epoch = summary['epoch_result']
        lines += ['## 完整真实数据一轮实测', '']
        if epoch['full_166_train_36_val_completed']:
            train, validation = epoch['phases']['train'], epoch['phases']['validation']
            lines += [f"使用真实 runner 完成 166 个训练案例 / 83 次更新，以及全部 36 个验证案例。训练 {train['seconds']:.2f} 秒，验证 {validation['seconds']:.2f} 秒，合计 **{epoch['train_plus_validation_seconds']/60:.2f} 分钟/轮**；峰值已分配显存 {epoch['peak_allocated_gib']:.2f} GiB。", '',
                f"按本次单轮耗时线性外推：300 轮约 **{epoch['extrapolated_300_epochs_hours']:.2f} 小时**，三个种子单卡串行约 **{epoch['extrapolated_three_seeds_serial_hours']:.2f} 小时**。这是单次共享负载下的预算估计，尚未实测独占 GPU 或完整 300 轮。", '',
                '该轮从新权重、新 AdamW 和空模型几何缓存开始；操作系统文件缓存未清空，计时前 hash 读取可能预热文件。计时包含实际数据读取、完整双分支更新和 runner 诊断/指标汇总，外层投影复用 model.operator。模型设置、采样顺序、源代码/data hash、梯度检查和同期 GPU 负载均留存。', '',
                '外推排除了 setup 和正式 runner checkpoint/freeze 序列化；未来视角采样、缓存淘汰、模型权重及共享负载都会改变耗时，不能以两案例热缓存结果替代该实测。', '']
        else:
            lines += [f"完整 epoch 证据状态：{epoch['status']}。{epoch.get('error')}；不进行 300 轮外推。", '']
        lines += [f"原始证据：`{epoch['path']}`。", '']
    if summary.get('acceptance_runs'):
        lines += ['## 优化后真实 smoke / gate 复验', '',
                  '复验读取既有运行记录与 hash，不反序列化 checkpoint 权重或打开 GT。其通过条件与工程逐元素数值对比条件分别报告。', '',
                  '| 运行 | 完成轮数 / 更新数 | 初始 / 最佳 H 相对 L1 | 验收 | 当前源码匹配 | checkpoint 字节 hash |',
                  '|---|---:|---:|---|---|---|']
        for name, run in summary['acceptance_runs'].items():
            training, acceptance = run.get('training_summary') or {}, run.get('acceptance') or {}
            initial, best = acceptance.get('initial_heldout_relative_l1'), acceptance.get('best_heldout_relative_l1')
            score = f'{initial:.6f} / {best:.6f}' if initial is not None and best is not None else '待完成'
            hashes = run['checkpoint_byte_hashes']
            hash_note = '一致' if hashes and all(value['matches'] for value in hashes.values()) else '缺失或不一致'
            lines.append(f"| {name} | {training.get('completed_epochs', '?')} / {training.get('optimizer_updates', '?')} | {score} | {'通过' if run['acceptance_passed'] else '未通过或未完成'} | {'一致' if run['source_matches_current'] else '缺失或不一致'} | {hash_note} |")
        lines += ['', '各次禁读审计、测量依赖性检查、完整验收条目、源码/data/config/checkpoint hash 见 summary.json；原始运行目录如下。', '']
        lines += [f"- {name}：`{run['path']}`。" for name, run in summary['acceptance_runs'].items()]
        lines += ['']
    lines += ['## 配置与证据', '']
    for name, report in [('基线', base), *summary['candidates'].items()]:
        lines += [f"- {name}：microbatch={report['microbatch']}，accum={report['gradient_accumulation']}，checkpoint_prior={report['resolved_model'].get('checkpoint_prior')}，ray_chunk={report['resolved_model'].get('ray_chunk')}，diagnostics={report['diagnostics']}；禁读审计{'通过' if report['gt_audit_passed'] else '未通过'}。",
                  f"  原始报告：`{report['path']}`；数值证据：`{report['artifact_path']}`。"]
    lines += ['', '完整时间样本、逐张量误差、环境、hash、源文件 hash 和审计见 `artifacts/plan_a_perf_v2/summary.json`。旧 `plan_a_v1` 验收报告保持原样；修改后的源码不能复用旧源码 hash 的正式资格。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, action='append', required=True,
                        help='Repeat for multiple completed optimization candidates')
    parser.add_argument('--baseline-repeat', type=Path)
    parser.add_argument('--epoch-result', type=Path, help='Completed full 166 train / 36 val epoch benchmark JSON')
    parser.add_argument('--smoke-run', type=Path, help='Optimized real smoke run directory, scalar/hash read only')
    parser.add_argument('--gate-run', type=Path, help='Optimized real gate run directory, scalar/hash read only')
    parser.add_argument('--report', type=Path, default=ROOT/'docs/reports/plan_a_perf_v2_zh.md')
    parser.add_argument('--summary', type=Path, default=ROOT/'docs/reports/artifacts/plan_a_perf_v2/summary.json')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace this v2 generated report')
    args = parser.parse_args()
    if not args.overwrite and (args.report.exists() or args.summary.exists()):
        parser.error('Report or summary exists; use fresh paths or --overwrite')
    if args.report.name == 'plan_a_v1_zh.md' or 'plan_a_v1' in args.summary.parts:
        parser.error('Historical plan_a_v1 reports cannot be replaced')
    torch.set_num_threads(2)
    baseline = load_report(args.baseline)
    result = dict(protocol='plan_a_full_update_speed_comparison_v2',
                  tolerance_scope='Fixed engineering FP32 elementwise comparison; no training trajectory or accuracy equivalence claim',
                  tolerances=TOLERANCES, baseline=public_report(baseline),
                  candidates={}, comparisons={})
    for path in args.candidate:
        name = path.stem
        if name in result['candidates']:
            raise ValueError(f'Duplicate candidate name {name}')
        loaded = load_report(path)
        result['candidates'][name] = public_report(loaded)
        result['comparisons'][name] = comparison(baseline, loaded)
    if args.baseline_repeat:
        loaded = load_report(args.baseline_repeat)
        result['baseline_repeat'] = public_report(loaded)
        result['baseline_repeat_comparison'] = comparison(baseline, loaded)
    if args.epoch_result:
        result['epoch_result'] = epoch_result(args.epoch_result)
    if args.smoke_run or args.gate_run:
        result['acceptance_runs'] = {}
        for name, path in (('smoke', args.smoke_run), ('gate', args.gate_run)):
            if path:
                result['acceptance_runs'][name] = run_evidence(path)
    all_comparisons = list(result['comparisons'].values())
    if args.baseline_repeat:
        all_comparisons.append(result['baseline_repeat_comparison'])
    result['all_numerical_checks_passed'] = all(comp['numerical_passed'] for comp in all_comparisons)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    args.report.write_text(markdown(result))
    print(json.dumps(dict(report=str(args.report), summary=str(args.summary),
                          all_numerical_checks_passed=result['all_numerical_checks_passed'])))


if __name__ == '__main__':
    main()
