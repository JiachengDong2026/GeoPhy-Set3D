#!/usr/bin/env python3
"""Rebuild the Plan A evidence index/report from scalar artifacts, without training.

Run directories are read-only inputs. Checkpoints and diagnostic tensors are never
loaded or copied. Re-running replaces only this script's report, plots and scalar
archives; missing or incomplete acceptance gates remain explicit in the report.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')
SCALARS = (
    'config_resolved.yaml', 'environment.json', 'provenance.json', 'metrics.jsonl',
    'sampling_manifest.jsonl', 'initial_validation.json', 'zero_volume_baseline.json',
    'gt_read_audit.json', 'acceptance_result.json', 'training_summary.json',
    'measurement_dependence.json', 'freeze.json', 'frozen_checkpoint.json',
    'failure.json', 'budget_partial_epoch.json',
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def relative_link(path, report):
    import os
    return os.path.relpath(Path(path).resolve(), Path(report).resolve().parent)


def _number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def _score(value):
    if not value:
        return None
    return value.get('measurement', {}).get('heldout', {}).get('relative_l1')


def _audit_status(audit, synthetic=False):
    if synthetic:
        return dict(recorded=bool(audit), real_file_audit=False, zero_forbidden_attempts=None,
                    note='解析合成测量，不读取外部三维标签；不能代替真实数据访问审计。')
    recorded = isinstance(audit, dict) and all(key in audit for key in COUNTERS)
    return dict(recorded=recorded, real_file_audit=recorded,
                zero_forbidden_attempts=all(audit[key] == 0 for key in COUNTERS) if recorded else None,
                note='三个违规访问计数均为零。' if recorded and all(audit[key] == 0 for key in COUNTERS)
                     else '访问审计尚未形成完整记录。' if not recorded else '存在违规文件访问尝试，不能通过验收。')


def collect_run(path, archive_root):
    path = Path(path).resolve()
    if not (path / 'config_resolved.yaml').is_file():
        return None
    config = yaml.safe_load((path / 'config_resolved.yaml').read_text())
    provenance = load_json(path / 'provenance.json') or {}
    summary = load_json(path / 'training_summary.json') or {}
    acceptance = load_json(path / 'acceptance_result.json') or {}
    freeze = load_json(path / 'frozen_checkpoint.json') or {}
    audit = load_json(path / 'gt_read_audit.json') or summary.get('gt_access')
    lines = []
    metrics = path / 'metrics.jsonl'
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            if not line.strip():
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                # An actively running writer may have an unfinished final line.
                break
    lines.sort(key=lambda row: row['epoch'])
    synthetic = config.get('stage') == 'synthetic_smoke' or config.get('dataset', {}).get('kind') == 'synthetic'
    state = 'running_or_interrupted'
    if summary.get('completed'):
        state = 'complete_pass' if acceptance.get('passed') else 'complete_fail'
    elif summary.get('stopping_reason') == 'budget_exhausted':
        state = 'budget_incomplete'
    elif (path / 'failure.json').exists():
        state = 'failed'
    audit_status = _audit_status(audit, synthetic)
    source_files = provenance.get('code', {})
    actual_source = {}
    changed = []
    for relative, expected in source_files.items():
        source_path = (ROOT / relative).resolve()
        actual = digest(source_path) if source_path.is_file() and source_path.is_relative_to(ROOT) else None
        actual_source[relative] = actual
        if actual != expected:
            changed.append(relative)
    actual_hash = canonical(actual_source) if actual_source else None
    source_matches = bool(source_files) and not changed and actual_hash == provenance.get('source_sha256')
    destination = archive_root / path.name
    destination.mkdir(parents=True, exist_ok=True)
    archive = []
    for name in SCALARS:
        original = path / name
        if original.is_file():
            target = destination / name
            shutil.copyfile(original, target)
            archive.append(dict(file=name, sha256=digest(target), bytes=target.stat().st_size))
        elif (destination / name).exists():
            # Do not keep a stale scalar from an earlier generation of this report.
            (destination / name).unlink()
    initial = load_json(path / 'initial_validation.json')
    zero = load_json(path / 'zero_volume_baseline.json')
    best_observed = min((row['selection_metric'] for row in lines), default=None)
    committed = summary.get('completed_epochs')
    return dict(name=path.name, path=display_path(path), stage=config['stage'], synthetic=synthetic, state=state,
        model=config['model'], seed=config['training']['seed'], device=config['training']['device'],
        train_cases=config['dataset'].get('train_limit') or len(provenance.get('train_case_ids', [])),
        val_cases=config['dataset'].get('val_limit') or len(provenance.get('val_case_ids', [])),
        planned_epochs=config['training']['max_epochs'], committed_completed_epochs=committed,
        logged_epochs=len(lines), observed_optimizer_updates=lines[-1]['global_step'] if lines else 0,
        initial_heldout_relative_l1=_score(initial), best_observed_heldout_relative_l1=best_observed,
        final_observed_heldout_relative_l1=lines[-1]['selection_metric'] if lines else None,
        zero_baseline_relative_l1=zero.get('relative_l1') if zero else None,
        initial_train_loss=lines[0]['train']['loss']['total'] if lines else None,
        final_train_loss=lines[-1]['train']['loss']['total'] if lines else None,
        total_logged_epoch_seconds=sum(row['train'].get('seconds', 0) + row.get('validation', {}).get('seconds', 0) for row in lines),
        peak_gpu_allocated_bytes=max((row.get('gpu_peak_allocated_bytes', 0) for row in lines), default=0),
        acceptance=acceptance, summary=summary, freeze=freeze, audit=audit, audit_status=audit_status,
        measurement_dependence=load_json(path / 'measurement_dependence.json'), budget=config.get('budget'),
        training_source_sha256=provenance.get('source_sha256'), actual_source_sha256=actual_hash,
        source_matches_training=source_matches, changed_source_files=changed,
        actual_source_files=actual_source, git_commit=provenance.get('git_commit'),
        data_sha256=provenance.get('data_sha256'), config_sha256=provenance.get('config_sha256'),
        archive=display_path(destination), scalar_files=archive,
        scientific_formal_result=bool(config['stage'] == 'formal' and summary.get('completed') and
            summary.get('completed_epochs') == 300 and freeze.get('stage') == 'formal' and
            freeze.get('completed_epochs') == 300 and freeze.get('formal_evaluation_allowed') is True and
            source_matches and audit_status['zero_forbidden_attempts'] is True),
        curves=[dict(epoch=row['epoch']+1, step=row['global_step'], train_loss=row['train']['loss']['total'],
                     validation_loss=row.get('validation', {}).get('loss', {}).get('total'),
                     heldout_relative_l1=row['selection_metric']) for row in lines])


def make_plots(runs, target):
    available = [run for run in runs if run['curves']]
    if not available:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for run in available:
        points = run['curves']
        label = run['name'] + (' (synthetic)' if run['synthetic'] else '')
        axes[0].plot([p['step'] for p in points], [p['train_loss'] for p in points], label=label)
        axes[1].plot([p['step'] for p in points], [p['heldout_relative_l1'] for p in points], label=label)
    for axis in axes:
        axis.set_yscale('log')
        axis.set_xlabel('Optimizer updates')
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel('Training measurement-only loss')
    axes[1].set_ylabel('Held-out measurement relative L1')
    axes[1].axhline(1., color='gray', linestyle=':', linewidth=1., label='Zero prediction (nonzero references)')
    axes[1].legend(fontsize=7)
    figure.suptitle('Plan A development evidence; synthetic and real-data runs have different protocols')
    result = []
    for suffix in ('png', 'pdf'):
        path = target / f'training_measurement_curves.{suffix}'
        figure.savefig(path, dpi=160)
        result.append(display_path(path))
    plt.close(figure)
    return result


def build_report(args):
    artifacts = Path(args.artifacts).resolve()
    report = Path(args.report).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    assets = load_json(artifacts / 'assets_audit.json')
    operator = load_json(artifacts / 'operator_numerics.json')
    log = artifacts / 'pytest.log'
    test_text = log.read_text() if log.exists() else ''
    matches = list(re.finditer(r'(\d+) passed(?:, (\d+) warnings?)?(?: in ([\d.]+)s)?', test_text))
    test_result = dict(recorded=bool(matches), passed=int(matches[-1].group(1)) if matches else None,
        warnings=int(matches[-1].group(2) or 0) if matches else None,
        seconds=float(matches[-1].group(3)) if matches and matches[-1].group(3) else None,
        failures_reported=bool(re.search(r'\d+ failed|ERROR collecting|FAILED ', test_text)),
        log_sha256=digest(log) if log.exists() else None)
    paths = [Path(path).resolve() for path in args.run]
    if args.discover:
        paths.extend(path.resolve() for path in (ROOT / 'outputs/research').glob('scheme_a_*') if path.is_dir())
    final_synthetic = artifacts / 'training/scheme_a_A2_synthetic20_final'
    if final_synthetic.is_dir():
        paths.append(final_synthetic)
    runs = []
    for path in sorted(set(paths)):
        value = collect_run(path, artifacts / 'run_scalars')
        if value is not None:
            runs.append(value)
    plots = make_plots(runs, artifacts)
    formal = [run for run in runs if run['scientific_formal_result']]
    formal_seed_coverage = {}
    for run in formal:
        key = run['model'].get('variant', 'unknown') + '/' + run['model'].get('preconditioner', 'geometry')
        formal_seed_coverage.setdefault(key, set()).add(run['seed'])
    summary = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='User confirmed implementation and small-scale acceptance only; preserve formal configurations without launching formal training',
        report_script_sha256=digest(Path(__file__)), assets=assets, operator=operator, tests=test_result,
        runs=runs, figures=plots, formal_completed_runs=len(formal),
        external_inference=load_json(artifacts / 'external_inference.json'),
        formal_acceptance_preflight=load_json(artifacts / 'formal_acceptance_preflight.json'),
        environment_snapshot=display_path(artifacts / 'environment_snapshot.json') if (artifacts / 'environment_snapshot.json').exists() else None,
        formal_seed_coverage={key: sorted(seeds) for key, seeds in formal_seed_coverage.items()},
        three_seed_formal_complete=any(set((20261011, 20261012, 20261013)).issubset(seeds) for seeds in formal_seed_coverage.values()),
        gt_boundary='Training and model selection read only measurements. Independent evaluation rejects synthetic/smoke/gate checkpoints and requires a frozen completed 300-epoch formal run.',
        input_files={display_path(path): digest(path) for path in (artifacts / 'assets_audit.json', artifacts / 'assets_after_implementation.json', artifacts / 'operator_numerics.json', artifacts / 'geometry_audit.json', artifacts / 'external_inference.json', artifacts / 'environment_snapshot.json', artifacts / 'formal_config_validation.json', artifacts / 'formal_acceptance_preflight.json', log) if path.exists()})
    (artifacts / 'evidence_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    status_text = dict(complete_pass='已完成，所列验收通过', complete_fail='已完成，所列验收未通过',
                       budget_incomplete='预算耗尽，未完成 / NO-GO', failed='运行失败 / NO-GO', running_or_interrupted='运行中或中断，未形成完成证据')
    completed_smoke = [run for run in runs if run['stage'] == 'smoke' and run['state'] == 'complete_pass']
    completed_gate = [run for run in runs if run['stage'] == 'gate' and run['state'] == 'complete_pass']
    opening = '方案 A 的显式算子、集合初始化与物理展开、独立自监督训练和受限评价入口已实现。'
    opening += f'当前归档包含真实数据 smoke 通过 {len(completed_smoke)} 次、16 案例 gate 通过 {len(completed_gate)} 次、完整正式运行 {len(formal)} 次。'
    opening += '这些开发证据不能替代 300 epochs、三个随机种子的公平比较，也不能据此宣称三维重建优于历史基线。'
    lines = ['# 方案 A 实施与验收记录', '', f"生成时间：{summary['generated_at']}（Asia/Shanghai）。", '', opening, '',
        '## 代码与协议', '',
        '- 新增 `physics/explicit_ray_operator.py`：显式三线性 stencil、质量分数插值后转摩尔分数、96 个中点积分、Jv/Jᵀu 与几何预条件。',
        '- 新增 `models/set_unrolled3d.py`：A0 集合初始化、A1 物理展开、A2 共享三维先验、A3 OARI 初始化和独立 A-control。',
        '- 新增 `training/research_selfsup.py` 与 `scripts/train_research_selfsup.py`：复用无标签数据、拼批与冻结外层损失；保留历史 selfsup 配置验证器。',
        '- 训练采用有效案例 batch=2、FP32、AdamW、cosine、固定 A/B/H 互斥采样；H 从不进入重建网络。每案例先算损失，再按实际案例数平均，尾组保留。',
        '- 仅以固定测量验证的 held-out relative L1 选择 best；保存 last、完整随机状态、scheduler、采样清单以及代码/配置/数据哈希。已验证 epoch 边界恢复与不中断运行逐位一致。',
        '- 正式训练前检查同模型、同代码和相同数据内容的真实 smoke 与 gate 通过证据；模板 YAML、合成 smoke、失败或预算未完成的 gate 均不能替代。', '',
        '## 资产与数值检查', '']
    if assets:
        baseline = assets.get('baseline_reproduction', {})
        lines += [f"资产审计：passed={assets.get('passed')}；分割 {assets.get('split_counts')}；历史源码未变={assets.get('historical_source_unchanged')}；数据元信息未变={assets.get('dataset_metadata_unchanged')}。",
            '', f"历史 MRGR 的固定 36 案例测量重放 relative L1={_number(baseline.get('actual'), 10)}，与记录差值={_number(baseline.get('absolute_difference'))}，容差={_number(baseline.get('tolerance'))}。这是复现检查；当前开发 smoke/gate 使用不同案例子集和训练预算，不能据此进行方法排名。", '']
    else:
        lines += ['尚无资产审计记录。', '']
    if operator:
        lines += ['| 数值环境 | 前向对齐相对 L2 | Jv 差分相对 L2 | 转置恒等式误差 | 二阶差分误差 | 通过 |', '|---|---:|---:|---:|---:|---|']
        for row in operator.get('numerical', []):
            checks = row['checks']
            error = lambda key: _number(checks.get(key, {}).get('error'))
            lines.append(f"| {row['device']} {row['dtype']} | {error('forward_vs_legacy_relative_l2')} | {error('jvp_finite_difference_relative_l2')} | {error('transpose_identity_relative')} | {error('hessian_finite_difference_relative_l2')} | {row['passed']} |")
        lines += ['', '数值对照使用合成解析场，CPU FP64 与部署 GPU FP32 的容差分别保存。小网格与有限几何通过不等于任意输入均已验证。', '']
    tests_label = f"{test_result['passed']} 项通过；warnings={test_result['warnings']}" if test_result['recorded'] else '尚未读取到已完成的测试汇总'
    lines += [f"自动测试：{tests_label}；记录到失败={test_result['failures_reported']}。完整输出见 [pytest.log]({relative_link(log, report)})。", '',
        '## 运行证据与失败保留', '',
        '| 运行 | 阶段 | train/val 案例 | 已记录/计划 epochs | 更新数 | 初始 H L1 | 最佳 H L1 | 零场 H L1 | 状态 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for run in runs:
        lines.append(f"| {run['name']} | {run['stage']} | {run['train_cases']}/{run['val_cases']} | {run['logged_epochs']}/{run['planned_epochs']} | {run['observed_optimizer_updates']} | {_number(run['initial_heldout_relative_l1'])} | {_number(run['best_observed_heldout_relative_l1'])} | {_number(run['zero_baseline_relative_l1'])} | {status_text[run['state']]} |")
    if not runs:
        lines += ['| — | — | — | — | — | — | — | — | 尚无可读取运行 |']
    for run in runs:
        lines += ['', f"### {run['name']}", '',
            f"状态：{status_text[run['state']]}。seed={run['seed']}，device={run['device']}，variant={run['model'].get('variant')}，K={run['model'].get('steps')}，Q={run['model'].get('samples')}，输出网格={run['model'].get('output_resolution')}³。",
            '', f"训练 loss：{_number(run['initial_train_loss'])} → {_number(run['final_train_loss'])}；已记录训练+测量验证耗时={_number(run['total_logged_epoch_seconds'])} 秒；峰值 GPU allocated={_number(run['peak_gpu_allocated_bytes']/2**30, 4)} GiB。耗时包括当时共享 GPU 负载，不能直接外推正式训练成本。",
            '', f"访问边界：{run['audit_status']['note']} 当前源码与训练源码哈希一致={run['source_matches_training']}。",
            '', f"训练源码 SHA256：`{run['training_source_sha256']}`；当前对应源码 SHA256：`{run['actual_source_sha256']}`。",
            '', f"标量记录：[归档目录]({relative_link(ROOT / run['archive'], report)})；原始运行目录：`{run['path']}`。"]
        if run['changed_source_files']:
            lines += ['', '训练后发生变化的源码：' + '、'.join(f'`{name}`' for name in run['changed_source_files']) + '。']
        if run['synthetic']:
            lines += ['', '此运行是解析合成测量上的工程烟雾检查，使用显式缩小的网格和积分设置；它不读取 CFD 测量，也不构成真实 G6/G7 通过证据。']
        if run['budget']:
            lines += ['', f"预算：{run['budget']}。在 optimizer 组边界及验证 batch 边界检查时间；未完整完成的 epoch/验证不保存为可恢复边界，不产生通过证明。"]
        if run['acceptance'].get('criteria'):
            failed = [key for key, value in run['acceptance']['criteria'].items() if not value]
            lines += ['', '验收未通过项：' + ('、'.join(failed) if failed else '无；只代表本次所列开发检查通过。')]
        if run['summary'].get('stopping_reason'):
            lines += ['', f"停机原因：`{run['summary']['stopping_reason']}`；所有失败/预算未完成记录原样保留。"]
        probe = run['measurement_dependence']
        if probe:
            lines += ['', f"同几何测量 ×1.2 干预：体场平均绝对变化={_number(probe.get('mean_absolute_output_change'))}，passed={probe.get('passed')}。"]
    if plots:
        png = next(path for path in plots if path.endswith('.png'))
        lines += ['', f"![开发运行的测量曲线]({relative_link(ROOT / png, report)})", '',
            '各曲线对应不同数据子集/阶段；图用于检查数值轨迹，不作为跨阶段优劣比较。另提供同名 PDF。']
    inference = summary['external_inference']
    if inference:
        timing = inference.get('inference', {}).get('timing', {})
        lines += ['', '## 外部输入与环境', '',
            f"实际完成 measurement-only NPZ 推理：输出形状 `{inference.get('field_shape')}`，finite={inference.get('finite')}，工程检查点={inference.get('engineering_checkpoint')}，GT 体场读取={inference.get('gt_body_read')}。",
            '', f"单次共享 GPU 重建 {_number(timing.get('reconstruct_s'))} 秒，另行重投影 {_number(timing.get('project_s'))} 秒；此调用不构成稳定吞吐基准。详见 [external_inference.json]({relative_link(artifacts / 'external_inference.json', report)})。"]
    if summary['environment_snapshot']:
        lines += ['', f"[环境与依赖快照]({relative_link(artifacts / 'environment_snapshot.json', report)})记录实际版本与 GPU；未进行全新环境安装验证。"]
    lines += ['', '## 正式比较与 GT 边界', '',
        '用户已确认本轮仅完成实现与小规模验收，保留正式训练配置；本轮未启动正式训练。', '',
        f"三份正式配置的结构验证记录见 [formal_config_validation.json]({relative_link(artifacts / 'formal_config_validation.json', report)})；当前源码、完整数据内容和真实 smoke/gate 的只读预检记录见 [formal_acceptance_preflight.json]({relative_link(artifacts / 'formal_acceptance_preflight.json', report)})。这些预检不执行模型更新。", '',
        '正式运行采用全量 train=166、val=36、300 epochs。预注册种子为 20261011、20261012、20261013；目前完整三个种子=' + str(summary['three_seed_formal_complete']) + '。', '',
        '```bash',
        'python scripts/train_research_selfsup.py --config configs/research/scheme_a_a2_formal_v1_seed20261011.yaml',
        'python scripts/train_research_selfsup.py --config configs/research/scheme_a_a2_formal_v1_seed20261012.yaml',
        'python scripts/train_research_selfsup.py --config configs/research/scheme_a_a2_formal_v1_seed20261013.yaml',
        '```', '',
        '以上是实现后的可执行入口，运行前需有配置指向的真实 smoke/gate 通过记录，并完成实测成本核对；列出命令不表示已经执行。A0/A1/A3/A-control 有各自独立 formal 配置，同样受验收限制。', '',
        '训练与检查点选择只读取测量。`GTReadGuard` 审计 GT64/GT128、非白名单 NPY 和外来检查点；最终审计必须三个违规计数均为零。合成数据的无文件读取声明不能替代真实训练审计。', '',
        '独立三维评价仅接收冻结且完成 300 epochs 的 formal 检查点，拒绝 synthetic/smoke/gate；先保存全部预测，再访问 GT64。当前未由本报告读取任何三维 GT，未产生新三维精度、测试集排名或三个种子统计。', '',
        '## 复现本报告', '', '```bash',
        'python scripts/report_plan_a.py --discover', '```', '',
        f"机器可读证据：[evidence_summary.json]({relative_link(artifacts / 'evidence_summary.json', report)})。脚本仅读取训练产物，复制标量与配置；不加载、不归档权重或大体场张量。", '']
    report.write_text('\n'.join(lines))
    print(json.dumps(dict(report=display_path(report), summary=display_path(artifacts / 'evidence_summary.json'), runs=len(runs)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default=str(ROOT / 'docs/reports/artifacts/plan_a_v1'))
    parser.add_argument('--report', default=str(ROOT / 'docs/reports/plan_a_v1_zh.md'))
    parser.add_argument('--run', action='append', default=[], help='Read an explicit run directory; can be repeated')
    parser.add_argument('--discover', action='store_true', help='Discover outputs/research/scheme_a_* run directories')
    build_report(parser.parse_args())


if __name__ == '__main__':
    main()
