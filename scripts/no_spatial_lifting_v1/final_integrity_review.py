"""Post-acceptance file review only; no model, dataset, or scoring imports.

This additional reviewer does not belong to or change the prelocked pipeline.
It checks saved evidence and exports after the original acceptance completes.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen(path):
    path = Path(path)
    assert digest(path) == path.with_suffix(path.suffix + '.sha256').read_text().strip(), path
    return read(path)


def check(mapping):
    for path, expected in mapping.items():
        assert digest(path) == expected, path
    return len(mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    project = Path(__file__).resolve().parents[2]
    artifact = project / 'docs/reports/artifacts/mrgr_no_spatial_lifting_v1'
    report = project / 'docs/reports/mrgr_no_spatial_lifting_v1_zh.md'
    destination = base / 'post_acceptance_integrity_review.json'
    assert not destination.exists()
    acceptance = frozen(base / 'acceptance.json')
    assert acceptance['completed']
    assert acceptance['fusion_full'] == acceptance['fusion_ablation'] == 'Uniform Fusion'
    evidence_count = check(acceptance['evidence'])
    source_count = sum(check(frozen(base / name)['sources']) for name in (
        'protocol.json', 'analysis_source_prelock.json', 'evaluation_lock.json'))
    inventory = frozen(base / 'prediction_inventory.json')
    assert len(inventory['files']) == check(inventory['files']) == 1190
    check(inventory['manifests'])
    baseline = frozen(base / 'baseline_manifest.json')
    assert len(baseline) == 1190
    for item in baseline:
        assert digest(item['path']) == item['sha256']
    seal = frozen(base / 'all_predictions_sealed.json')
    metrics = frozen(base / 'metric_verification.json')
    assert metrics['passed']
    assert datetime.fromisoformat(seal['at']) < datetime.fromisoformat(metrics['first_GT64_access'])
    methods = frozen(base / 'evaluation_lock.json')['methods']
    for method in methods:
        assert digest(method['checkpoint']) == method['checkpoint_sha256']
        summary = read(Path(method['run']) / 'summary.json')
        assert summary['completed'] and summary['epochs'] == 300 and summary['updates'] == 24900
        assert summary['best_epoch'] == method['best_epoch']
    exports = frozen(artifact / 'export_manifest.json')['files']
    for relative, item in exports.items():
        target = project / relative
        assert digest(target) == item['sha256'], target
        if item.get('transform') == 'lossless gzip':
            assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == item['source_sha256']
        else:
            assert digest(item['source']) == item['sha256'], item['source']
    assert digest(artifact / 'acceptance.json') == digest(base / 'acceptance.json')
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        assert (report.parent / link).exists(), link
    assert frozen(base / 'parallel_OARI_diagnostic_boundary.json')['post_OARI_source_hashes_verified']
    recovery = frozen(base / 'recovery_execution_v1.json')
    check(recovery['recovery_sources'])
    continuation = frozen(base / 'post_training_recovery_v2.json')
    check(continuation['sources'])
    assert digest(continuation['traceback_log']) == continuation['traceback_log_sha256']
    reporting = frozen(base / 'report_recovery_v3.json')
    check(reporting['sources'])
    check(reporting['completed_analysis_evidence'])
    check(reporting['resolved_report_failures'])
    assert acceptance['anomalies'] == reporting['anomalies']
    for seed, interrupted in recovery['runs'].items():
        check(interrupted['preserved'])
        run = base / 'training' / f'no_spatial_lifting_seed{seed}'
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines() if line.strip()]
        resumed = [event for event in events if event['event'] == 'resume']
        assert len(resumed) == 1 and resumed[0]['epoch'] == interrupted['epoch']
        assert resumed[0]['rng_restored'] and resumed[0]['optimizer_scheduler_restored']
        rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
        assert [row['epoch'] for row in rows] == list(range(1, 301))
    result = dict(
        at=datetime.now(timezone.utc).isoformat(), passed=True,
        review='post-acceptance file integrity only; original locked analysis unchanged',
        reviewer_sha256=digest(__file__), acceptance_sha256=digest(base / 'acceptance.json'),
        evidence_files_checked=evidence_count, frozen_source_entries_checked=source_count,
        new_prediction_hashes_checked=1190, baseline_prediction_hashes_checked=1190,
        exported_files_checked=len(exports), report_links_verified=True,
        prediction_seal_before_first_GT64_verified=True,
        checkpoints_match_selection=True, GT64_read=False, GT128_read=False,
        interruption_recovery_verified=True,
    )
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    for target in (destination, artifact / destination.name):
        with target.open('xb') as stream:
            stream.write(data)
        target.with_suffix(target.suffix + '.sha256').write_text(digest(target) + '\n')
        target.chmod(0o444)
    supplement = report.with_name('mrgr_no_spatial_lifting_v1_execution_addendum_zh.md')
    assert not supplement.exists()
    supplement.write_text('\n'.join([
        '# GeometryLifting 消融：执行与数据边界补充', '',
        '本页补充[冻结分析生成的正式报告](mrgr_no_spatial_lifting_v1_zh.md)，不改变模型、选轮、指标或统计结果。', '',
        '训练期间一次会话中断后发现原控制器和两个训练进程均已退出；实际退出信号与退出码不可得。'
        '最后完整提交分别为seed20261011第268轮（22244次更新）和seed20261019第260轮（21580次更新）。'
        '原始环境、状态、指标、GT访问审计、last/best检查点均在恢复前保存副本及哈希。', '',
        '独立恢复登记锁定新增调度代码，原115份训练源码和7份分析源码未修改。'
        '恢复在原GPU UUID上通过排他锁和双次空闲检查启动，使用原worker恢复模型、最佳模型、optimizer、scheduler及Python/NumPy/Torch/CUDA/loader RNG。'
        '未提交的中途epoch按原恢复规则重放；最终每种子恰有连续300个epoch记录、24900次更新，无新增种子或训练因素。'
        '本次中断属于执行异常，应随结果披露。', '',
        '另一次控制器错误发生在首个训练任务完成后：完成状态含有seed字段，原progress()汇总重复传入seed，触发TypeError。'
        '第二个训练worker继续完成300轮；两份检查点均已保存。独立v2入口只修复状态合并并衔接原冻结评价流程，未重新训练、改变检查点选择或统计。'
        '错误堆栈和新增源码登记均保留。', '',
        '评分和独立统计完成后，还发生同名report模块导入冲突。独立v3报告入口只复用已完成结果，'
        '核验并保留该失败记录，未重跑训练、推理、评分或统计。正式报告与原始验收已披露全部三项异常。'
        '[补充最终验收](artifacts/mrgr_no_spatial_lifting_v1/acceptance_recovery_v1.json)额外绑定'
        '[独立完整性复查](artifacts/mrgr_no_spatial_lifting_v1/post_acceptance_integrity_review.json)。'
        '原报告、原验收和原始日志均保留，不覆盖历史证据。', '',
        '训练进程只读取登记的train/val投影，未读取测试数组、指标或GT。'
        '与此同时，用户另行授权对旧冻结Uniform Fusion模型开展OARI机制诊断；该独立任务复用了历史测试标量指标并读取少量测试投影，未读取GT64或GT128，'
        '也未读取当前GeometryLifting新模型测试结果。该并行授权不等于整个会话从未接触历史测试指标；'
        '其结果未用于改变当前结构、训练、选轮或分析协议。', '',
        '两完整对照和两消融均采用Uniform Fusion。新增预测统一封存时间早于第一次获准GT64评分；GT128始终禁止。'
        '本次是已使用测试集上的补充机制分析，不能单独证明可泛化几何规律。', '',
        '[恢复登记](artifacts/mrgr_no_spatial_lifting_v1/recovery_execution_v1.json) · '
        '[状态汇总恢复](artifacts/mrgr_no_spatial_lifting_v1/post_training_recovery_v2.json) · '
        '[报告恢复](artifacts/mrgr_no_spatial_lifting_v1/report_recovery_v3.json) · '
        '[并行OARI边界](artifacts/mrgr_no_spatial_lifting_v1/parallel_OARI_diagnostic_boundary.json)',
    ]) + '\n')
    final = dict(acceptance)
    final.update(
        at=datetime.now(timezone.utc).isoformat(),
        original_acceptance_sha256=digest(base / 'acceptance.json'),
        supersedes='adds post-acceptance file review; original results, verification, and anomaly disclosures unchanged',
        anomalies=list(acceptance['anomalies']),
        interruption_recovery_verified=True,
    )
    final['evidence'] = dict(acceptance['evidence'])
    for path in (destination, supplement, base / 'recovery_execution_v1.json', base / 'post_training_recovery_v2.json', base / 'report_recovery_v3.json'):
        final['evidence'][str(path)] = digest(path)
    data = (json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    for target in (base / 'acceptance_recovery_v1.json', artifact / 'acceptance_recovery_v1.json'):
        with target.open('xb') as stream:
            stream.write(data)
        target.with_suffix(target.suffix + '.sha256').write_text(digest(target) + '\n')
        target.chmod(0o444)
    logs = artifact / 'complete_execution_logs_v1'
    logs.mkdir(exist_ok=False)
    archive = {}
    for source in sorted((base / 'scheduling').iterdir()):
        if source.suffix not in ('.log', '.jsonl'):
            continue
        target = logs / (source.name + '.gz')
        with target.open('xb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                with source.open('rb') as raw:
                    shutil.copyfileobj(raw, compressed)
        assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == digest(source)
        archive[str(target.relative_to(project))] = dict(source=str(source), source_sha256=digest(source), sha256=digest(target))
    archive_path = artifact / 'complete_execution_archive_v1.json'
    archive_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + '\n')
    archive_path.with_suffix('.json.sha256').write_text(digest(archive_path) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
