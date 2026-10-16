#!/usr/bin/env python3
"""Post-training validation of frozen B, with historical input IDs replayed.

Run predict and score in separate processes. This is validation development
evaluation, not the locked final test or a three-seed research conclusion.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

import evaluate_plan_b as evaluation
from geophy_set3d.data.build_utils import digest, write_json, timestamp
from geophy_set3d.training.gt_read_guard import GTReadGuard

REFERENCE = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1/selfsup_oari_mrgr_v1')
CONDITIONS = [('Val-A-V4', 4, None), ('V3', 3, 0), ('V6', 6, 0),
              ('clustered-V4', 4, 0), ('spread-V4', 4, None)]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(run):
    frozen = evaluation.frozen_record(run)
    config = yaml.safe_load((run / 'config_resolved.yaml').read_text())
    source = read_rows(Path(config['dataset']['root']) / 'source_index.jsonl')
    expected = {r['case_id'] for r in source if r['split'] == 'val'}
    settings = []
    for name, views, draw in CONDITIONS:
        suffix = Path(name) if draw is None else Path(name) / f'draw{draw}'
        reference = REFERENCE / suffix / 'C1/per_case_metrics.jsonl'
        rows = read_rows(reference)
        assert len(rows) == len(expected) == 36
        assert {r['case_id'] for r in rows} == expected
        ids = {r['case_id']: r['selected_station_ids'] for r in rows}
        assert all(len(v) == len(set(v)) == views and min(v) >= 0 and max(v) < 64 for v in ids.values())
        settings.append(dict(name=name, views=views, draw=draw,
                             reference=str(reference), reference_sha256=digest(reference),
                             input_ids=ids, layout='matched_' + name))
    return frozen, config, settings


def predict(run, device):
    frozen, config, settings = prepare(run)
    output = run / 'evaluation/frozen_model_test'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'registration.json', dict(
        registered_at=timestamp(), checkpoint_sha256=frozen['sha256'],
        completed_epochs=300, best_epoch=frozen['best_epoch'], settings=settings,
        script_sha256=digest(Path(__file__)), dataset_root=config['dataset']['root'],
        split='val', geometry_pool='train', final_test_evaluated=False,
        metrics='macro per-case GT64 relative L1/L2, NMSE, MAE, RMSE; observed/heldout projection error',
        paired_statistics='36 paired cases, 10000 bootstrap resamples, seed 20261016',
        historical_heldout_not_paired=True,
        scope='single seed; V3/V6/clustered each draw0; primary legacy fixed Val-A-V4'))
    saved_paths = {str(run / 'evaluation' / f"val_train_{s['layout']}_v{s['views']}_draw0"
                       / 'predictions' / f'{i:03d}.npy')
                   for s in settings for i in range(36)}

    class PredictionGuard(GTReadGuard):
        def _audit(self, event, args):
            # The training guard forbids all nonmeasurement NPY reads. Here only
            # the exact 180 output files may additionally be hashed after save.
            if (self.active and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike))
                    and os.path.abspath(os.fsdecode(args[0])) in saved_paths):
                return
            return super()._audit(event, args)

    with PredictionGuard(config['dataset']['root'], run / 'checkpoints') as guard:
        for setting in settings:
            class ReplaySampler:
                def __init__(self, azimuths, views, draw, pool, seed, protocol):
                    self.views, self.draw_id, self.pool, self.protocol = views, draw, pool, protocol

                def sample(self, station_count, *, epoch, sample_index, case_id):
                    ids = np.asarray(setting['input_ids'][case_id], dtype=np.int64)
                    assert len(ids) == self.views and ids.max() < station_count
                    return ids, np.empty(0, dtype=np.int64)

            evaluation.EvaluationViewSampler = ReplaySampler
            directory = evaluation.generate_predictions(run, device, views=setting['views'],
                layout=setting['layout'], draw=0, pool='train')
            rows = read_rows(directory / 'predictions_manifest.jsonl')
            assert len(rows) == 36
            for row in rows:
                assert row['selected_station_ids'] == setting['input_ids'][row['case_id']]
                assert not set(row['selected_station_ids']) & set(row['heldout_station_ids'])
            write_json(output / (setting['name'] + '_prediction.json'), dict(directory=str(directory)))
            print('Saved all 36 predictions:', setting['name'], flush=True)
    write_json(output / 'prediction_read_audit.json', guard.summary())
    write_json(output / 'all_predictions_complete.json', dict(completed_at=timestamp(),
        registration_sha256=digest(output / 'registration.json'), cases=36, settings=5,
        predictions=180, gt_opened=False, final_test_evaluated=False))


def score(run):
    output = run / 'evaluation/frozen_model_test'
    registration_path = output / 'registration.json'
    registration = json.loads(registration_path.read_text())
    complete = json.loads((output / 'all_predictions_complete.json').read_text())
    assert digest(registration_path) == complete['registration_sha256']
    assert evaluation.frozen_record(run)['sha256'] == registration['checkpoint_sha256']
    # Validate every registered prediction and historical reference before any GT.
    directories = []
    for setting in registration['settings']:
        assert digest(setting['reference']) == setting['reference_sha256']
        directory = Path(json.loads((output / (setting['name'] + '_prediction.json')).read_text())['directory'])
        boundary = json.loads((directory / 'prediction_boundary.json').read_text())
        assert digest(directory / 'predictions_manifest.jsonl') == boundary['prediction_manifest_sha256']
        rows = read_rows(directory / 'predictions_manifest.jsonl')
        assert len(rows) == 36 and {r['case_id'] for r in rows} == set(setting['input_ids'])
        for row in rows:
            assert row['selected_station_ids'] == setting['input_ids'][row['case_id']]
            assert digest(directory / row['prediction']) == row['prediction_sha256']
        directories.append(directory)
    summaries = []
    for setting, directory in zip(registration['settings'], directories):
        evaluation.score_saved_predictions(directory)
        rows = read_rows(directory / 'per_case_metrics.jsonl')
        old = {r['case_id']: r for r in read_rows(Path(setting['reference']))}
        current = np.array([r['relative_l1'] for r in rows])
        historical = np.array([old[r['case_id']]['relative_l1'] for r in rows])
        delta = current - historical
        rng = np.random.default_rng(20261016)
        ci = np.quantile(delta[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(1), [.025, .975])
        summary = json.loads((directory / 'summary.json').read_text())
        timing = np.array([r['timing']['encode_sample_and_continuous_projects_s'] for r in rows])
        summaries.append(dict(condition=setting['name'], cases=len(rows),
            metrics=summary['mean_per_case'], measurement=summary['measurement'],
            historical_relative_l1=float(historical.mean()),
            relative_error_reduction=float(1 - current.mean()/historical.mean()),
            paired_mean_delta=float(delta.mean()), paired_delta_ci95=ci.tolist(),
            case_wins=int((delta < -1e-8).sum()), case_ties=int((np.abs(delta) <= 1e-8).sum()),
            case_losses=int((delta > 1e-8).sum()),
            reconstruction_seconds_median=float(np.median(timing)),
            reconstruction_seconds_first=float(timing[0]),
            reconstruction_seconds_median_excluding_first=float(np.median(timing[1:])),
            directory=str(directory)))
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    write_json(output / 'summary.json', dict(
        completed_at=timestamp(), checkpoint_sha256=registration['checkpoint_sha256'],
        best_epoch=registration['best_epoch'], results=summaries,
        development_gt_evaluation=True, final_test_evaluated=False,
        gt_for_training=False, gt_for_selection=False, seeds=1,
        prediction_read_audit=json.loads((output / 'prediction_read_audit.json').read_text())))
    lines = ['# 方案 B：300 轮完成后的冻结模型评价', '',
        f"完成时间（北京时间，UTC+8）：{timestamp()}。最佳检查点第 {registration['best_epoch']} 轮，单种子 20261011。",
        '', '采用官方 36 个验证案例、Pool A、64³ XYZ 端点网格。每组精确重放历史自监督 MRGR 的输入站位及顺序；180 份预测全部保存并哈希封存后，独立进程读取验证 GT64。',
        '', '这是验证集开发评价，最终 test 未读取；V3/V6/聚簇仅使用历史 draw0，不能视为原五 draw 汇总。',
        '', '| 条件 | B 三维相对 L1 | 历史三维相对 L1 | 误差相对降低 | 胜/平/负 |',
        '|---|---:|---:|---:|---:|']
    for result in summaries:
        lines.append(f"| {result['condition']} | {result['metrics']['relative_l1']:.6f} | {result['historical_relative_l1']:.6f} | {100*result['relative_error_reduction']:.2f}% | {result['case_wins']}/{result['case_ties']}/{result['case_losses']} |")
    lines += ['', '误差均为逐案例计算后取均值，relative L1 = Σ|预测−GT| / Σ|GT|；三维浓度误差与训练时的留出投影误差不同。',
        '', '配对 bootstrap 使用 36 个案例、10,000 次重采样、固定种子 20261016；置信区间只反映案例变化，未包含训练种子变化。',
        '', '下表 Δ 为 B 减历史模型，负值代表误差降低。', '', '| 条件 | 三维 L2 | NMSE | RMSE | ΔL1 的 95% CI | 编码、采样及投影中位数（秒） |', '|---|---:|---:|---:|---|---:|']
    for result in summaries:
        m, ci = result['metrics'], result['paired_delta_ci95']
        lines.append(f"| {result['condition']} | {m['relative_l2']:.6f} | {m['nmse']:.6f} | {m['rmse']:.6g} | [{ci[0]:.6f}, {ci[1]:.6f}] | {result['reconstruction_seconds_median_excluding_first']:.3f} |")
    lines += ['', '计时为 GPU1 上单案例编码、64³ 连续场采样及输入/H4 两组连续投影的合计，表中排除首例；与 A 仅重建的计时口径不同，不可直接比较。投影指标来自连续场直接积分，三维指标来自同一函数的64³网格采样。',
        '', f"检查点 SHA256：`{registration['checkpoint_sha256']}`。",
        '', f'完整逐案例指标、预测、读取审计及汇总：`{output.relative_to(ROOT)}`。',
        '', '评估入口：`scripts/test_plan_b_frozen.py predict --run-dir <run> --device cuda:0`，完成后单独执行 `score --run-dir <run>`。']
    report = ROOT / 'docs/reports/plan_b_completed_model_test_zh.md'
    report.write_text('\n'.join(lines) + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict', 'score'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'predict':
        predict(args.run_dir.resolve(), args.device)
    else:
        score(args.run_dir.resolve())
