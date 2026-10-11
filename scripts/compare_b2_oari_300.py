#!/usr/bin/env python3
"""Reuse frozen B2 metrics; evaluate supervised OARI under the identical protocol."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from geophy_set3d.data.build_utils import digest, source_snapshot
from geophy_set3d.evaluation.analysis import aggregate_draws, read_jsonl, write_csv
from geophy_set3d.evaluation.protocols import evaluation_dataset, evaluate, load_model
from geophy_set3d.evaluation.oari_diagnostics import interaction_diagnostics
from geophy_set3d.training.comparison import assert_fair_b2_oari
from geophy_set3d.training.utils import load_config, seed_everything, write_json

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_oari_300')
REFERENCE = ROOT.parent/'b0_b1_b2_300'
LABELS = ('B2-GCSI-300', 'OARI-300')
METRICS = ('mae', 'rmse', 'relative_l1', 'relative_l2')
CONDITIONS = ((3, 'random', 'V3'), (4, 'random', 'V4'), (6, 'random', 'V6'),
              (4, 'clustered', 'clustered-V4'))


def prepare(configs):
    assert_fair_b2_oari(*configs)
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(configs=configs, reference_summary=str(REFERENCE/'summary.json'),
        reference_summary_sha256=digest(REFERENCE/'summary.json'), train_seed=20261011,
        validation_seed=20261011, validation_epoch=0, validation_views=4,
        model_selection='minimum fixed Val-A-V4 supervised loss', epochs=300,
        evaluation_seed=20261011, sparse_views=[3, 4, 6], draw_ids=list(range(5)),
        clustered_views=4, clustered_draw_ids=list(range(5)), spread_views=4, spread_draw_ids=[0],
        test_used=False, pool_b_evaluation_only=True, retrain_b0_b1_b2=False,
        physics_loss=False, diagnostics_val_indices=[0, 18, 35],
        observability_dimension=5, pairwise_dimension=3, pairwise_bias_network=[8, 16, 4],
        geometry_diversity='mean(1-abs(dot(d_i,d_j))) over distinct station pairs at domain center (0.3,0.3,0.3)m; actual 3D origins',
        geometry_analysis='descriptive Spearman only; repeated case/draw samples are not independent',
        engineering_go=dict(max_relative_main_l1_worsening=.02,
                            clear_sparse_l1_absolute_gain=.005,
                            shortcut_min_ratio=1.2, fp32_set_max_difference=1e-5),
        engineering_go_note='Same operational references as B2: 2% relative main L1 tolerance, 0.5 percentage point V3 or clustered L1 gain; report L2 separately, no statistical significance claim')
    target = ROOT/'effective_protocol.yaml'
    if target.exists() and yaml.safe_load(target.read_text()) != protocol:
        raise ValueError('OARI effective protocol changed')
    if not target.exists():
        target.write_text(yaml.safe_dump(protocol, sort_keys=False))
    # The inventory was captured before editing any baseline code or artifacts.
    if not (ROOT/'inventory_before.json').exists():
        raise RuntimeError('Capture protected inventory before implementing/training OARI')
    original = json.loads((ROOT/'inventory_before.json').read_text())['dataset']
    baseline = json.loads((REFERENCE/'inventory_before.json').read_text())['dataset']
    if original != baseline:
        raise AssertionError('Dataset differs from frozen B2 baseline')
    if not (ROOT/'training_source_manifest.json').exists():
        raise RuntimeError('Training source manifest must be frozen before full training')
    return protocol


def integrity(config):
    before = json.loads((ROOT/'inventory_before.json').read_text())
    base = ROOT.parent.parent
    current = source_snapshot(config['dataset']['root'])
    sources = json.loads((ROOT/'training_source_manifest.json').read_text())['sha256']
    old = before['existing_artifacts']; protected = before['protected_code']
    unchanged = lambda mapping, parent: all((parent/p).is_file() and digest(parent/p) == sha for p, sha in mapping.items())
    result = dict(dataset_files=len(current), dataset_unchanged=current == before['dataset'],
        all_historical_artifacts_unchanged=unchanged(old, base),
        b0_b1_b2_checkpoints_unchanged=unchanged({p:s for p,s in old.items() if p.startswith('checkpoints/') and 'b3' not in p}, base),
        b3_artifacts_unchanged=unchanged({p:s for p,s in old.items() if 'b3' in p}, base),
        b0_b1_b2_model_sources_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/models/')}, REPO),
        projector_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/physics/')}, REPO),
        protected_code_unchanged=unchanged(protected, REPO),
        numerical_training_sources_unchanged=unchanged(sources, REPO),
        test_used=False, pool_b_evaluation_only=True)
    result['passed'] = all(result[k] for k in result if k.endswith('_unchanged'))
    write_json(ROOT/'integrity_summary.json', result)
    if not result['passed']:
        raise AssertionError('Protected artifacts or training sources changed')
    return result


def pairing(rows):
    return [(r['case_id'], r['selected_station_ids'], r['station_azimuth_deg']) for r in rows]


class RelationMismatch(torch.nn.Module):
    """Change only relations inside interaction, keeping all measured inputs fixed."""
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, batch):
        permutation = torch.arange(batch['view_mask'].shape[1], device=batch['view_mask'].device).roll(1)
        return self.model(batch, relation_permutation=permutation)


def geometry_analysis(config, output, reference_output):
    rows = []
    center = np.full(3, .3)
    for views, selection, key in CONDITIONS:
        for draw in range(5):
            suffix = Path(key)/f'draw{draw}'/'per_case_metrics.jsonl'
            a, b = read_jsonl(reference_output/suffix), read_jsonl(output/suffix)
            if pairing(a) != pairing(b):
                raise AssertionError(f'Case/station pairing differs: {suffix}')
            ds = evaluation_dataset(config, views=views, selection=selection, draw=draw)
            for before, after in zip(a, b):
                ids = before['selected_station_ids']
                origins = np.asarray(ds.geometry['station_origin_m'][ids], dtype=np.float64)
                rays = center-origins
                rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-8)
                cosine = np.abs(rays @ rays.T).clip(0, 1)
                pairwise = np.triu_indices(views, 1)
                diversity = float((1-cosine[pairwise]).mean())
                angles = float(np.rad2deg(np.arccos(cosine[pairwise])).mean())
                rows.append(dict(case_id=before['case_id'], condition=key, draw_id=draw,
                    selected_station_ids=json.dumps(ids), geometry_diversity=diversity,
                    mean_pairwise_crossing_angle_deg=angles,
                    b2_relative_l1=before['relative_l1'], oari_relative_l1=after['relative_l1'],
                    l1_absolute_improvement=before['relative_l1']-after['relative_l1'],
                    l1_improvement_percentage_points=100*(before['relative_l1']-after['relative_l1']),
                    b2_relative_l2=before['relative_l2'], oari_relative_l2=after['relative_l2'],
                    l2_absolute_improvement=before['relative_l2']-after['relative_l2']))
    write_csv(ROOT/'geometry_stratified.csv', rows)
    correlations = {}
    for condition in ('all', 'V3', 'V4', 'V6', 'clustered-V4'):
        chosen = rows if condition == 'all' else [r for r in rows if r['condition'] == condition]
        correlations[condition] = dict(n=len(chosen), rho={})
        for metric in ('b2_relative_l1', 'oari_relative_l1', 'l1_absolute_improvement'):
            x, y = [r['geometry_diversity'] for r in chosen], [r[metric] for r in chosen]
            rho = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
            correlations[condition]['rho'][metric] = rho
    result = dict(score_definition='mean(1-abs(dot)) of actual station-to-domain-center 3D rays; off-diagonal unordered pairs',
        correlations=correlations, descriptive_only=True,
        repeated_cases_and_draws_not_independent=True, causal_claim=False)
    write_json(ROOT/'geometry_stratified_summary.json', result)
    return result, rows


def improvement_table(models):
    rows = []
    for condition in ('Val-A-V4', 'Val-B-V4', 'V3', 'V4', 'V6', 'clustered-V4', 'spread-V4'):
        values = []
        for label in LABELS:
            m = models[label]
            metrics = m['evaluations'][condition] if condition.startswith('Val-') else \
                m['spread'] if condition == 'spread-V4' else m['sparse'][condition]
            values.append(metrics['mean_per_case'])
        for metric in ('relative_l1', 'relative_l2'):
            a, b = [v[metric] for v in values]; gain = a-b
            rows.append(dict(condition=condition, metric=metric, b2=a, oari=b,
                absolute_improvement=gain, improvement_percentage_points=100*gain,
                relative_improvement_percent=100*gain/a))
    return rows


def make_figures(summary, geometry_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT/'figures'; directory.mkdir(exist_ok=True)
    models = summary['models']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, condition in zip(axes, ('Val-A-V4', 'Val-B-V4')):
        for i, label in enumerate(LABELS):
            m = models[label]['evaluations'][condition]['mean_per_case']
            ax.bar(np.arange(2)+i*.35, [m['relative_l1'], m['relative_l2']], .35, label=label)
        ax.set_xticks(np.arange(2)+.175, ['Relative L1', 'Relative L2']); ax.set_title(condition); ax.legend()
    fig.savefig(directory/'b2_oari_main_metrics.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, label in enumerate(LABELS):
        ax.bar(np.arange(3)+i*.35, [models[label]['sparse'][k]['mean_per_case']['relative_l1'] for k in ('V3', 'V4', 'V6')], .35, label=label)
    ax.set_xticks(np.arange(3)+.175, ['V3', 'V4 random', 'V6']); ax.set_ylabel('Mean per-case relative L1'); ax.legend()
    fig.savefig(directory/'b2_oari_sparse_views.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(LABELS, [models[k]['sparse']['clustered-V4']['mean_per_case']['relative_l1'] for k in LABELS])
    ax.set(ylabel='Mean per-case relative L1', title='Clustered V4: five paired deterministic draws')
    fig.savefig(directory/'b2_oari_clustered.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, ('b2_relative_l1', 'oari_relative_l1', 'l1_improvement_percentage_points'), ('B2 relative L1', 'OARI relative L1', 'B2 minus OARI (percentage points)')):
        for condition in ('V3', 'V4', 'V6', 'clustered-V4'):
            chosen = [r for r in geometry_rows if r['condition'] == condition]
            ax.scatter([r['geometry_diversity'] for r in chosen], [r[metric] for r in chosen], s=10, alpha=.5, label=condition)
        ax.set(xlabel='Geometry diversity at domain center', ylabel=label)
    axes[2].axhline(0, color='black', linewidth=.7); axes[0].legend(fontsize=8)
    fig.savefig(directory/'geometry_diversity_vs_improvement.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    for e in summary['interaction']['examples']:
        ax = axes[(0, 18, 35).index(e['case_index']), 0 if e['selection'] == 'spread' else 1]
        im = ax.imshow(e['attention_matrix_mean'], vmin=0, vmax=1, cmap='viridis')
        labels = [f'{sid}\n{az:.1f}°' for sid, az in zip(e['station_ids'], e['station_azimuth_deg'])]
        ax.set_xticks(range(4), labels, fontsize=8); ax.set_yticks(range(4), labels, fontsize=8)
        ax.set(xlabel='Key view', ylabel='Query view', title=f"Val {e['case_index']}: {e['selection']}")
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(directory/'oari_attention_example.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()
    configs = [load_config(f'configs/train_supervised_{name}_300.yaml') for name in ('b2', 'oari')]
    protocol = prepare(configs)
    if args.prepare:
        print('OARI protocol frozen; existing B2/B3/data protected'); return
    seed_everything(20261011); torch.set_num_threads(2)
    reference = json.loads((REFERENCE/'summary.json').read_text())
    if not reference['completed']:
        raise ValueError('Frozen B2 comparison incomplete')
    baseline = copy.deepcopy(reference['models'][LABELS[0]])
    if digest(baseline['checkpoint']) != baseline['checkpoint_sha256']:
        raise AssertionError('Frozen B2 checkpoint changed')
    config = configs[1]; name = config['experiment']['name']; base = Path(config['experiment']['output_root'])
    logs = base/'logs'/name
    training = json.loads((logs/'training_summary.json').read_text())
    if training['completed_epochs'] != 300 or training['stopping_reason'] != 'max_epochs':
        raise ValueError('OARI must finish all 300 epochs before any Pool B evaluation')
    checkpoint = base/'checkpoints'/name/'best.pt'; checkpoint_hash = digest(checkpoint)
    model, ckpt = load_model(checkpoint, args.device)
    if ckpt['config'] != config:
        raise ValueError('OARI checkpoint configuration changed')
    if json.loads((logs/'validation_station_ids.json').read_text()) != json.loads((Path(baseline['logs'])/'validation_station_ids.json').read_text()):
        raise AssertionError('B2/OARI model-selection views differ')
    config = copy.deepcopy(config); config['training']['device'] = args.device
    output = ROOT/name; output.mkdir(exist_ok=True)
    result = dict(logs=str(logs), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        best_epoch=ckpt['epoch']+1, best_val_loss=ckpt['best_val_metric'],
        parameter_count=sum(p.numel() for p in model.parameters()),
        interaction_parameter_count=sum(p.numel() for p in model.interaction.parameters()), evaluations={}, sparse={})
    reference_output = REFERENCE/Path(baseline['logs']).name
    def paired_evaluate(dataset, suffix, condition, draw=0, shortcut='normal', predictor=model):
        s, rows = evaluate(predictor, dataset, config, output/suffix,
                           condition=condition, draw_id=draw, shortcut=shortcut)
        if pairing(rows) != pairing(read_jsonl(reference_output/suffix/'per_case_metrics.jsonl')):
            raise AssertionError(f'OARI/B2 station pairing differs: {suffix}')
        print(f'{condition} draw={draw}: L1={s["mean_per_case"]["relative_l1"]:.9f}', flush=True)
        return s, rows
    for pool, condition in (('train', 'Val-A-V4'), ('ood', 'Val-B-V4')):
        result['evaluations'][condition], _ = paired_evaluate(evaluation_dataset(config, pool=pool), Path(condition), condition)
    for metric in METRICS:
        expected = training['best_val_metrics']['mean_per_case'][metric]
        actual = result['evaluations']['Val-A-V4']['mean_per_case'][metric]
        if abs(expected-actual) > 1e-7+1e-5*abs(expected):
            raise AssertionError('Best training Val-A metrics failed reproduction')
    for shortcut in ('wrong_geometry', 'zero_measurement'):
        result['evaluations'][shortcut], _ = paired_evaluate(evaluation_dataset(config), Path(shortcut), 'Val-A-V4', shortcut=shortcut)
    relation_model = RelationMismatch(model).eval()
    result['evaluations']['wrong_relation'], _ = evaluate(relation_model, evaluation_dataset(config), config,
        output/'wrong_relation', condition='Val-A-V4', shortcut='normal')
    result['evaluations']['wrong_relation']['shortcut'] = 'pairwise relation permutation only'
    write_json(output/'wrong_relation'/'metrics.json', result['evaluations']['wrong_relation'])
    for views, selection, key in CONDITIONS:
        summaries = []; rows = []
        for draw in range(5):
            s, r = paired_evaluate(evaluation_dataset(config, views=views, selection=selection, draw=draw), Path(key)/f'draw{draw}', key, draw)
            summaries.append(s); rows.extend(r)
        result['sparse'][key] = aggregate_draws(summaries, rows)
    result['spread'], _ = paired_evaluate(evaluation_dataset(config, selection='spread'), Path('spread-V4'), 'spread-V4')
    interaction = interaction_diagnostics(model, config, ROOT/'diagnostics')
    geometry, geometry_rows = geometry_analysis(config, output, reference_output)
    models = {LABELS[0]: baseline, LABELS[1]: result}
    b2a, oa = [m['evaluations']['Val-A-V4']['mean_per_case']['relative_l1'] for m in models.values()]
    b2b, ob = [m['evaluations']['Val-B-V4']['mean_per_case']['relative_l1'] for m in models.values()]
    go = protocol['engineering_go']
    criteria = dict(val_a_not_materially_worse=oa <= b2a*(1+go['max_relative_main_l1_worsening']),
        val_b_not_materially_worse=ob <= b2b*(1+go['max_relative_main_l1_worsening']),
        v3_or_clustered_clear_gain=any(baseline['sparse'][key]['mean_per_case']['relative_l1']-result['sparse'][key]['mean_per_case']['relative_l1'] >= go['clear_sparse_l1_absolute_gain'] for key in ('V3', 'clustered-V4')),
        shortcuts_normal=all(result['evaluations'][key]['mean_per_case']['relative_l1'] > oa*go['shortcut_min_ratio'] for key in ('wrong_geometry', 'zero_measurement')),
        mask_permutation_and_connectivity_correct=interaction['passed'])
    runtime = {}
    for label, item in models.items():
        training_rows = read_jsonl(Path(item['logs'])/'metrics.jsonl')
        runtime[label] = dict(mean_epoch_seconds=float(np.mean([r['seconds'] for r in training_rows])),
            peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in training_rows),
            peak_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in training_rows),
            note='Observed runs, not a controlled throughput benchmark')
    improvement = improvement_table(models)
    check = integrity(configs[1])
    if digest(checkpoint) != checkpoint_hash:
        raise AssertionError('OARI best checkpoint changed during evaluation')
    summary = dict(completed=True, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip(),
        training_source_commit=json.loads((ROOT/'training_source_manifest.json').read_text())['commit'],
        protocol=protocol, models=models, interaction=interaction, improvements=improvement,
        geometry_analysis=geometry, runtime=runtime, go_criteria=criteria, retain_oari=all(criteria.values()),
        final_architecture_candidate_v1=all(criteria.values()), integrity=check,
        b0_b1_b2_retrained=False, actual_case_station_pairings_identical=True,
        single_training_seed=True, test_used=False, pool_b_evaluation_only=True)
    main_rows = []; sparse_rows = []; shortcut_rows = []; draw_rows = []; case_rows = []
    for label, m in models.items():
        for condition in ('Val-A-V4', 'Val-B-V4'):
            main_rows.append(dict(model=label, condition=condition, best_epoch=m['best_epoch'], best_val_loss=m['best_val_loss'], **m['evaluations'][condition]['mean_per_case']))
        for condition in ('V3', 'V4', 'V6', 'clustered-V4'):
            sparse_rows.append(dict(model=label, condition=condition, draws=5, **m['sparse'][condition]['mean_per_case']))
        for condition in ('Val-A-V4', 'wrong_geometry', 'zero_measurement'):
            shortcut_rows.append(dict(model=label, condition=condition, **m['evaluations'][condition]['mean_per_case']))
        location = reference_output if label == LABELS[0] else output
        for draw in range(5):
            ds = json.loads((location/'clustered-V4'/f'draw{draw}'/'metrics.json').read_text())
            draw_rows.append(dict(model=label, draw_id=draw, **ds['mean_per_case']))
            for r in read_jsonl(location/'clustered-V4'/f'draw{draw}'/'per_case_metrics.jsonl'):
                case_rows.append(dict(model=label, case_id=r['case_id'], draw_id=draw,
                    selected_station_ids=json.dumps(r['selected_station_ids']), **{k:r[k] for k in METRICS}))
    shortcut_rows.append(dict(model=LABELS[1], condition='wrong_relation', **result['evaluations']['wrong_relation']['mean_per_case']))
    write_json(ROOT/'summary.json', summary)
    write_csv(ROOT/'comparison.csv', main_rows); write_csv(ROOT/'sparse_views.csv', sparse_rows)
    write_csv(ROOT/'clustered_v4.csv', [r for r in sparse_rows if r['condition'] == 'clustered-V4'])
    write_csv(ROOT/'clustered_v4_per_draw.csv', draw_rows); write_csv(ROOT/'clustered_v4_per_case.csv', case_rows)
    write_csv(ROOT/'shortcuts.csv', shortcut_rows); write_csv(ROOT/'improvements.csv', improvement)
    write_csv(ROOT/'spread_v4.csv', [dict(model=k, **m['spread']['mean_per_case']) for k,m in models.items()])
    make_figures(summary, geometry_rows)
    from geophy_set3d.evaluation.oari_report import render_oari_report
    render_oari_report(ROOT, REPO)


if __name__ == '__main__':
    main()
