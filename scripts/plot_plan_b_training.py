#!/usr/bin/env python3
"""Plot completed B learning diagnostics without opening training data or weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    snapshot = (args.run_dir / 'metrics.jsonl').read_bytes()
    rows = [json.loads(line) for line in snapshot.decode().splitlines() if line]
    epochs = [r['epoch'] + 1 for r in rows]
    if not rows or epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError('Expected contiguous completed epochs')
    initial = json.loads((args.run_dir / 'initial_validation.json').read_text())
    baseline = json.loads((args.run_dir / 'zero_volume_baseline.json').read_text())
    train = [r['train']['loss']['total'] for r in rows]
    val = [r['validation']['loss']['total'] for r in rows]
    heldout = [initial['measurement']['heldout']['relative_l1']] + [r['selection_metric'] for r in rows]
    best = min(rows, key=lambda r: r['selection_metric'])
    out = args.run_dir / 'plots'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, train, label='Train')
    axes[0].plot([0] + epochs, [initial['loss']['total']] + val, label='Validation')
    axes[0].set(xlabel='Completed epoch', ylabel='Total measurement-only loss', title='Scheme B learning diagnostic')
    axes[1].plot([0] + epochs, heldout, label='Validation held-out relative L1', color='#16836b')
    axes[1].axhline(baseline['relative_l1'], ls='--', color='#777777', label='Zero prediction baseline')
    axes[1].set(xlabel='Completed epoch', ylabel='Relative L1 (ratio)', title=f"Best: {best['selection_metric']:.4f} at epoch {best['epoch'] + 1}")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle(f'{args.run_dir.name} | small diagnostic, not formal evaluation', fontsize=10)
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'loss_curve.{suffix}', dpi=160)
    plt.close(fig)
    with (out / 'loss_curve.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_total_loss', 'val_total_loss', 'heldout_relative_l1', 'train_seconds', 'updates'])
        for r in rows:
            writer.writerow([r['epoch'] + 1, r['train']['loss']['total'], r['validation']['loss']['total'], r['selection_metric'], r['train']['seconds'], r['train']['optimizer_updates']])
    summary = dict(completed_epochs=len(rows), metrics_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
                   initial_heldout_relative_l1=heldout[0], best_heldout_relative_l1=best['selection_metric'],
                   best_epoch=best['epoch'] + 1, final_heldout_relative_l1=heldout[-1],
                   train_seconds=sum(r['train']['seconds'] for r in rows),
                   validation_seconds=sum(r['validation']['seconds'] for r in rows),
                   optimizer_updates=sum(r['train']['optimizer_updates'] for r in rows),
                   peak_allocated_gib=max(r['gpu_peak_allocated_bytes'] for r in rows) / 1024**3)
    summary['mean_seconds_per_update'] = summary['train_seconds'] / summary['optimizer_updates']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
