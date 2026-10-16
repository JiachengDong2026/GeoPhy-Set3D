#!/usr/bin/env python3
"""Plot a consistent snapshot of completed epochs without touching training."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = args.run_dir/'metrics.jsonl'
    snapshot = source.read_bytes()
    lines = snapshot.decode().splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
            # A live writer may have only written part of the final line.
    if not records:
        raise ValueError('No completed epochs found')
    epochs = [r['epoch']+1 for r in records]
    if epochs != list(range(1, epochs[-1]+1)):
        raise ValueError('Expected one contiguous record per completed epoch')
    train = [r['train']['loss']['total'] for r in records]
    val = [r['validation']['loss']['total'] for r in records]
    heldout = [r['selection_metric']*100 for r in records]
    best = min(range(len(records)), key=lambda i: heldout[i])
    snapshot_time = datetime.now(timezone.utc)
    generated = snapshot_time.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M China Standard Time')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    colors = ('#2563eb', '#e67824')
    for ax in axes[:2]:
        ax.plot(epochs, train, label='Train', color=colors[0], lw=1.7)
        ax.plot(epochs, val, label='Validation', color=colors[1], lw=1.7)
        ax.set_xlabel('Completed epoch')
        ax.set_ylabel('Total self-supervised loss')
        ax.legend(frameon=False)
    axes[0].set_yscale('log')
    axes[0].set_title('Full training history (log scale)')
    axes[1].set_title('Loss after initial convergence')
    start = min(10, max(0, len(records)-2))
    axes[1].set_xlim(epochs[start], max(epochs[-1], epochs[start]+1))
    recent = train[start:]+val[start:]
    padding = max((max(recent)-min(recent))*.12, max(recent)*.02, 1e-6)
    axes[1].set_ylim(max(0, min(recent)-padding), max(recent)+padding)
    axes[1].text(.98, .97, f'Latest train: {train[-1]:.5f}\nLatest val: {val[-1]:.5f}',
        transform=axes[1].transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax = axes[2]
    ax.plot(epochs, heldout, color='#16836b', lw=1.7, label='A: validation held-out')
    ax.scatter(epochs[best], heldout[best], color='#16836b', s=38, zorder=3)
    ax.axhline(3.275178736, color='#7c6a9b', ls='--', lw=1.3,
               label='Historical MRGR: 3.275% (300 ep)')
    ax.set_yscale('log')
    ax.set_ylabel('Held-out projection relative L1 (%)')
    ax.set_xlabel('Completed epoch')
    ax.set_title('Validation measurement error (log scale)')
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    ax.annotate(f'Best: {heldout[best]:.3f}% @ epoch {epochs[best]}',
                xy=(epochs[best], heldout[best]), xycoords='data',
                xytext=(.04, .32), textcoords='axes fraction', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#16836b'))
    for ax in axes:
        ax.grid(True, which='major', alpha=.19)
    fig.suptitle(f'Scheme A budget v4 | seed 20261011 | {epochs[-1]}/300 completed epochs', fontsize=15)
    fig.supxlabel('Raw epoch means; no smoothing. Total = hold + 0.5 cross + 0.1 observed + 0.05 consistency\n'
                  f'Snapshot: {generated} | Training is still running. Projection error is not 3D field error.', fontsize=9)
    output = args.output_dir or args.run_dir/'plots'
    output.mkdir(parents=True, exist_ok=True)
    prefix = output/f'loss_curve_epoch_{epochs[-1]:04d}'
    for suffix in ('.png', '.svg'):
        fig.savefig(prefix.with_suffix(suffix), dpi=180, facecolor='white')
    plt.close(fig)
    columns = ('epoch', 'global_step', 'train_total', 'validation_total', 'validation_heldout_relative_l1',
               'train_hold', 'train_cross', 'train_observed', 'train_consistency')
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow(dict(epoch=r['epoch']+1, global_step=r['global_step'],
                train_total=r['train']['loss']['total'], validation_total=r['validation']['loss']['total'],
                validation_heldout_relative_l1=r['selection_metric'],
                **{'train_'+k: r['train']['loss'][k] for k in ('hold', 'cross', 'observed', 'consistency')}))
    summary = dict(completed_epochs=epochs[-1], latest_train_loss=train[-1], latest_validation_loss=val[-1],
                   latest_heldout_relative_l1=heldout[-1]/100, best_epoch=epochs[best],
                   best_heldout_relative_l1=heldout[best]/100, snapshot_utc=snapshot_time.isoformat(),
                   source=str(source.resolve()), snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
                   png=str(prefix.with_suffix('.png').resolve()), svg=str(prefix.with_suffix('.svg').resolve()))
    prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
