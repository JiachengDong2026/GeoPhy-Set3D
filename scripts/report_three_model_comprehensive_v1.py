"""Render report-only figures from frozen scalar logs; never load model/GT arrays."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research/three_model_comprehensive_v1"
RUNS = {
    "History": Path("/data/djc/GeoPhy-Set3D/outputs/logs/selfsup_oari_mrgr_v1"),
    "A": ROOT / "outputs/research/scheme_a_budget_v4_wide_seed20261011",
    "B": ROOT / "outputs/research/scheme_b_perf_v2_formal_seed20261011",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"History": "#2864ad", "A": "#dd7722", "B": "#389153"}
    facts = {}
    hashes = {}
    late_values = []
    for name, run in RUNS.items():
        path = run / "metrics.jsonl"
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == 300
        assert [r["epoch"] for r in rows] == list(range(300))
        val_key = "val" if name == "History" else "validation"
        epochs = np.arange(1, 301)
        train_h = np.array([r["train"]["measurement"]["heldout"]["relative_l1"] for r in rows])
        val_h = np.array([r[val_key]["measurement"]["heldout"]["relative_l1"] for r in rows])
        late_values.extend((val_h[199:] * 100).tolist())
        train_loss = np.array([r["train"]["loss"]["total"] for r in rows])
        val_loss = np.array([r[val_key]["loss"]["total"] for r in rows])
        for ax, values, label in [
            (axes[0, 0], train_loss, name + " train"),
            (axes[0, 1], val_loss, name + " validation"),
            (axes[1, 0], val_h * 100, name),
            (axes[1, 1], val_h * 100, name),
        ]:
            ax.plot(epochs, values, color=colors[name], label=label, linewidth=1.25)
        best = int(np.argmin(val_h))
        facts[name] = {
            "best_epoch_1based": best + 1,
            "best_val_H": float(val_h[best]),
            "last_train_H": float(train_h[-1]),
            "last_val_H": float(val_h[-1]),
            "last_train_total": float(train_loss[-1]),
            "last_val_total": float(val_loss[-1]),
            "val_H_epoch_250": float(val_h[249]),
            "val_H_last25_mean": float(val_h[-25:].mean()),
            "val_H_last25_range": [float(val_h[-25:].min()), float(val_h[-25:].max())],
            "epoch250_to300_relative_reduction": float(1 - val_h[-1] / val_h[249]),
        }
    titles = ["Training objective (raw, unsmoothed)", "Validation objective (raw, unsmoothed)",
              "Held-out projection relative L1", "Held-out projection: final 100 epochs"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch (1-based)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_xlim(200, 300)
    late_low, late_high = min(late_values), max(late_values)
    late_margin = (late_high - late_low) * .12
    axes[1, 1].set_ylim(late_low - late_margin, late_high + late_margin)
    axes[0, 0].set_ylabel("Total objective")
    axes[0, 1].set_ylabel("Total objective")
    axes[1, 0].set_ylabel("Projection relative L1 (%)")
    axes[1, 1].set_ylabel("Projection relative L1 (%)")
    fig.suptitle("Three frozen 300-epoch runs: measurement-domain logs, not 3D loss", fontsize=14)
    fig.savefig(OUT / "training_curves.png", dpi=170)
    plt.close(fig)

    gt = np.array([[.4, .1], [.1, .4]])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    toy = []
    for ax, t, title in zip(axes, [.4, .25, .1], ["Reference", "Uniform alternative", "Swapped alternative"]):
        field = np.array([[t, .5-t], [.5-t, t]])
        assert np.allclose(field.sum(0), [.5, .5])
        assert np.allclose(field.sum(1), [.5, .5])
        l1 = float(np.abs(field-gt).sum()/gt.sum())
        toy.append({"t": t, "field": field.tolist(), "relative_l1_vs_reference": l1,
                    "row_sums": field.sum(1).tolist(), "column_sums": field.sum(0).tolist()})
        im = ax.imshow(field, vmin=0, vmax=.5, cmap="viridis")
        for (i, j), v in np.ndenumerate(field):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=18)
        ax.set_title(f"{title}\nField L1: {l1:.0%}; projection error: 0")
        ax.set_xticks([0, 1], ["col sum 0.5", "col sum 0.5"])
        ax.set_yticks([0, 1], ["row sum 0.5", "row sum 0.5"])
    fig.colorbar(im, ax=axes, shrink=.8, label="Illustrative mole fraction")
    fig.suptitle("2D teaching example: identical row/column projections, different fields")
    fig.savefig(OUT / "projection_ambiguity_toy.png", dpi=170)
    plt.close(fig)
    result = {"scope": "Frozen scalar-log reporting and synthetic algebraic illustration only",
              "training": facts, "toy": toy, "source_sha256": hashes,
              "model_checkpoint_loaded": False, "GT_volume_loaded": False,
              "final_test_used": False}
    (OUT / "report_facts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
