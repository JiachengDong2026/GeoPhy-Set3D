"""Train-only concentration diagnostics under a hypothetical steady-state premise.

These are spatial descriptors, not a test of temporal steadiness or CFD residuals.
Inputs are read-only; no projections, checkpoints, validation or test arrays opened.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1")
OUT = ROOT / "outputs/research/assumed_steady_plume_v1"
LENGTH = 0.6


def weights(n):
    w = np.full(n, LENGTH / (n - 1), dtype=np.float64)
    w[[0, -1]] *= 0.5
    return w


def describe(c):
    assert c.ndim == 3 and len(set(c.shape)) == 1
    assert np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1
    n = c.shape[0]
    w = weights(n)
    coords = np.linspace(0, LENGTH, n)
    # Positive endpoint quadrature, with double precision accumulation.
    xz_integral = (c * w[:, None, None]).sum(axis=0, dtype=np.float64)
    section = (xz_integral * w[None, :]).sum(axis=1)
    integral = float(section @ w)
    assert integral > 0
    face_values = {
        "x0": c[0, :, :], "x1": c[-1, :, :],
        "y0": c[:, 0, :], "y1": c[:, -1, :],
        "z0": c[:, :, 0], "z1": c[:, :, -1],
    }
    faces = {}
    for name, values in face_values.items():
        area = w[:, None] * w[None, :]
        faces[name + "_max"] = float(values.max())
        faces[name + "_integral_m2"] = float((values * area).sum())
        faces[name + "_area_C_ge_0p01_m2"] = float(((values >= .01) * area).sum())
    return {
        "integral_C_m3": integral,
        "peak": float(c.max()),
        "centroid_y_m": float((section * coords) @ w / integral),
        "volume_C_ge_0p01_m3": float(
            np.einsum("ijk,i,j,k->", c >= .01, w, w, w, optimize=True)),
        "top_to_mean_section_ratio": float(section[-1] / (integral / LENGTH)),
        **faces,
    }, section


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(["min", "p25", "median", "p75", "max"],
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def main():
    index_path = POOL / "source_index.jsonl"
    split_path = POOL / "splits/train_cases.txt"
    index_bytes = index_path.read_bytes()
    all_records = [json.loads(s) for s in index_bytes.splitlines() if s.strip()]
    records = [r for r in all_records if r["split"] == "train"]
    train_ids = set(split_path.read_text().split())
    assert len(records) == 166 and {r["case_id"] for r in records} == train_ids
    rows, profiles, reads = [], [], []
    for r in records:
        source = Path(r["source_target_path"])
        target = POOL / r["gt_volume_64_path"]
        raw128 = source.read_bytes()
        assert hashlib.sha256(raw128).hexdigest() == r["source_gt_sha256"]
        c128 = np.load(source, allow_pickle=False)
        c64 = np.load(target, allow_pickle=False)
        assert c128.shape == (128,) * 3 and c64.shape == (64,) * 3
        d128, section = describe(c128)
        d64, _ = describe(c64)
        row = {"case_id": r["case_id"], "split": "train", **r["condition_meta"],
               **d128,
               "integral64_relative_change": d64["integral_C_m3"] / d128["integral_C_m3"] - 1,
               "peak64_to_peak128": d64["peak"] / d128["peak"],
               "centroid64_minus128_mm": 1000 * (d64["centroid_y_m"] - d128["centroid_y_m"])}
        rows.append(row)
        profiles.append(section)
        reads.append({"case_id": r["case_id"], "source": str(source),
                      "source_sha256": r["source_gt_sha256"], "target": str(target),
                      "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    metric_names = ["integral_C_m3", "peak", "centroid_y_m", "top_to_mean_section_ratio",
                    "integral64_relative_change", "peak64_to_peak128", "centroid64_minus128_mm"]
    summary = {
        "premise": "Hypothetical instantaneous steady state at t=2 s; not verified here",
        "scope": "166 train cases only; concentration arrays 128^3 and 64^3",
        "quadrature": "XYZ endpoint trapezoidal weights on [0,0.6]^3 m",
        "integral_C_is_mass": False,
        "PDE_or_temporal_residual_computed": False,
        "validation_or_test_arrays_opened": False,
        "wind_counts": dict(Counter(r["wind_speed_m_s"] for r in rows)),
        "overall": {k: stats([r[k] for r in rows]) for k in metric_names},
        "boundary_cases_at_C_ge_0p01": {
            face: sum(r[face + "_max"] >= .01 for r in rows)
            for face in ["x0", "x1", "y0", "y1", "z0", "z1"]},
        "by_wind_descriptive_unmatched": {
            str(wind): {"n": sum(r["wind_speed_m_s"] == wind for r in rows),
                        **{k: stats([r[k] for r in rows if r["wind_speed_m_s"] == wind])
                           for k in metric_names}}
            for wind in sorted({r["wind_speed_m_s"] for r in rows})},
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "train_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "array_reads.json").write_text(json.dumps(reads, indent=2) + "\n")
    np.savez_compressed(OUT / "height_profiles.npz", case_ids=np.array([r["case_id"] for r in rows]),
                        y_m=np.linspace(0, LENGTH, 128), integral_C_dx_dz_m2=np.array(profiles))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    y = np.linspace(0, LENGTH, 128)
    for wind in sorted({r["wind_speed_m_s"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["wind_speed_m_s"] == wind]
        p = np.array([profiles[i] / (rows[i]["integral_C_m3"] / LENGTH) for i in ids])
        q25, median, q75 = np.quantile(p, [.25, .5, .75], axis=0)
        line, = axes[0].plot(y, median, label=f"{wind:g} m/s (n={len(ids)})")
        axes[0].fill_between(y, q25, q75, color=line.get_color(), alpha=.15)
    axes[0].set(xlabel="Height y (m)", ylabel="I(y) / mean-height I",
                title="Concentration sections: median / IQR")
    axes[0].legend(fontsize=8)
    for ax, key, title in [(axes[1], "peak64_to_peak128", "64 / 128 peak ratio"),
                            (axes[2], "integral64_relative_change", "Integral change: 64 vs 128 (%)")]:
        values = np.array([r[key] for r in rows])
        ax.hist(values * (100 if key == "integral64_relative_change" else 1), bins=25)
        ax.set(title=title, ylabel="Train cases")
    fig.suptitle("Spatial diagnostics only: no temporal or flux residual test")
    fig.savefig(OUT / "concentration_diagnostics.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
