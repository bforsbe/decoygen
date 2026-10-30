"""Summarise a hallucination run directory: metrics, ModelAngelo builds, and a figure.

    python experiments/hallucination/summarise.py runs/hallucination
writes summary.md, summary.json and summary.png into the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gemmi
import matplotlib
import mrcfile
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["protein", "protein_mirror", "decoy_initial"]


def build_stats(d: Path, name: str) -> dict:
    out = {}
    model = d / f"{name}.cif"
    if model.exists():
        st = gemmi.read_structure(str(model))
        lengths = [len(ch) for ch in st[0]]
        ca = [a for ch in st[0] for r in ch for a in r if a.name == "CA"]
        out.update(built_residues=sum(lengths), built_chains=len(lengths),
                   longest_chain=max(lengths, default=0),
                   chains_ge_10=sum(n >= 10 for n in lengths),
                   mean_b=float(np.mean([a.b_iso for a in ca])) if ca else float("nan"))
    else:
        out.update(built_residues=0, built_chains=0, longest_chain=0, chains_ge_10=0)
    see = d / "see_alpha_output" / "see_alpha_output_ca.cif"
    if see.exists():
        out["calpha_points"] = sum(1 for ch in gemmi.read_structure(str(see))[0] for _ in ch)
    log = d.parent / f"{name}.log"
    if not model.exists() and log.exists() and "No Cα atoms were predicted" in log.read_text():
        out["calpha_points"] = 0
    return out


def peak_stats(path: Path) -> dict:
    """Local maxima of the C-alpha probability; confident peaks separate protein from gas."""
    with mrcfile.open(path) as m:
        p = m.data.astype(np.float64)
    peaks = p[(p == ndimage.maximum_filter(p, size=3)) & (p > 0.05)]
    return {"ca_peaks": int(peaks.size), "ca_peaks_above_0.5": int((peaks > 0.5).sum()),
            "ca_prob_sum_p4": float((p**4).sum())}


def main(root: Path):
    metrics = {}
    for f in sorted(root.glob("*_metrics.json")):
        m = json.loads(f.read_text())
        metrics[m["name"]] = m
    names = [n for n in ORDER if n in metrics] + sorted(n for n in metrics if n not in ORDER)
    for n in names:
        metrics[n].update(build_stats(root / "build" / n, n))
        metrics[n].update(peak_stats(root / f"{n}_ca_prob.mrc"))

    cols = [("ca_prob_sum", "Σp", "{:.0f}"), ("ca_prob_sum_p4", "Σp⁴", "{:.1f}"),
            ("ca_peaks", "Cα peaks", "{}"), ("ca_peaks_above_0.5", "peaks >0.5", "{}"),
            ("calpha_points", "MA Cα points", "{}"),
            ("built_residues", "residues built", "{}"), ("built_chains", "chains", "{}"),
            ("longest_chain", "longest chain", "{}"), ("chains_ge_10", "chains ≥10", "{}"),
            ("chirality_clean", "map chirality", "{:+.2e}"),
            ("nn_distance_mean_A", "mean NN dist (Å)", "{:.2f}"),
            ("neighbours_within_1.8A_mean", "neighbours <1.8 Å", "{:.2f}"),
            ("signal_max_over_mean_in_support", "peak/mean density", "{:.1f}"),
            ("displacement_rms_A", "RMS move (Å)", "{:.2f}")]
    lines = ["| state | " + " | ".join(c[1] for c in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for n in names:
        m = metrics[n]
        cells = [fmt.format(m[k]) if k in m and m[k] is not None else "–" for k, _, fmt in cols]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps(
        {n: {k: v for k, v in metrics[n].items() if k != "pair_histogram"} for n in names}, indent=1))
    print("\n".join(lines))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for n in names:
        h = metrics[n]["pair_histogram"]
        ax[0, 0].plot(h["r"], h["density"], label=n, lw=1)
    ax[0, 0].set(xlabel="r (Å)", ylabel="pair density (Å⁻³)", title="Atom pair distances", ylim=(0, None))
    ax[0, 0].legend(fontsize=7)
    for f in sorted(root.glob("*_history.json")):
        hist = json.loads(f.read_text())["history"]
        s = [h["score"] for h in hist]
        if np.isfinite(s).any():
            ax[0, 1].plot([h["step"] for h in hist], s, label=f.name.replace("_history.json", ""))
    for n in ("protein", "protein_mirror", "decoy_initial"):
        if n in metrics:
            ax[0, 1].axhline(metrics[n]["ca_prob_sum"], ls="--", lw=0.8, color="k" if n == "protein" else "grey")
            ax[0, 1].text(0, metrics[n]["ca_prob_sum"], n, fontsize=7, va="bottom")
    ax[0, 1].set(xlabel="step", ylabel="S (stochastic, per step)", title="Objective")
    ax[0, 1].legend(fontsize=7)

    show = [n for n in names if n in ("protein", "decoy_initial") or n.startswith("decoy_hallucinated")][:4]
    tiles = []
    for n in show:
        with mrcfile.open(root / f"{n}_clean.mrc") as m:
            c = m.data[m.data.shape[0] // 2]
        with mrcfile.open(root / f"{n}_ca_prob.mrc") as m:
            p = m.data[m.data.shape[0] // 2]
        tiles.append((n, c, p))
    if tiles:
        ax[1, 0].imshow(np.concatenate([t[1] / t[1].max() for t in tiles], 1), cmap="gray")
        ax[1, 0].set(title="Clean map, central slice: " + " | ".join(t[0] for t in tiles))
        ax[1, 1].imshow(np.concatenate([t[2] for t in tiles], 1), cmap="magma", vmin=0, vmax=1)
        ax[1, 1].set(title="ModelAngelo Cα probability, same slice")
        for a in ax[1]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(root / "summary.png", dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
