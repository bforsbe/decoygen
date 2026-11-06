"""Score DeepEMhancer outputs against the known ground truth, per rung of the decoy ladder.

    python experiments/deepemhancer/analyse.py runs/deepemhancer
writes summary.md, summary.json, slices.png, chains.png and fidelity.png.

Per map (input = mean of the half-maps; outputs of wideTarget / tightTarget / highRes):
  cc_truth          correlation with the noise-free truth low-passed to 3.5 A, inside
                    the support dilated by 3 A
  outside_fraction  share of the map's squared density outside the dilated support
  kurtosis, linearity, chirality   texture inside the support (decoygen.diagnostics)
  ModelAngelo build_no_seq: residues, chains, longest chain, N50 (as for genome assemblies: half of
  the built residues are in chains at least this long), residue-weighted mean chain length,
  % of residues in chains >= 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gemmi
import matplotlib
import mrcfile
import numpy as np
import torch

from decoygen.diagnostics import chirality, structure_tensor_stats
from decoygen.fourier import Grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADDER = ["protein", "protein_mirror", "polymer", "gas", "hardcore"]
LABELS = {"protein": "protein", "protein_mirror": "mirrored\nprotein", "polymer": "raw\npolymer",
          "gas": "atom gas", "hardcore": "hard-core\nliquid"}
VARIANTS = ["full", "deh_wideTarget", "deh_tightTarget", "deh_highRes"]
VLABELS = {"full": "input (half-map mean)", "deh_wideTarget": "DeepEMhancer wideTarget",
           "deh_tightTarget": "DeepEMhancer tightTarget", "deh_highRes": "DeepEMhancer highRes"}
# validated categorical slots 1-3 (dataviz reference palette, light mode) + neutral for the input
VCOLORS = {"full": "#9a9893", "deh_wideTarget": "#2a78d6", "deh_tightTarget": "#eb6834",
           "deh_highRes": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RESOLUTION = 3.5


def load(path: Path) -> torch.Tensor:
    with mrcfile.open(path, permissive=True) as m:
        return torch.as_tensor(m.data.astype(np.float32).copy())


def build_stats(d: Path, name: str) -> dict:
    model = d / name / f"{name}.cif"
    if not model.exists():
        return {"residues": 0, "chains": 0, "longest_chain": 0, "n50": 0, "weighted_mean_length": 0.0,
                "in_chains_ge10": 0.0}
    lengths = np.sort([len(ch) for ch in gemmi.read_structure(str(model))[0]])[::-1]
    return {"residues": int(lengths.sum()), "chains": len(lengths), "longest_chain": int(lengths.max()),
            "n50": n50(lengths), "weighted_mean_length": float((lengths**2).sum() / lengths.sum()),
            "in_chains_ge10": float(lengths[lengths >= 10].sum() / lengths.sum())}


def n50(lengths) -> int:
    """Chain length L such that chains of length >= L hold half of all built residues."""
    lengths = np.sort(np.asarray(lengths))[::-1]
    cs = np.cumsum(lengths)
    return int(lengths[np.searchsorted(cs, cs[-1] / 2)])


def score(grid: Grid, x: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> dict:
    wide = grid.gaussian_filter(support.float(), 3.0) > 0.05
    t35 = grid.cosine_lowpass(truth, 1 / RESOLUTION, 0.1 / RESOLUTION)
    a, b = x[wide].double(), t35[wide].double()
    cc = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))
    v = x[support].double()
    z = (v - v.mean()) / v.std()
    st = structure_tensor_stats(grid, x, support, scale=2.0)
    return {"cc_truth": cc,
            "outside_fraction": float((x[~wide] ** 2).sum() / (x**2).sum()),
            "kurtosis": float((z**4).mean() - 3), "linearity": st["linearity_mean"],
            "chirality": chirality(grid, x, support)}


def main(root: Path):
    results = {}
    for name in LADDER:
        truth = load(root / f"{name}_truth.mrc")
        support = load(root / f"{name}_support.mrc") > 0.5
        grid = Grid(truth.shape[0], 1.0, "cpu")
        for v in VARIANTS:
            path = root / f"{name}_{v}.mrc"
            if not path.exists():
                continue
            x = load(path)
            if x.shape != truth.shape:
                print(f"skip {path.name}: shape {tuple(x.shape)}")
                continue
            results[f"{name}_{v}"] = {"model": name, "variant": v, **score(grid, x, truth, support),
                                      **build_stats(root / "build", f"{name}_{v}")}
    (root / "summary.json").write_text(json.dumps(results, indent=1))
    cols = ["cc_truth", "outside_fraction", "kurtosis", "linearity", "chirality",
            "residues", "chains", "longest_chain", "n50", "weighted_mean_length", "in_chains_ge10"]
    lines = ["| map | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, r in results.items():
        cells = [f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_slices(root)
    plot_metric(root, results, "n50", "ModelAngelo chain N50 (residues)", "chains.png", percent=False)
    plot_metric(root, results, "cc_truth", "Agreement with the known truth (correlation, 3.5 Å)",
                "fidelity.png", percent=False)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_metric(root, results, key, title, fname, percent):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    style(ax)
    width = 0.19
    x = np.arange(len(LADDER))
    for k, v in enumerate(VARIANTS):
        vals = [results.get(f"{n}_{v}", {}).get(key, np.nan) for n in LADDER]
        vals = np.array(vals) * (100 if percent else 1)
        pos = x + (k - 1.5) * (width + 0.02)
        ax.bar(pos, vals, width, color=VCOLORS[v], label=VLABELS[v], edgecolor="white", linewidth=1)
        for p, val in zip(pos, vals):
            if np.isfinite(val):
                ax.text(p, val, f"{val:.0f}" if percent else f"{val:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_xticks(x, [LABELS[n] for n in LADDER], fontsize=11, color=INK)
    ax.set_ylabel("%" if percent else "correlation", color=INK2)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    if percent:
        ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(root / fname, dpi=200)
    plt.close(fig)


def plot_slices(root):
    cols = [("truth", "truth (noise-free)")] + [(v, VLABELS[v].replace("DeepEMhancer ", "DeepEMh.\n"))
                                                for v in VARIANTS]
    fig, axes = plt.subplots(len(LADDER), len(cols), figsize=(2.3 * len(cols), 2.3 * len(LADDER)))
    for i, name in enumerate(LADDER):
        support = load(root / f"{name}_support.mrc")
        c = int(torch.nonzero(support.sum((1, 2))).float().mean())
        for j, (v, label) in enumerate(cols):
            ax = axes[i, j]
            ax.axis("off")
            path = root / f"{name}_{v}.mrc"
            if not path.exists():
                continue
            img = load(path)[c].numpy()
            if v == "truth":
                img = Grid(img.shape[0], 1.0, "cpu").cosine_lowpass(load(path), 1 / RESOLUTION,
                                                                    0.1 / RESOLUTION)[c].numpy()
            lo, hi = np.quantile(img, [0.01, 0.999])
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            if i == 0:
                ax.set_title(label, fontsize=10, color=INK)
            if j == 0:
                ax.text(-0.08, 0.5, LABELS[name], transform=ax.transAxes, ha="right", va="center",
                        fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(root / "slices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
