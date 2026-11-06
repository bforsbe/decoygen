"""Collect every test run on every rung of runs/ladder into tables and an N50 figure.

    python experiments/ladder/summarise.py

Per rung (whatever exists so far):
  atoms      atoms, bonds, connected pieces (bonds from CONECT)
  detector   ModelAngelo's own frame (1.5 A/px, its noise model): C-alpha CNN probability sum;
             full build_no_seq: confident C-alpha points, residues, chains, longest, N50
  maps       evaluation frame (1.0 A/px, half-maps at 3.5 A), input and DeepEMhancer x3:
             agreement with truth, texture (experiments/deepemhancer/analyse.py), and the same
             ModelAngelo build statistics
Writes <rung>/results.json, rewrites the Results section of <rung>/RUNG.md, and writes
runs/ladder/README.md, summary.json, n50.png.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import gemmi
import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "deepemhancer"))
import analyse
from rungs import ROOT, RUNGS, rung_dir

from decoygen import atoms, io
from decoygen.fourier import Grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["full", "deh_wideTarget", "deh_tightTarget", "deh_highRes"]
SHORT = {"full": "input", "deh_wideTarget": "DeepEMh wide", "deh_tightTarget": "DeepEMh tight",
         "deh_highRes": "DeepEMh highRes"}


def build(cif: Path) -> dict | None:
    """ModelAngelo build statistics, or None if the build was never run."""
    if not cif.parent.exists():
        return None
    out = analyse.build_stats(cif.parent.parent, cif.stem)
    see = cif.parent / "see_alpha_output" / "see_alpha_output_ca.cif"
    if see.exists():
        out["calpha_points"] = sum(len(ch) for ch in gemmi.read_structure(str(see))[0])
    return out


def atom_stats(d: Path) -> dict:
    src = sorted(d.glob("atoms.*"))
    if not src:
        return {}
    if src[0].suffix != ".pdb":
        return {"atoms": len(analyse_atoms(src[0]))}
    pos, _, edges = io.read_pdb(src[0], atoms.ELEMENTS)
    from scipy import sparse
    from scipy.sparse import csgraph
    n = len(pos)
    pieces = csgraph.connected_components(
        sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)), directed=False)[0]
    return {"atoms": n, "bonds": len(edges), "pieces": int(pieces) if len(edges) else n}


def analyse_atoms(path: Path) -> list:
    st = gemmi.read_structure(str(path))
    return [a for ch in st[0] for r in ch for a in r]


def rung_results(d: Path) -> dict:
    res = {"atoms": atom_stats(d)}
    det = d / "detector"
    if (det / "rung_metrics.json").exists():
        m = json.loads((det / "rung_metrics.json").read_text())
        res["detector"] = {"ca_prob_sum": m["ca_prob_sum"], "ca_prob_max": m["ca_prob_max"],
                           "build": build(det / "build" / "rung" / "rung.cif")}
    maps = d / "maps"
    if (maps / "truth.mrc").exists():
        # map scores are slow (texture statistics on 144^3): reuse them while the map is unchanged
        cache = d / "results.json"
        old = json.loads(cache.read_text()).get("maps", {}) if cache.exists() else {}
        truth = support = grid = None
        res["maps"] = {}
        for v in VARIANTS:
            path = maps / f"{v}.mrc"
            if not path.exists():
                continue
            if v in old and path.stat().st_mtime < cache.stat().st_mtime:
                s = {k: x for k, x in old[v].items() if k != "build"}
            else:
                if truth is None:
                    truth = analyse.load(maps / "truth.mrc")
                    support = analyse.load(maps / "support.mrc") > 0.5
                    grid = Grid(truth.shape[0], 1.0, "cpu")
                with torch.no_grad():
                    s = analyse.score(grid, analyse.load(path), truth, support)
            res["maps"][v] = {**s, "build": build(d / "modelangelo" / v / f"{v}.cif")}
    return res


def fmt(x, digits=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def rung_table(res: dict) -> str:
    lines = []
    a = res.get("atoms", {})
    if a:
        lines.append("Atoms: " + ", ".join(f"{k} {v}" for k, v in a.items()) + "\n")
    rows = []
    if "detector" in res:
        b = res["detector"]["build"] or {}
        rows.append(("detector frame (ModelAngelo's 1.5 A/px)", fmt(res["detector"]["ca_prob_sum"], 4), "–",
                     b.get("calpha_points"), b.get("residues"), b.get("chains"), b.get("longest_chain"),
                     b.get("n50")))
    for v, s in res.get("maps", {}).items():
        b = s["build"] or {}
        rows.append((f"evaluation frame, {SHORT[v]}", "–", fmt(s["cc_truth"]), b.get("calpha_points"),
                     b.get("residues"), b.get("chains"), b.get("longest_chain"), b.get("n50")))
    if rows:
        lines.append("| map | C-alpha CNN sum p | cc with truth | confident C-alpha | residues built | chains "
                     "| longest | N50 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def n50_of(res, v):
    if v == "detector":
        b = (res.get("detector") or {}).get("build")
    else:
        b = ((res.get("maps") or {}).get(v) or {}).get("build")
    return None if b is None else b.get("n50")


def plot(results: dict, path: Path):
    groups = [("main", "Protein-likeness ladder"), ("sym", "Symmetric assemblies"),
              ("branch", "Branch: optimised against the C-alpha CNN")]
    groups = [g for g in groups if any(r.group == g[0] and r.id in results for r in RUNGS)]
    widths = [max(3, sum(r.group == g and r.id in results for r in RUNGS)) for g, _ in groups]
    fig, axes = plt.subplots(1, len(groups), figsize=(0.95 * sum(widths) + 2, 5.0),
                             gridspec_kw={"width_ratios": widths}, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (group, title) in zip(axes, groups):
        ids = [r.id for r in RUNGS if r.group == group and r.id in results]
        analyse.style(ax)
        x = np.arange(len(ids))
        width = 0.19
        for k, v in enumerate(VARIANTS):
            vals = np.array([np.nan if n50_of(results[i], v) is None else n50_of(results[i], v) for i in ids],
                            dtype=float)
            pos = x + (k - 1.5) * (width + 0.02)
            ax.bar(pos, vals, width, color=analyse.VCOLORS[v], label=analyse.VLABELS[v], edgecolor="white",
                   linewidth=1)
            for p, val in zip(pos, vals):
                if np.isfinite(val):                       # N50 0 = no model: label it at the axis floor
                    ax.text(p, max(val, 1) * 1.05, f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                            color=analyse.INK2)
        ax.set_yscale("log")
        ax.set_ylim(1, 400)
        titles = {r.id: r.title for r in RUNGS}
        labels = [i.split("_", 1)[0] + "\n" + textwrap.fill(re.sub(r"^C\d+r? ", "", titles[i]), 12) for i in ids]
        ax.set_xticks(x, labels, fontsize=8.5,
                      color=analyse.INK)
        ax.set_title(title, loc="left", fontsize=12, color=analyse.INK)
    axes[0].set_ylabel("ModelAngelo chain N50 (residues)", color=analyse.INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    results = {}
    for r in RUNGS:
        d = rung_dir(r)
        if not d.exists():
            continue
        res = rung_results(d)
        results[r.id] = res
        (d / "results.json").write_text(json.dumps(res, indent=1))
        md = (d / "RUNG.md").read_text().split("\n## Results")[0].rstrip() + "\n"
        (d / "RUNG.md").write_text(md + "\n## Results\n\n" + rung_table(res))
        print(f"{r.id}: N50 detector {n50_of(res, 'detector')}, input {n50_of(res, 'full')}", flush=True)
    (ROOT / "summary.json").write_text(json.dumps(results, indent=1))

    lines = ["# Protein-likeness ladder\n",
             ("Each rung adds one named piece of protein knowledge; the branch holds objects optimised "
             "against ModelAngelo's C-alpha CNN instead. One directory per rung, with its atoms, maps, "
             "ModelAngelo builds and `RUNG.md` (what it adds, how it was made, every result). "
             "Regenerate with `python experiments/ladder/summarise.py`.\n"),
             ("N50: the chain length such that half of all built residues are in chains at least that long "
             "(as for genome assemblies). Detector frame = ModelAngelo's own 1.5 A/px frame and noise model; "
             "evaluation frame = 1.0 A/px half-maps whose FSC crosses 0.143 at 3.5 A, input and after "
             "DeepEMhancer.\n"),
             ("| rung | adds | atoms | C-alpha CNN sum p | N50 detector | N50 input | N50 DeepEMh wide / tight / "
             "highRes | cc truth input |"),
             "|---|---|---|---|---|---|---|---|"]
    for r in RUNGS:
        if r.id not in results:
            continue
        res = results[r.id]
        det = (res.get("detector") or {}).get("ca_prob_sum")
        cc = ((res.get("maps") or {}).get("full") or {}).get("cc_truth")
        deh = " / ".join(fmt(n50_of(res, v)) for v in VARIANTS[1:])
        lines.append(f"| [{r.id}]({r.group}/{r.id}/RUNG.md) | {r.adds} | {fmt(res['atoms'].get('atoms'))} | "
                     f"{fmt(det, 4)} | {fmt(n50_of(res, 'detector'))} | {fmt(n50_of(res, 'full'))} | {deh} | "
                     f"{fmt(cc)} |")
    lines.append("\n![N50 per rung](n50.png)\n")
    (ROOT / "README.md").write_text("\n".join(lines))
    plot(results, ROOT / "n50.png")
    print(f"wrote {ROOT}/README.md, summary.json, n50.png")


if __name__ == "__main__":
    main()
