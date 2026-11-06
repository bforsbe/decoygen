"""Figure for packing.json: per-structure summary bars and the density-vs-depth profile."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DECOY, PROT, INK, MUTED = "#2a6fdb", "#8a8f98", "#1f2328", "#6b7280"

res = json.loads(Path(sys.argv[1]).read_text())
names = list(res)
is_decoy = [n[:2] in {f"{k} " for k in range(8)} for n in names]   # "0 " .. "7 ": rung names
colors = [DECOY if d else PROT for d in is_decoy]
proteins = [n for n, d in zip(names, is_decoy) if not d]

panels = [("A3_per_atom", "molecular volume per atom (A³)"),
          ("core_packing", "core packing density"),
          ("pocket_fraction", "surface pocket fraction (5 A vs 1.4 A probe)")]
fig = plt.figure(figsize=(13, 8.5), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
for k, (key, title) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    vals = np.array([res[n][key] for n in names])
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.7, edgecolor="white", linewidth=1)
    pv = [res[n][key] for n in proteins]
    ax.axvspan(min(pv), max(pv), color=PROT, alpha=0.15, lw=0)
    ax.set_yticks(y, names if k == 0 else [""] * len(names), fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    lo = 0.55 if key == "core_packing" else 0
    hi = vals.max() * 1.08
    ax.set_xlim(lo, hi)
    for yi, v in zip(y, vals):
        ax.text(v + (hi - lo) * 0.01, yi, f"{v:.2f}" if v < 1 else f"{v:.1f}", va="center", fontsize=8, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, :2])
show = {"3 threaded chain": ("--", DECOY), "5 real side chains": ("-", DECOY), "1 atom gas": (":", DECOY),
        "8 protein (6CVM A)": ("-", INK)}
for n in proteins[1:]:
    sh = res[n]["shells"]
    ax.plot([i for i in range(len(sh))], [s["atoms_per_nm3"] for s in sh], color=PROT, lw=1, alpha=0.6)
for n, (ls, c) in show.items():
    sh = res[n]["shells"]
    ax.plot(range(len(sh)), [s["atoms_per_nm3"] for s in sh], ls=ls, color=c, lw=2, marker="o", ms=5, label=n)
labels = [s["depth_A"] for s in res["3 threaded chain"]["shells"]]
ax.set_xticks(range(len(labels)), [f"{x} A" for x in labels], fontsize=8)
ax.set_ylabel("heavy atoms per nm³", fontsize=9, color=INK)
ax.set_title("atom density vs depth below the molecular surface (grey: the 8 reference proteins)",
             fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0, 1, "blue: decoy rungs\ngrey: real proteins\nshaded band: protein range\n\n"
        "Rung 0 has overlapping atom spheres\n(hard-core spacing < vdW contact),\nso its packing is not comparable.",
        va="top", fontsize=9, color=MUTED)
fig.savefig(sys.argv[2], dpi=150)
