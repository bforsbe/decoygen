"""How much more protein-like does a decoy look once its atom distances are right?

Four atom models, rendered identically (decoygen renderer, B = 40 A^2, 1.0 A/px, 144^3):
  protein    6CVM chain A (beta-galactosidase, 108 kDa); not in the reference set
  gas        the default 100 kDa decoy's atoms (v0 Poisson gas, seed 0)
  hardcore   the same atoms pushed apart to protein's non-bonded contact distances
             (no bonds): removes overlaps and clumps only
  polymer    decoygen.polymer: branched bonded tree with protein bond, angle and
             contact distances, built from the same gas
No Gerchberg-Saxton step and no noise, so the maps differ only through the atoms.

    python experiments/polymer/compare.py --out runs/polymer
writes per state: NAME_atoms.pdb (with CONECT for the polymer), NAME_2.0A.mrc,
NAME_3.5A.mrc, and compare.json / compare.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gemmi
import matplotlib
import numpy as np
import torch

from decoygen import atoms, io, polymer, roughness
from decoygen.constants import MASS_PER_HEAVY_ATOM
from decoygen.diagnostics import structure_tensor_stats
from decoygen.envelope import make_envelope
from decoygen.fourier import Grid
from decoygen.params import DecoyParams
from decoygen.turing import turing_field

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTEIN = Path(__file__).parents[1] / "hallucination/data/6CVM.cif"
BOX, APIX, BFACTOR = 144, 1.0, 40.0
RESOLUTIONS = (2.0, 3.5)
COLORS = {"protein": "k", "gas": "tab:red", "hardcore": "tab:orange", "polymer": "tab:blue"}


def decoy_gas(params: DecoyParams, device):
    g = Grid(params.box, params.apix, device)
    rough = roughness.roughness_parameters(params.roughness_D, params.alpha, params.roughness_amplitude)
    mask, _ = make_envelope(
        g, volume=params.volume, alpha=rough["alpha"], roughness_amplitude=rough["roughness_amplitude"],
        compactness=params.compactness, n_domains=params.n_domains, elongation=params.elongation,
        bias=params.envelope_bias, clearance=params.edge_clearance, seed=params.seed)
    turing, _ = turing_field(
        g, mask, wavelength=params.meso_wavelength, quotient=params.turing_quotient,
        fill_fraction=params.fill_fraction, iterations=params.turing_iterations, dt=params.turing_dt,
        seed=params.seed)
    intensity = ((1 - params.meso_strength) + params.meso_strength * turing) * torch.as_tensor(mask, device=device)
    n_atoms = round(params.mass_kda * 1e3 / MASS_PER_HEAVY_ATOM)
    pos, el = atoms.sample_gas(g, intensity, n_atoms, params.seed, "atoms")
    return pos, el, mask


def protein_atoms(device):
    st = gemmi.read_structure(str(PROTEIN))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    xyz = np.array([a.pos.tolist() for r in st[0]["A"] for a in r])
    els = [a.element.name for r in st[0]["A"] for a in r]
    xyz = xyz - (xyz.max(0) + xyz.min(0)) / 2 + BOX * APIX / 2
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}
    return (torch.tensor(xyz[:, ::-1].copy(), dtype=torch.float32, device=device),
            torch.tensor([lookup[e] for e in els], device=device))


def write_pdb(path: Path, pos: torch.Tensor, el: torch.Tensor, edges: np.ndarray | None = None) -> None:
    """HETATM records (one atom per UNK residue, chains of 9999) plus CONECT for the bonds."""
    xyz = pos.detach().cpu().numpy()[:, ::-1]
    lines = []
    for k, (p, e) in enumerate(zip(xyz, el.tolist())):
        name = atoms.ELEMENTS[e]
        chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k // 9999]
        lines.append(f"HETATM{k + 1:5d}  {name:<3s} UNK {chain}{k % 9999 + 1:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00{BFACTOR:6.2f}          {name:>2s}")
    if edges is not None and len(edges):
        partners = [[] for _ in range(len(xyz))]
        for i, j in edges:
            partners[i].append(j)
            partners[j].append(i)
        for i, ps in enumerate(partners):
            for s in range(0, len(ps), 4):
                lines.append(f"CONECT{i + 1:5d}" + "".join(f"{j + 1:5d}" for j in ps[s:s + 4]))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def pair_density(pos: torch.Tensor, rmax=8.0, dr=0.1) -> np.ndarray:
    hist = np.zeros(round(rmax / dr))
    for s in range(0, len(pos), 4096):
        d = torch.cdist(pos[s:s + 4096], pos, compute_mode="donot_use_mm_for_euclid_dist")
        d = d[(d > 1e-6) & (d < rmax)]
        hist += np.histogram(d.cpu().numpy(), bins=len(hist), range=(0, rmax))[0]
    r = (np.arange(len(hist)) + 0.5) * dr
    return hist / len(pos) / (4 * np.pi * r**2 * dr)


def map_stats(grid: Grid, x: torch.Tensor, support: torch.Tensor) -> dict:
    v = (x[support] / x[support].mean()).double()
    z = (v - v.mean()) / v.std()
    st = structure_tensor_stats(grid, x, support, scale=2.0)
    return {"cv": float(v.std()), "skew": float((z**3).mean()), "kurtosis": float((z**4).mean() - 3),
            "p99_over_mean": float(torch.quantile(v.float()[:: max(1, len(v) // 2_000_000)], 0.99)),
            **{f"st_{k}": float(val) for k, val in st.items() if not isinstance(val, (list, dict))}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    grid = Grid(BOX, APIX, dev)
    ref = polymer.load_reference()

    params = DecoyParams(seed=args.seed)
    gas_pos, gas_el, mask = decoy_gas(params, dev)
    env = polymer.Envelope(mask, params.apix, dev)
    offset = (BOX * APIX - params.box * params.apix) / 2

    t = time.perf_counter()
    hc_pos, hc_info = polymer.relax(gas_pos, gas_el.cpu().numpy(), np.zeros((0, 2), dtype=np.int64), ref,
                                    envelope=env, steps=args.steps, seed=args.seed, log=print)
    t_hc = time.perf_counter() - t
    t = time.perf_counter()
    pol_pos, pol_el, edges, pol_info = polymer.build_polymer(gas_pos, ref, envelope=env, seed=args.seed,
                                                             steps=args.steps, log=print)
    t_pol = time.perf_counter() - t
    prot_pos, prot_el = protein_atoms(dev)

    states = {
        "protein": (prot_pos, prot_el, None, {}),
        "gas": (gas_pos + offset, gas_el, None, {}),
        "hardcore": (hc_pos + offset, gas_el, None, {**hc_info, "time_s": t_hc}),
        "polymer": (pol_pos + offset, pol_el, edges, {**pol_info, "time_s": t_pol}),
    }
    results, slices = {}, {}
    for name, (pos, el, e, info) in states.items():
        with torch.no_grad():
            signal = atoms.render(grid, pos, el, BFACTOR)
            support = grid.gaussian_filter(signal, 2.0)
            support = support > 0.2 * support.max()
            maps = {d: grid.cosine_lowpass(signal, 1 / d, 0.1 / d) for d in RESOLUTIONS}
            power = grid.power_spectrum(signal) / len(pos)
        write_pdb(args.out / f"{name}_atoms.pdb", pos, el, e)
        for d, m in maps.items():
            io.write_mrc(args.out / f"{name}_{d:.1f}A.mrc", m.cpu().numpy(), APIX)
        results[name] = {
            "n_atoms": len(pos),
            "pair_density": pair_density(pos).tolist(),
            "power_per_atom": power.cpu().numpy().tolist(),
            **{f"map_{d:.1f}A": map_stats(grid, m, support) for d, m in maps.items()},
            "info": {k: v for k, v in info.items() if k != "history"},
        }
        c = int(pos[:, 0].mean() / APIX)               # slice through the atoms' centroid
        slices[name] = {d: m[c].cpu().numpy() for d, m in maps.items()}
        print(name, {k: round(v, 3) for k, v in results[name]["map_3.5A"].items()}, flush=True)
    io.write_json(args.out / "compare.json", {"results": results, "reference_pair_density": ref["pair_density"]})
    plot(args.out, results, slices, ref)


def plot(out: Path, results: dict, slices: dict, ref: dict):
    names = list(results)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[0, :2])
    r = (np.arange(80) + 0.5) * 0.1
    ax.plot(ref["pair_density"]["r"], ref["pair_density"]["density"], color="0.6", lw=3, alpha=0.6,
            label="reference set (8 proteins)")
    for n in names:
        ax.plot(r, results[n]["pair_density"], color=COLORS[n], lw=1.2, label=n)
    ax.set(xlabel="r (Å)", ylabel="pairs per atom per Å³", title="Atom pair distances", xlim=(0, 8), ylim=(0, 0.4))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    s = np.arange(len(results["protein"]["power_per_atom"])) / (BOX * APIX)
    for n in names:
        ax.semilogy(s[1:], results[n]["power_per_atom"][1:], color=COLORS[n], lw=1, label=n)
    for d in RESOLUTIONS:
        ax.axvline(1 / d, color="0.7", ls=":", lw=0.8)
    ax.set(xlabel="s (1/Å)", ylabel="power per atom", title="Radial power spectrum (B = 40)")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 3])
    keys = ["cv", "skew", "kurtosis", "st_linearity_mean", "st_planarity_mean"]
    x = np.arange(len(keys))
    for k, n in enumerate(names):
        ax.bar(x + (k - 1.5) * 0.2, [results[n]["map_3.5A"].get(key, np.nan) for key in keys], 0.2,
               color=COLORS[n], label=n)
    ax.set_xticks(x, ["CV", "skew", "kurtosis", "linearity", "planarity"], fontsize=8)
    ax.set(title="Density statistics inside the molecule, 3.5 Å")
    ax.axhline(0, color="k", lw=0.5)

    for row, d in ((1, 2.0), (2, 3.5)):
        for col, n in enumerate(names):
            ax = fig.add_subplot(gs[row, col])
            img = slices[n][d]
            ax.imshow(img, cmap="gray", vmin=0, vmax=np.quantile(img, 0.999))
            ax.set_title(f"{n}, {d:.1f} Å", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=110)


if __name__ == "__main__":
    main()
