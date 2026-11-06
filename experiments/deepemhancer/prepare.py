"""Half-maps with known ground truth for DeepEMhancer, one per rung of the decoy ladder.

Atom models come from the raw-polymer bookmark (experiments/polymer/compare.py, same
144 A frame); the mirror image of the protein is made here. Each is rendered with the
decoygen renderer (B = 40 A^2, 1.0 A/px, 144^3) and given two independent white-noise
half-maps whose FSC crosses 0.143 at 3.5 A (decoygen.noise). Writes per model:
NAME_half1.mrc, NAME_half2.mrc, NAME_full.mrc (their mean), NAME_truth.mrc (noise-free
signal) and NAME_support.mrc (where the atoms are).

    python experiments/deepemhancer/prepare.py --atoms runs/bookmarks/raw-polymer --out runs/deepemhancer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gemmi
import numpy as np
import torch

from decoygen import atoms, io, noise
from decoygen.fourier import Grid

BOX, APIX, BFACTOR, RESOLUTION = 144, 1.0, 40.0, 3.5
LADDER = ("protein", "protein_mirror", "polymer", "gas", "hardcore")


def load_atoms(path: Path, mirror: bool = False):
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}
    found = [(a.pos.tolist(), lookup[a.element.name]) for ch in st[0] for r in ch for a in r
             if a.element.name in lookup]
    xyz = np.array([f[0] for f in found])
    if mirror:
        xyz[:, 0] = BOX * APIX - xyz[:, 0]
    return (torch.tensor(xyz[:, ::-1].copy(), dtype=torch.float32),
            torch.tensor([f[1] for f in found]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atoms", type=Path, required=True, help="directory with NAME_atoms.pdb")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    grid = Grid(BOX, APIX, "cpu")
    for name in LADDER:
        src = "protein" if name == "protein_mirror" else name
        pos, el = load_atoms(args.atoms / f"{src}_atoms.pdb", mirror=name == "protein_mirror")
        signal = atoms.render(grid, pos, el, BFACTOR)
        h1, h2, info = noise.half_maps(grid, signal, RESOLUTION, 1.0, "white", args.seed)
        support = grid.gaussian_filter(signal, 2.0)
        support = support > 0.2 * support.max()
        for suffix, m in (("half1", h1), ("half2", h2), ("full", (h1 + h2) / 2), ("truth", signal),
                          ("support", support.float())):
            io.write_mrc(args.out / f"{name}_{suffix}.mrc", m.numpy(), APIX)
        print(f"{name}: {len(pos)} atoms, noise sd {info['noise_sd_real']:.4g}", flush=True)


if __name__ == "__main__":
    main()
