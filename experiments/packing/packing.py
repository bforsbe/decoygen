"""Packing efficiency of the ladder rungs against real proteins.

All measures are computed on one grid (default 0.5 A) from heavy atoms with united-atom radii
(hydrogens folded in; Chothia 1975 values, aliphatic C), so decoys and proteins are treated identically.
Absolute packing densities depend on the radii; compare rows, not against literature to 2 decimals.

  molecular volume  inside the solvent-excluded surface of a 1.4 A probe (Connolly); pockets the
                    probe reaches from outside are outside, enclosed probe-sized voids are inside
  packing           vdW volume (union of atom spheres) / molecular volume
  core packing      the same over power-Voronoi cells (radius-weighted) of buried atoms only, i.e. cells
                    that do not touch the molecular surface (Richards 1974 style)
  cavities          enclosed voids that hold a 1.4 A probe (count per 1000 atoms, volume fraction)
  pocket fraction   volume between the 1.4 A surface and a 5 A probe surface, over the 5 A volume
  depth profile     packing and atom number density in shells of depth below the molecular surface
  contact spread    atoms within 8 A of atoms at least 6 A deep: mean and coefficient of variation

usage: python packing.py OUT_DIR NAME=FILE [NAME=FILE ...]   (PDB or mmCIF; decoygen models fine)
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

RADII = {"C": 1.87, "N": 1.65, "O": 1.40, "S": 1.85}   # Chothia 1975 united atom
PROBE, BIG_PROBE, H = 1.4, 5.0, 0.5
SHELLS = [0, 2, 4, 6, 8, 12, 16, 99]


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom positions and radii; polymer atoms only for mmCIF (no waters or ligands)."""
    if path.endswith(".cif"):
        import gemmi
        st = gemmi.read_structure(path)
        st.remove_hydrogens()
        st.remove_ligands_and_waters()
        pos, el = [], []
        for ch in st[0]:
            for res in ch:
                for a in res:
                    pos.append(a.pos.tolist())
                    el.append("S" if a.element.name == "Se" else a.element.name)
    else:
        pos, el = [], []
        for line in Path(path).read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                e = line[76:78].strip()
                if e in RADII:
                    pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    el.append(e)
    keep = [e in RADII for e in el]
    return np.array(pos)[keep], np.array([RADII[e] for e, k in zip(el, keep) if k])


def analyse(pos: np.ndarray, rad: np.ndarray) -> dict:
    pad = BIG_PROBE * 2 + rad.max() + 2
    lo = pos.min(0) - pad
    shape = tuple(np.ceil((pos.max(0) + pad - lo) / H).astype(int))
    tree = cKDTree(pos)
    n_vox = int(np.prod(shape))
    surf = np.full(n_vox, np.inf, dtype=np.float32)       # distance to nearest atom surface
    owner = np.full(n_vox, -1, dtype=np.int32)            # power-diagram cell
    idx = np.indices(shape).reshape(3, -1)
    for s in range(0, n_vox, 4_000_000):
        pts = lo + H * idx[:, s:s + 4_000_000].T
        d, j = tree.query(pts, k=12, distance_upper_bound=BIG_PROBE + rad.max() + 1, workers=16)
        ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        surf[s:s + len(pts)] = np.where(ok, d - rad[jj], np.inf).min(1)
        power = np.where(ok, d ** 2 - rad[jj] ** 2, np.inf)
        best = power.argmin(1)
        owner[s:s + len(pts)] = np.where(ok[np.arange(len(pts)), best], jj[np.arange(len(pts)), best], -1)
    surf, owner = surf.reshape(shape), owner.reshape(shape)
    vdw = surf < 0

    def inside(probe):
        acc = surf >= probe                                  # where the probe centre fits
        lab, _ = ndimage.label(acc)
        border = np.unique(np.concatenate([lab[[0, -1]].ravel(), lab[:, [0, -1]].ravel(),
                                           lab[:, :, [0, -1]].ravel()]))
        ext = np.isin(lab, border[border > 0])
        outside = ndimage.distance_transform_edt(~ext) * H < probe
        return ~outside, acc & ~ext

    mol, cav_centres = inside(PROBE)
    big, _ = inside(BIG_PROBE)
    _, n_cav = ndimage.label(cav_centres)
    cav_vol = (ndimage.distance_transform_edt(~cav_centres) * H < PROBE) & mol & ~vdw
    v = H ** 3
    depth = ndimage.distance_transform_edt(mol) * H
    a_idx = tuple(np.clip(np.round((pos - lo) / H).astype(int), 0, np.array(shape) - 1).T)
    a_depth = depth[a_idx]

    # power-Voronoi cells inside the molecular volume; buried = cell never reaches the surface layer
    o = owner[mol]
    cell = np.bincount(o, minlength=len(pos)) * v
    cell_vdw = np.bincount(owner[mol & vdw], minlength=len(pos)) * v
    min_depth = np.full(len(pos), np.inf)
    np.minimum.at(min_depth, o, depth[mol])
    buried = min_depth > 1.0

    shells = []
    for a, b in pairwise(SHELLS):
        m = mol & (depth > a) & (depth <= b)
        if m.sum() * v < 500:
            continue
        shells.append({"depth_A": f"{a}-{b if b < 99 else ''}", "volume_A3": float(m.sum() * v),
                       "packing": float((m & vdw).sum() / m.sum()),
                       "atoms_per_nm3": float(((a_depth > a) & (a_depth <= b)).sum() / (m.sum() * v) * 1000)})

    deep = a_depth >= 6
    contacts = np.array([len(x) - 1 for x in tree.query_ball_point(pos[deep], 8.0, workers=16)])
    return {
        "atoms": len(pos),
        "molecular_volume_A3": float(mol.sum() * v),
        "A3_per_atom": float(mol.sum() * v / len(pos)),
        "packing": float((mol & vdw).sum() / mol.sum()),
        "core_packing": float(cell_vdw[buried].sum() / cell[buried].sum()) if buried.any() else None,
        "buried_atom_fraction": float(buried.mean()),
        "core_A3_per_atom": float(cell[buried].mean()) if buried.any() else None,
        "cavities_per_1000_atoms": 1000 * n_cav / len(pos),
        "cavity_volume_fraction": float(cav_vol.sum() / mol.sum()),
        "pocket_fraction": float((big & ~mol).sum() / big.sum()),
        "max_depth_A": float(depth.max()),
        "deep_atoms": int(deep.sum()),
        "contacts_8A_mean": float(contacts.mean()) if len(contacts) else None,
        "contacts_8A_cv": float(contacts.std() / contacts.mean()) if len(contacts) else None,
        "shells": shells,
    }


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for item in argv[1:]:
        name, path = item.split("=", 1)
        r = analyse(*load(path))
        r["file"] = path
        res[name] = r
        print(f"{name:28s} atoms {r['atoms']:6d}  packing {r['packing']:.3f}  core {r['core_packing'] or 0:.3f}  "
              f"A3/atom {r['A3_per_atom']:.1f}  cav/1000 {r['cavities_per_1000_atoms']:.2f}  "
              f"pocket {r['pocket_fraction']:.3f}  contacts {r['contacts_8A_mean'] or 0:.1f} "
              f"cv {r['contacts_8A_cv'] or 0:.3f}", flush=True)
        (out / "packing.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
