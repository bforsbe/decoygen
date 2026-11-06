"""MRC and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np


class MRCHeaderError(RuntimeError):
    pass


def write_mrc(path: str | Path, data: np.ndarray, apix: float, labels=()) -> None:
    """float32 (mode 2), origin 0, axis order 1/2/3; array indexed [z, y, x]; up to 10 text labels
    of 80 characters."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = apix
        m.header.origin = (0.0, 0.0, 0.0)
        m.header.nxstart = m.header.nystart = m.header.nzstart = 0
        labels = list(labels)[:10]
        if labels:                                     # else keep mrcfile's own label
            for k in range(10):
                m.header.label[k] = labels[k].encode("ascii", "replace")[:80] if k < len(labels) else b""
            m.header.nlabl = len(labels)
        m.update_header_stats()
    check_mrc(path, data.shape, apix)


def check_mrc(path: str | Path, shape, apix: float) -> None:
    with mrcfile.open(path, permissive=False) as m:
        h = m.header
        problems = []
        if int(h.mode) != 2:
            problems.append(f"mode {int(h.mode)} != 2")
        if (int(h.mapc), int(h.mapr), int(h.maps)) != (1, 2, 3):
            problems.append("axis order not 1/2/3")
        if m.data.shape != tuple(shape):
            problems.append(f"shape {m.data.shape} != {tuple(shape)}")
        vs = m.voxel_size
        if not np.allclose([vs.x, vs.y, vs.z], apix, rtol=1e-5):
            problems.append(f"voxel size {vs} != {apix}")
        if any(float(o) != 0.0 for o in (h.origin.x, h.origin.y, h.origin.z)):
            problems.append("origin not 0")
    if problems:
        raise MRCHeaderError(f"{path}: " + "; ".join(problems))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


def write_pdb(path: str | Path, pos_zyx, elements, element_names, edges=None, bfactor=0.0) -> None:
    """Pseudo-atoms as HETATM records (one atom per UNK residue, chains of 9999) + CONECT for bonds.

    pos_zyx: (N, 3) positions in A, [z, y, x] like the maps; elements: indices into element_names;
    bfactor: a scalar or one value per atom.
    """
    xyz = np.asarray(pos_zyx)[:, ::-1]
    bfac = np.broadcast_to(np.asarray(bfactor, dtype=float), (len(xyz),))
    lines = []
    for k, (p, e, bf) in enumerate(zip(xyz, list(elements), bfac)):
        name = element_names[int(e)]
        chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k // 9999]
        lines.append(f"HETATM{k + 1:5d}  {name:<3s} UNK {chain}{k % 9999 + 1:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00{bf:6.2f}          {name:>2s}")
    if edges is not None and len(edges):
        partners = [[] for _ in range(len(xyz))]
        for i, j in edges:
            partners[i].append(j)
            partners[j].append(i)
        for i, ps in enumerate(partners):
            for s in range(0, len(ps), 4):
                lines.append(f"CONECT{i + 1:5d}" + "".join(f"{j + 1:5d}" for j in ps[s:s + 4]))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def read_pdb(path: str | Path, element_names) -> tuple:
    """Inverse of write_pdb: ([z, y, x] positions (N, 3), element indices (N,), edges (E, 2))."""
    lookup = {e: i for i, e in enumerate(element_names)}
    pos, els, serial, edges = [], [], {}, set()
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial[int(line[6:11])] = len(pos)
            pos.append([float(line[46:54]), float(line[38:46]), float(line[30:38])])
            els.append(lookup[line[76:78].strip()])
        elif line.startswith("CONECT"):
            fields = [int(line[k:k + 5]) for k in range(6, len(line.rstrip()), 5)]
            for j in fields[1:]:
                a, b = serial[fields[0]], serial[j]
                edges.add((min(a, b), max(a, b)))
    return (np.array(pos), np.array(els, dtype=np.int64),
            np.array(sorted(edges), dtype=np.int64).reshape(-1, 2))
