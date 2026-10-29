"""MRC and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np


class MRCHeaderError(RuntimeError):
    pass


def write_mrc(path: str | Path, data: np.ndarray, apix: float) -> None:
    """float32 (mode 2), origin 0, axis order 1/2/3; array indexed [z, y, x]."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = apix
        m.header.origin = (0.0, 0.0, 0.0)
        m.header.nxstart = m.header.nystart = m.header.nzstart = 0
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
