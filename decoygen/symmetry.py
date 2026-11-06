"""Symmetric envelopes: competing copies of one subunit field (docs/SYMMETRY.md).

A symmetry group is a list of 3x3 rotation matrices in [z, y, x] coordinates. Only Cn is parsed
(`point_group`), but nothing below assumes it: neighbour distances and the swap partner come from
the rotation list, so Dn, T, O and I only need their matrices.

1. subunit field   f1 = coarse + a * fine - b * q in the subunit's frame; q is a bowl of the subunit's
                   radius R centred at distance r from the axis (plus a notch and an arm for a domain swap)
2. copies          f_k(x) = f1(g_k^-1 x), sampled by trilinear interpolation
3. competition     F = max_k f_k is the assembly field; owner = argmax_k of the lightly smoothed f_k
4. threshold       F > t with volume n * V_subunit (largest component, cavities filled, not re-centred)
5. symmetrise      MBO smoothing ignores symmetry, so it is followed by group averaging
                   m <- mean_k m(g_k^-1 x) and a re-threshold; owner islands go to the surrounding subunit
6. placement       the axis direction and centre are inputs, so the axis need not run through the box centre

Contact depth: neighbouring bowls overlap by `contact` x R, so r = (2 - contact) R / min_{g != 1} |g u - u|
for the subunit direction u (Cn: |g u - u| = 2 sin(pi / n)).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import rng
from .envelope import EnvelopeError, clearance_ok, mbo_flow, solid_in_place, split_random_field, threshold_to_volume
from .fourier import Grid

SLOT = 0.45                                 # radius of the swap slot (notch and arm lobe), in R
PLANNED = {"D": "dihedral", "T": "tetrahedral", "O": "octahedral", "I": "icosahedral"}


# ---------------------------------------------------------------- groups

def cyclic(n: int) -> np.ndarray:
    """Cn about the z axis, as (n, 3, 3) matrices acting on [z, y, x] column vectors."""
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        out.append([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array(out, dtype=np.float64)


def point_group(spec: str) -> np.ndarray:
    """Rotations of a point group given as 'C3' etc. (canonical frame: principal axis along z)."""
    m = re.fullmatch(r"([CDTOI])(\d*)", spec.strip().upper())
    if not m:
        raise ValueError(f"unknown symmetry {spec!r}; expected e.g. C1, C3, C7")
    kind, num = m.groups()
    if kind == "C":
        n = int(num or 0)
        if n < 1:
            raise ValueError(f"{spec}: Cn needs n >= 1")
        return cyclic(n)
    raise NotImplementedError(f"{spec}: {PLANNED[kind]} symmetry is planned; the envelope code takes any list "
                              f"of rotations, only this parser is missing it")


def rotation_from_z(axis: np.ndarray) -> np.ndarray:
    """Rotation taking the z axis [1, 0, 0] (in [z, y, x]) to `axis`."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    z = np.array([1.0, 0, 0])
    v, c = np.cross(z, a), float(z @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])     # 180 deg about x
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def neighbour_distance(ops: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    """min_{g != identity} |g u - u| and the index of that g (unit direction u)."""
    if len(ops) == 1:
        return math.inf, 0
    d = np.linalg.norm(ops[1:] @ u - u, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k + 1


# ---------------------------------------------------------------- envelope

@dataclass
class SymmetricEnvelope:
    mask: np.ndarray                    # bool, [z, y, x]
    labels: np.ndarray                  # uint8, 0 solvent, 1..n subunits (1 = the identity copy)
    ops: np.ndarray                     # (n, 3, 3) in the box frame: x' = centre + R (x - centre), [z, y, x] in A
    centre: np.ndarray                  # (3,) A, a point on every symmetry element
    info: dict = field(default_factory=dict)


def _sample(field_: torch.Tensor, pos: torch.Tensor, apix: float) -> torch.Tensor:
    """Trilinear sample of a periodic [z, y, x] field at positions (..., 3) in A ([z, y, x])."""
    n = field_.shape[0]
    idx = torch.remainder(pos / apix, n)                               # periodic
    norm = 2 * idx / (n - 1) - 1
    g = norm.flip(-1).reshape(1, *pos.shape[:-1], 3)                   # grid_sample wants (x, y, z)
    return F.grid_sample(field_[None, None], g, mode="bilinear", padding_mode="border",
                         align_corners=True)[0, 0]


def _subunit_q(p: torch.Tensor, c: np.ndarray, radius: float, swap: float, slot: np.ndarray | None,
               target: np.ndarray | None) -> torch.Tensor:
    """Confinement of the identity subunit at positions p (A). Swap: notch at its own slot, arm to `target`
    (the swap partner's slot)."""
    def t(v):
        return torch.as_tensor(v, dtype=p.dtype, device=p.device)

    q = ((p - t(c)) ** 2).sum(-1) / radius**2
    if swap > 0:
        # the notch and the lobe at the arm's tip have the same size, so the volume one subunit gives up
        # is the volume its neighbour's arm takes; a thinner shaft joins the lobe to the body
        q = q + 3.0 * swap * torch.exp(-((p - t(slot)) ** 2).sum(-1) / (SLOT * radius) ** 2)
        d = t(target - c)
        s = torch.clamp(((p - t(c)) @ d) / (d @ d), 0, 1)
        shaft = ((p - (t(c) + s[..., None] * d)) ** 2).sum(-1) / (0.3 * radius) ** 2
        lobe = ((p - t(target)) ** 2).sum(-1) / (SLOT * radius) ** 2
        q = torch.minimum(q, torch.minimum(shaft, lobe) + 0.2 * (1 - swap))
    return q


def _symmetrise(mask: np.ndarray, pull: list[torch.Tensor], apix: float, n_vox: int, device) -> np.ndarray:
    """Group average of a binary mask, re-thresholded to n_vox voxels."""
    m = torch.as_tensor(mask, dtype=torch.float32, device=device)
    avg = sum(_sample(m, p, apix) for p in pull) / len(pull)
    return threshold_to_volume(avg.cpu().numpy(), n_vox, solid=solid_in_place)


def _tidy_owners(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep each subunit's largest piece; every other voxel of the mask goes to the nearest kept subunit.
    Applied identically to every copy, so the symmetry is kept (up to interpolation)."""
    out = np.where(mask, labels, 0).astype(np.uint8)
    for k in np.unique(out[out > 0]):
        lab, m = ndimage.label(out == k)
        if m > 1:
            big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            out[(lab > 0) & (lab != big)] = 0
    hole = mask & (out == 0)
    if hole.any():
        _, idx = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out[hole] = out[tuple(i[hole] for i in idx)]
    return out


def interfaces(labels: np.ndarray, n: int, apix: float) -> np.ndarray:
    """(n, n) contact area in A^2 between subunits (shared voxel faces)."""
    area = np.zeros((n + 1, n + 1))
    for axis in range(3):
        a = np.moveaxis(labels, axis, 0)
        lo, hi = a[:-1].ravel(), a[1:].ravel()
        diff = (lo != hi) & (lo > 0) & (hi > 0)
        np.add.at(area, (lo[diff], hi[diff]), 1)
    area = area + area.T
    return area[1:, 1:] * apix**2


def make_symmetric_envelope(grid: Grid, *, symmetry: str, volume: float, alpha: float,
                            roughness_amplitude: float, compactness: float, bias: float, clearance: float,
                            seed: int, contact: float = 0.6, swap: float = 0.0,
                            axis: np.ndarray | None = None, shift: np.ndarray | None = None,
                            owner_sigma: float = 1.5, owner_fine: float = 0.3,
                            max_attempts: int = 10) -> SymmetricEnvelope:
    """Envelope with point-group symmetry, divided into subunits. `volume` is the whole assembly's;
    `axis` ([z, y, x], default z) and `shift` (A from the box centre) place the symmetry axis.
    Ownership is decided on coarse + owner_fine * a * fine - b * q, smoothed by owner_sigma (A): with
    owner_fine < 1 interfaces are smoother than the free surface, which keeps its full roughness."""
    canon = point_group(symmetry)
    n = len(canon)
    P = rotation_from_z(axis if axis is not None else np.array([1.0, 0, 0]))
    ops = np.einsum("ij,kjl,ml->kim", P, canon, P)                    # P g P^T
    centre = np.full(3, grid.n * grid.apix / 2) + (np.zeros(3) if shift is None else np.asarray(shift, float))

    v_sub = volume / n
    radius = (3 * v_sub / (4 * math.pi)) ** (1 / 3)
    u = P @ np.array([0.0, 1.0, 0.0])                                  # subunit direction, perpendicular to z
    d_nb, k_nb = neighbour_distance(ops, u)
    r_axis = 0.0 if math.isinf(d_nb) else (2 - contact) * radius / d_nb
    c = centre + r_axis * u
    slot = target = None
    if swap > 0:
        if n == 1:
            raise ValueError("a domain swap needs a partner: symmetry must have n > 1")
        prev = centre + ops[k_nb].T @ (c - centre)                     # the copy whose partner is us
        slot = c + 0.55 * radius * (prev - c) / np.linalg.norm(prev - c)
        target = centre + ops[k_nb] @ (slot - centre)

    n_vox = round(volume / grid.apix**3)
    coords = torch.arange(grid.n, device=grid.device, dtype=torch.float32) * grid.apix
    pos = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), -1)
    cen = torch.as_tensor(centre, dtype=torch.float32, device=grid.device)
    # g_k^-1 x for every voxel x: where copy k reads the subunit field
    pull = [cen + (pos - cen) @ torch.as_tensor(g, dtype=torch.float32, device=grid.device) for g in ops]

    for attempt in range(max_attempts):
        gen = rng.stage_generator(seed, "envelope", attempt)
        coarse, fine = split_random_field(grid, alpha, 1.0 / radius, gen)
        base = coarse + roughness_amplitude * fine
        own = coarse + owner_fine * roughness_amplitude * fine
        f, o = [], []
        for p in pull:
            q = bias * _subunit_q(p, c, radius, swap, slot, target)
            f.append(_sample(base, p, grid.apix) - q)
            o.append(grid.gaussian_filter(_sample(own, p, grid.apix) - q, owner_sigma))
        assembly = torch.stack(f).max(0).values
        owner = torch.stack(o).argmax(0).cpu().numpy() + 1
        del f, o

        mask = threshold_to_volume(assembly.cpu().numpy(), n_vox, solid=solid_in_place)
        mask = mbo_flow(grid, mask, compactness, n_vox, solid=solid_in_place)
        mask = _symmetrise(mask, pull, grid.apix, n_vox, grid.device)
        labels = _tidy_owners(owner, mask)
        present = np.unique(labels[labels > 0])
        if len(present) < n:
            raise EnvelopeError(f"{symmetry}, contact {contact}: only {len(present)} of {n} subunits in the "
                                f"assembly; the copies do not touch (raise contact)")
        if not clearance_ok(mask, math.ceil(clearance / grid.apix)):
            continue

        # checks
        sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:] * grid.apix**3
        pieces = [int(ndimage.label(labels == k + 1)[1]) for k in range(n)]
        area = interfaces(labels, n, grid.apix)
        m = torch.as_tensor(mask, dtype=torch.float32, device=grid.device)
        sym_err = [float(((_sample(m, p, grid.apix) > 0.5).cpu().numpy() != mask).sum() / mask.sum())
                   for p in pull[1:]]
        info = {
            "symmetry": symmetry.upper(), "n_subunits": n, "contact": contact, "swap": swap,
            "owner_fine": owner_fine, "owner_sigma_A": owner_sigma,
            "axis_zyx": (P @ np.array([1.0, 0, 0])).tolist(), "centre_zyx_A": centre.tolist(),
            "subunit_radius_A": radius, "axis_distance_A": r_axis, "attempt": attempt,
            "volume_A3": float(mask.sum() * grid.apix**3), "target_volume_A3": volume,
            "subunit_volumes_A3": sizes.tolist(), "subunit_pieces": pieces,
            "interface_A2_per_subunit": float(area.sum() / n),
            "interface_A2": area.round(1).tolist(),
            "symmetry_mismatch_fraction_max": max(sym_err, default=0.0),
            "operators_zyx": [g.round(9).tolist() for g in ops],
        }
        return SymmetricEnvelope(mask=mask, labels=labels, ops=ops, centre=centre, info=info)
    raise EnvelopeError(f"no {symmetry} envelope with {clearance} A edge clearance after {max_attempts} attempts")


# ---------------------------------------------------------------- output

# Okabe-Ito, fixed order around the ring (neighbours never alike)
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]


def save(env: SymmetricEnvelope, out, name: str, apix: float, params: dict | None = None):
    """NAME_envelope.mrc (0/1), NAME_subunits.mrc (labels), NAME_subunit_K.mrc (0/1 each), NAME.json,
    NAME.cxc (ChimeraX: subunits as coloured surfaces plus the symmetry axis)."""
    import json
    from pathlib import Path

    from . import __version__, io

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    io.write_mrc(out / f"{name}_envelope.mrc", env.mask.astype(np.float32), apix)
    io.write_mrc(out / f"{name}_subunits.mrc", env.labels.astype(np.float32), apix)
    n = env.info["n_subunits"]
    for k in range(n):
        io.write_mrc(out / f"{name}_subunit_{k + 1}.mrc", (env.labels == k + 1).astype(np.float32), apix)
    (out / f"{name}.json").write_text(json.dumps({"decoygen_version": __version__, "params": params or {},
                                                  "symmetry": env.info}, indent=1))
    axis = np.array(env.info["axis_zyx"])[::-1]                      # ChimeraX: x, y, z in A
    c = env.centre[::-1]
    half = 0.45 * env.mask.shape[0] * apix
    a, b = c - half * axis, c + half * axis
    lines = ["close session"]
    for k in range(n):
        lines += [f"open {name}_subunit_{k + 1}.mrc",
                  f"volume #{k + 1} style surface level 0.5 color {COLOURS[k % len(COLOURS)]}"]
    lines += [f"marker #{n + 1} position {a[0]:.2f},{a[1]:.2f},{a[2]:.2f} color red radius 0.8",
              f"marker #{n + 1} position {b[0]:.2f},{b[1]:.2f},{b[2]:.2f} color red radius 0.8",
              f"marker link #{n + 1}.1,2 color red radius 0.5",
              "view", "lighting soft"]
    (out / f"{name}.cxc").write_text("\n".join(lines) + "\n")
    return out
