"""Random envelope (mask): rough Gaussian random field + confinement, MBO smoothing.

1. GRF with power ~ k^-alpha, split at k ~ 1/R into a coarse part (global shape,
   unit variance) and a fine part (surface roughness, variance amplitude^2).
   alpha sets the Hurst exponent H = (alpha - 3)/2 of the fine roughness, but a
   self-affine surface only shows D = 3 - H below the crossover length where
   its slopes reach ~1; the amplitude moves that crossover, so both are needed
   to set the Richardson D between the resolution and R (see docs/METHODS.md).
2. A quadratic confinement term (one ellipsoid per domain) keeps a single compact
   object; it is smooth, so it changes global shape but not fine-scale roughness.
3. Threshold so the largest (periodic) connected component, with enclosed
   cavities filled, has volume V.
4. MBO mean-curvature flow at constant volume for compactness.
5. Centre periodically and check edge clearance.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from . import rng
from .fourier import Grid


class EnvelopeError(RuntimeError):
    pass


def split_random_field(grid: Grid, alpha: float, s_split: float,
                       gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Power-law GRF (power ~ s^-alpha) split into unit-variance coarse and fine parts at s_split (1/A)."""
    white = grid.rfft(rng.randn(grid.shape, gen, grid.device))
    amp = torch.where(grid.s > 0, grid.s.clamp(min=1e-12) ** (-alpha / 2), torch.zeros_like(grid.s))
    low = 1.0 / (1.0 + (grid.s / s_split) ** 4)
    coarse = grid.irfft(white * amp * low)
    fine = grid.irfft(white * amp * (1 - low))
    return coarse / coarse.std(), fine / fine.std()


def _random_unit_vector(gen: torch.Generator) -> np.ndarray:
    v = torch.randn(3, generator=gen, dtype=torch.float64).numpy()
    return v / np.linalg.norm(v)


def confinement(grid: Grid, volume: float, n_domains: int, elongation: float,
                gen: torch.Generator) -> torch.Tensor:
    """min_i (d_i / R_i)^2 over domains, d_i an (optionally anisotropic) distance in A."""
    r_dom = (3 * volume / n_domains / (4 * math.pi)) ** (1 / 3)
    centres = [np.zeros(3)]
    while len(centres) < n_domains:
        anchor = centres[int(torch.randint(len(centres), (1,), generator=gen))]
        centres.append(anchor + 1.6 * r_dom * _random_unit_vector(gen))
    centres = np.array(centres)
    centres -= centres.mean(axis=0)
    centre_box = grid.n * grid.apix / 2
    axis = torch.tensor(_random_unit_vector(gen), dtype=torch.float32, device=grid.device)

    coords = torch.arange(grid.n, device=grid.device, dtype=torch.float32) * grid.apix
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    pos = torch.stack([z, y, x], dim=-1)
    out = torch.full(grid.shape, float("inf"), device=grid.device)
    for c in centres:
        d = pos - torch.tensor(c + centre_box, dtype=torch.float32, device=grid.device)
        along = d @ axis
        perp2 = (d**2).sum(-1) - along**2
        # Volume-preserving stretch: e along the axis, e^-1/2 across it.
        q = (along / elongation) ** 2 + perp2 * elongation
        out = torch.minimum(out, q / r_dom**2)
    return out


def periodic_label(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Face-connected labels with periodic boundaries. Returns (labels, sizes) with sizes[0] = 0."""
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for axis in range(3):
        a = np.take(lab, 0, axis=axis).ravel()
        b = np.take(lab, -1, axis=axis).ravel()
        both = (a > 0) & (b > 0)
        for i, j in set(zip(a[both].tolist(), b[both].tolist())):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n + 1)])
    lab = roots[lab]
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    return lab, sizes


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    lab, sizes = periodic_label(mask)
    return lab == sizes.argmax()


def solid_object(mask: np.ndarray) -> np.ndarray:
    """Largest periodic component, centred, with enclosed cavities filled."""
    return ndimage.binary_fill_holes(periodic_centre(largest_component(mask)))


def periodic_centre(mask: np.ndarray) -> np.ndarray:
    """Roll so the circular mean of the object lies at the box centre."""
    n = mask.shape[0]
    shifts = []
    for axis in range(3):
        profile = mask.sum(axis=tuple(a for a in range(3) if a != axis)).astype(np.float64)
        theta = 2 * np.pi * np.arange(n) / n
        c = np.arctan2((profile * np.sin(theta)).sum(), (profile * np.cos(theta)).sum()) * n / (2 * np.pi)
        shifts.append(int(np.rint(n // 2 - c)))
    return np.roll(mask, shifts, axis=(0, 1, 2))


def clearance_ok(mask: np.ndarray, clearance_px: int) -> bool:
    c = clearance_px
    if c <= 0:
        return True
    inner = np.zeros_like(mask)
    inner[c:-c, c:-c, c:-c] = True
    return not (mask & ~inner).any()


def threshold_to_volume(field: np.ndarray, n_vox: int, tol: float = 0.005, max_iter: int = 30) -> np.ndarray:
    """Largest component of {field > t} with volume ~ n_vox (bisection on t; monotone in t)."""
    flat = np.sort(field.ravel())[::-1]
    lo, hi = n_vox, min(flat.size - 1, 4 * n_vox)   # ranks: top-`rank` voxels kept
    best = None
    for _ in range(max_iter):
        rank = (lo + hi) // 2
        comp = solid_object(field > flat[rank])
        v = int(comp.sum())
        if best is None or abs(v - n_vox) < abs(best[1] - n_vox):
            best = (comp, v)
        if abs(v - n_vox) <= tol * n_vox or hi - lo <= 1:
            break
        if v < n_vox:
            lo = rank
        else:
            hi = rank
    return best[0]


def mbo_flow(grid: Grid, mask: np.ndarray, time: float, n_vox: int) -> np.ndarray:
    """Merriman-Bence-Osher mean-curvature flow at constant volume for `time` A^2."""
    sigma = 2.0 * grid.apix
    step = sigma**2 / 2
    n_steps = round(time / step)
    for _ in range(n_steps):
        phi = grid.gaussian_filter(torch.from_numpy(mask.astype(np.float32)).to(grid.device), sigma)
        phi = phi.cpu().numpy()
        t = np.partition(phi.ravel(), phi.size - n_vox)[phi.size - n_vox]
        mask = solid_object(phi >= t)
    return mask


def make_envelope(grid: Grid, *, volume: float, alpha: float, roughness_amplitude: float,
                  compactness: float, n_domains: int, elongation: float, bias: float,
                  clearance: float, seed: int, max_attempts: int = 10) -> tuple[np.ndarray, int]:
    """Returns (bool mask, attempt index used)."""
    n_vox = round(volume / grid.apix**3)
    clearance_px = math.ceil(clearance / grid.apix)
    radius = (3 * volume / (4 * math.pi)) ** (1 / 3)
    for attempt in range(max_attempts):
        gen = rng.stage_generator(seed, "envelope", attempt)
        coarse, fine = split_random_field(grid, alpha, 1.0 / radius, gen)
        field = coarse + roughness_amplitude * fine
        field = field - bias * confinement(grid, volume, n_domains, elongation, gen)
        mask = threshold_to_volume(field.cpu().numpy(), n_vox)
        mask = mbo_flow(grid, mask, compactness, n_vox)
        mask = periodic_centre(mask)
        if clearance_ok(mask, clearance_px):
            return mask, attempt
    raise EnvelopeError(f"no envelope with {clearance} A edge clearance after {max_attempts} attempts")
