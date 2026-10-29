"""Pseudo-atom sampling and rendering with electron scattering factors.

Electron form factors are the 5-Gaussian elastic fits of International Tables
Vol. C Table 4.3.2.2 (Peng et al. 1996), taken from gemmi (tier 0):

    f(s) = sum_i a_i exp(-b_i s^2 / 4)                          (s = |s| = 1/d)
    rho(r) = sum_i a_i (4 pi / (b_i + B))^1.5 exp(-4 pi^2 r^2 / (b_i + B))

v0 uses an inhomogeneous Poisson gas (no hard core) whose intensity follows the
mesoscale field; milestone 2 replaces it with a hard-core process + RMC.
"""

from __future__ import annotations

import math
from functools import cache

import torch

from . import rng
from .constants import HEAVY_ATOM_COMPOSITION
from .fourier import Grid

ELEMENTS = tuple(HEAVY_ATOM_COMPOSITION)


@cache
def form_factor_coefs(element: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import gemmi

    coefs = gemmi.Element(element).c4322.get_coefs()   # a1..a5, b1..b5
    return tuple(coefs[:5]), tuple(coefs[5:])


def form_factor(element: str, s: torch.Tensor, bfactor: float = 0.0) -> torch.Tensor:
    a, b = form_factor_coefs(element)
    out = torch.zeros_like(s)
    for ai, bi in zip(a, b):
        out = out + ai * torch.exp(-(bi + bfactor) * s**2 / 4)
    return out


def sample_gas(grid: Grid, intensity: torch.Tensor, n_atoms: int, seed: int,
               stage: str = "atoms") -> tuple[torch.Tensor, torch.Tensor]:
    """Inhomogeneous Poisson-like gas: n_atoms positions (A) drawn with density ~ intensity.

    Returns (positions (N, 3) float32 in A, element indices (N,) into ELEMENTS).
    """
    gen = rng.stage_generator(seed, stage)
    p = intensity.reshape(-1).double().cpu()
    vox = torch.multinomial(p / p.sum(), n_atoms, replacement=True, generator=gen)
    n = grid.n
    ijk = torch.stack([vox // (n * n), (vox // n) % n, vox % n], dim=1).float()
    jitter = torch.rand((n_atoms, 3), generator=gen) - 0.5
    pos = (ijk + jitter) * grid.apix
    comp = torch.tensor([HEAVY_ATOM_COMPOSITION[e] for e in ELEMENTS], dtype=torch.float64)
    elem = torch.multinomial(comp / comp.sum(), n_atoms, replacement=True, generator=gen)
    return pos.to(grid.device), elem.to(grid.device)


def render(grid: Grid, positions: torch.Tensor, elements: torch.Tensor, bfactor: float,
           cutoff: float | None = None) -> torch.Tensor:
    """Real-space Gaussian splatting with periodic wrap; deterministic (sort + cumsum)."""
    if cutoff is None:
        bmax = max(max(form_factor_coefs(e)[1]) for e in ELEMENTS) + bfactor
        # exp(-4 pi^2 r^2 / bmax) < 1e-4 of peak for the widest Gaussian
        cutoff = math.sqrt(bmax * math.log(1e4)) / (2 * math.pi)
    r = math.ceil(cutoff / grid.apix)
    off = torch.arange(-r, r + 1, device=grid.device)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    offsets = torch.stack([oz, oy, ox], dim=-1).reshape(-1, 3)
    offsets = offsets[(offsets.float() ** 2).sum(-1) * grid.apix**2 <= (cutoff + grid.apix) ** 2]

    n = grid.n
    flat_idx, values = [], []
    for ei, e in enumerate(ELEMENTS):
        sel = elements == ei
        if not sel.any():
            continue
        p = positions[sel] / grid.apix
        base = torch.round(p).long()
        vox = base[:, None, :] + offsets[None, :, :]                       # (Na, M, 3)
        d2 = ((vox.float() - p[:, None, :]) ** 2).sum(-1) * grid.apix**2   # (Na, M)
        a, b = form_factor_coefs(e)
        val = torch.zeros_like(d2)
        for ai, bi in zip(a, b):
            bb = bi + bfactor
            val = val + ai * (4 * math.pi / bb) ** 1.5 * torch.exp(-4 * math.pi**2 * d2 / bb)
        vox = vox % n
        flat_idx.append((vox[..., 0] * n * n + vox[..., 1] * n + vox[..., 2]).reshape(-1))
        values.append(val.reshape(-1))
    flat_idx = torch.cat(flat_idx)
    values = torch.cat(values)
    order = torch.argsort(flat_idx, stable=True)
    flat_idx, values = flat_idx[order], values[order].double()
    uniq, counts = torch.unique_consecutive(flat_idx, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    cs = torch.cumsum(values, 0)[ends]
    sums = torch.cat([cs[:1], cs[1:] - cs[:-1]])
    out = torch.zeros(n**3, dtype=torch.float64, device=grid.device)
    out[uniq] = sums
    return out.reshape(grid.shape).float()


def mean_form_factor_sq(grid_s: torch.Tensor, bfactor: float) -> torch.Tensor:
    """Composition-weighted <f(s)^2> including the B-factor."""
    out = torch.zeros_like(grid_s)
    total = sum(HEAVY_ATOM_COMPOSITION.values())
    for e in ELEMENTS:
        out = out + HEAVY_ATOM_COMPOSITION[e] / total * form_factor(e, grid_s, bfactor) ** 2
    return out
