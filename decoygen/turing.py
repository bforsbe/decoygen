"""Mesoscale Turing field: x <- tanh(x + dt * (F^-1[g(k) F[x]] + h)) inside the mask.

g(k) is a difference of Gaussians designed to peak at k* = 1/wavelength. The
scalar h is a feedback term that steers the fraction of x > 0 inside the mask to
`fill_fraction` during pattern formation, so the morphology (spots, tubes,
bicontinuous, lamellae) forms at that fraction rather than being thresholded
afterwards.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import rng
from .fourier import Grid


def dog_sigmas(wavelength: float, quotient: float) -> tuple[float, float]:
    """sigma_short, sigma_long (A) so the DoG growth function peaks at s = 1/wavelength.

    For G(s) = exp(-2 pi^2 ss^2 s^2) - exp(-2 pi^2 sl^2 s^2), sl = q ss, the peak is at
    s*^2 = ln(q^2) / (2 pi^2 ss^2 (q^2 - 1)).
    """
    q2 = quotient**2
    ss = math.sqrt(math.log(q2) / (2 * math.pi**2 * (q2 - 1))) * wavelength
    return ss, quotient * ss


def growth_function(grid: Grid, wavelength: float, quotient: float) -> torch.Tensor:
    ss, sl = dog_sigmas(wavelength, quotient)
    g = torch.exp(-2 * math.pi**2 * ss**2 * grid.s2) - torch.exp(-2 * math.pi**2 * sl**2 * grid.s2)
    s_star = 1.0 / wavelength
    g_peak = math.exp(-2 * math.pi**2 * ss**2 * s_star**2) - math.exp(-2 * math.pi**2 * sl**2 * s_star**2)
    return g / g_peak


def turing_field(grid: Grid, mask: np.ndarray, *, wavelength: float, quotient: float,
                 fill_fraction: float, iterations: int, dt: float, seed: int,
                 feedback: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Returns (field in [0, 1], zero outside mask; info dict)."""
    m = torch.as_tensor(mask, device=grid.device)
    gen = rng.stage_generator(seed, "turing")
    x = 0.1 * rng.randn(grid.shape, gen, grid.device) * m
    g = growth_function(grid, wavelength, quotient)
    h = 0.0
    n_in = float(m.sum())
    for _ in range(iterations):
        y = grid.irfft(grid.rfft(x) * g)
        x = torch.where(m, torch.tanh(x + dt * (y + h)), torch.zeros_like(x))
        frac = float(((x > 0) & m).sum()) / n_in
        h += feedback * (fill_fraction - frac)
    out = ((x + 1) / 2).clamp(0, 1) * m
    info = {"turing_fill_realized": float(((x > 0) & m).sum()) / n_in,
            "turing_bias_h": h,
            "turing_sigmas_A": list(dog_sigmas(wavelength, quotient))}
    return out, info
