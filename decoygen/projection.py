"""Gerchberg-Saxton-style alternating projection.

Fourier step: keep phases, rescale each shell so its mean power equals the target
(relative amplitudes within a shell are kept, so speckle statistics stay natural).
Real-space step: zero outside the support and rank-match the 1-point histogram
inside it to the reference (which also enforces positivity).
"""

from __future__ import annotations

import torch

from .fourier import Grid


def histogram_match(x: torch.Tensor, support: torch.Tensor, sorted_target: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    vals = x[support]
    ranks = torch.argsort(vals, stable=True)
    matched = torch.empty_like(vals)
    matched[ranks] = sorted_target
    out[support] = matched
    return out


def spectral_error(power: torch.Tensor, target: torch.Tensor, shells: slice) -> float:
    """RMS of log10(P / P_target) over the given shells."""
    p, t = power[shells], target[shells]
    ok = (p > 0) & (t > 0)
    return float(torch.sqrt(torch.mean(torch.log10(p[ok] / t[ok]) ** 2)))


def gerchberg_saxton(grid: Grid, x0: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
                     sorted_hist: torch.Tensor, iterations: int, error_shells: slice):
    x = x0
    history = []
    for _ in range(iterations):
        X = grid.rfft(x)
        p = grid.shell_mean(X.real**2 + X.imag**2)
        scale = torch.where(p > 0, torch.sqrt(target / p.clamp(min=1e-300)), torch.zeros_like(p))
        x = grid.irfft(X * grid.expand(scale).float())
        x = histogram_match(x, support, sorted_hist)
        history.append(spectral_error(grid.power_spectrum(x), target, error_shells))
    return x, history
