"""Half-map noise.

'white': flat (white) Gaussian noise of power N0 per Fourier coefficient, with N0
chosen so the half-map SNR at s = 1/resolution equals SNR_AT_THRESHOLD * snr_scale,
i.e. the half-map FSC crosses 0.143 at the nominal resolution when snr_scale = 1.
The FSC falloff is then set by the signal's own decay (B-factor, form factors), as
for real reconstructions. The same noise covers solvent, so there is no mask edge.
"""

from __future__ import annotations

import math

import torch

from . import rng
from .constants import SNR_AT_THRESHOLD
from .fourier import Grid


def white_noise_level(grid: Grid, signal: torch.Tensor, resolution: float, snr_scale: float) -> float:
    p = grid.power_spectrum(signal)
    shell = round(grid.n * grid.apix / resolution)
    return float(p[shell]) / (SNR_AT_THRESHOLD * snr_scale)


def half_maps(grid: Grid, signal: torch.Tensor, resolution: float, snr_scale: float,
              noise_model: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if noise_model == "none":
        return signal.clone(), signal.clone(), {"noise_power": 0.0}
    n0 = white_noise_level(grid, signal, resolution, snr_scale)
    # unit-variance white noise has <|DFT|^2> = n^3 per coefficient
    sd = math.sqrt(n0 / grid.n**3)
    halves = []
    for stage in ("noise_half1", "noise_half2"):
        gen = rng.stage_generator(seed, stage)
        halves.append(signal + sd * rng.randn(grid.shape, gen, grid.device))
    return halves[0], halves[1], {"noise_power": n0, "noise_sd_real": sd}
