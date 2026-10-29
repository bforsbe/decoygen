"""Target radial power spectra.

The reference is a homogeneous random-atom gas rendered in the envelope (tier 0).
Under the random-atom model its power is

    P(s) = <f(s)>^2 n^2 |M^(s)|^2 + N <f(s)^2> S(s),       S = 1 for the gas,

so the envelope (shape) term comes from the reference itself and a structure-factor
multiplier only acts on the atomic (Wilson) term:

    P_target(s) = P_ref(s) + (S_mult(s) - 1) * W(s),   W = N <f^2> / apix^6 (DFT units).
"""

from __future__ import annotations

import numpy as np
import torch

from .atoms import mean_form_factor_sq
from .fourier import Grid


def wilson_power(grid: Grid, n_atoms: int, bfactor: float) -> torch.Tensor:
    return n_atoms * mean_form_factor_sq(grid.shell_s, bfactor) / grid.apix**6


def peaks_multiplier(s: torch.Tensor, peak_d, peak_heights, rel_width: float) -> torch.Tensor:
    """1 + sum of isotropic Gaussian bumps at s = 1/d."""
    out = torch.ones_like(s)
    for d, h in zip(peak_d, peak_heights):
        s0 = 1.0 / d
        out = out + h * torch.exp(-0.5 * ((s - s0) / (rel_width * s0)) ** 2)
    return out


def file_multiplier(s: torch.Tensor, path: str) -> torch.Tensor:
    """Tier 3: S(s) multiplier from a two-column file (s in 1/A, multiplier); 1 outside its range."""
    data = np.loadtxt(path)
    vals = np.interp(s.cpu().numpy(), data[:, 0], data[:, 1], left=1.0, right=1.0)
    return torch.as_tensor(vals, dtype=s.dtype, device=s.device)


def target_power(grid: Grid, reference_power: torch.Tensor, n_atoms: int, params) -> torch.Tensor:
    s = grid.shell_s
    if params.target_spectrum == "physics":
        return reference_power.clone()
    if params.target_spectrum == "physics+peaks":
        mult = peaks_multiplier(s, params.peak_d, params.peak_heights, params.peak_rel_width)
    elif params.target_spectrum == "best":
        mult = file_multiplier(s, params.spectrum_file)
    else:
        raise ValueError(params.target_spectrum)
    return (reference_power + (mult - 1) * wilson_power(grid, n_atoms, params.bfactor)).clamp(min=0)
