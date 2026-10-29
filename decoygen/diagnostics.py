"""Tier-0 physics diagnostics for a generated decoy."""

from __future__ import annotations

import numpy as np
import torch

from . import roughness
from .constants import FSC_THRESHOLD, SASA_EXPONENT, SASA_PREFACTOR
from .fourier import Grid


def fsc_resolution(grid: Grid, fsc: torch.Tensor, threshold: float = FSC_THRESHOLD) -> float:
    """Resolution (A) at the first crossing below threshold, linearly interpolated."""
    f = fsc.cpu().numpy()
    s = grid.shell_s.cpu().numpy()
    for i in range(1, grid.nyquist_shell + 1):
        if f[i] < threshold:
            t = (f[i - 1] - threshold) / (f[i - 1] - f[i])
            return float(1.0 / (s[i - 1] + t * (s[i] - s[i - 1])))
    return float(2 * grid.apix)


def structure_tensor_stats(grid: Grid, x: torch.Tensor, mask: torch.Tensor, scale: float) -> dict:
    """Westin linearity/planarity/sphericity of the structure tensor inside the mask.

    High linearity marks rod-like (helix/backbone-like) local structure,
    high planarity marks sheet-like structure.
    """
    g = grid.gradient(x)
    comps = {}
    for i in range(3):
        for j in range(i, 3):
            comps[i, j] = grid.gaussian_filter(g[i] * g[j], scale)[mask]
    J = torch.empty((comps[0, 0].numel(), 3, 3), device=x.device)
    for (i, j), v in comps.items():
        J[:, i, j] = v
        J[:, j, i] = v
    ev = torch.linalg.eigvalsh(J.double()).flip(-1).clamp(min=0)   # descending
    l1 = ev[:, 0].clamp(min=1e-30)
    cl = (ev[:, 0] - ev[:, 1]) / l1
    cp = (ev[:, 1] - ev[:, 2]) / l1
    cs = ev[:, 2] / l1
    return {"linearity_mean": float(cl.mean()), "planarity_mean": float(cp.mean()),
            "sphericity_mean": float(cs.mean()),
            "linearity_p90": float(torch.quantile(cl.float()[:: max(1, cl.numel() // 100000)], 0.9))}


def chirality(grid: Grid, x: torch.Tensor, mask: torch.Tensor) -> float:
    """Normalized pseudoscalar <grad x . (grad Lx  x  grad L^2 x)>; flips sign under mirroring."""
    lx = grid.laplacian(x)
    l2x = grid.laplacian(lx)
    a = torch.stack(grid.gradient(x), -1)[mask]
    b = torch.stack(grid.gradient(lx), -1)[mask]
    c = torch.stack(grid.gradient(l2x), -1)[mask]
    triple = (a * torch.cross(b, c, dim=-1)).sum(-1).double()
    norm = (a.norm(dim=-1) * b.norm(dim=-1) * c.norm(dim=-1)).double()
    return float(triple.sum() / norm.sum().clamp(min=1e-300))


def moments(v: torch.Tensor) -> dict:
    v = v.double()
    m, sd = v.mean(), v.std()
    return {"mean": float(m), "std": float(sd), "skew": float(((v - m) ** 3).mean() / sd**3)}


def compute(grid: Grid, params, *, mask: np.ndarray, support: torch.Tensor, clean: torch.Tensor,
            signal: torch.Tensor, reference: torch.Tensor, half1: torch.Tensor, half2: torch.Tensor,
            target_power: torch.Tensor) -> dict:
    d = params.resolution
    vol = float(mask.sum()) * grid.apix**3
    radius = (3 * vol / (4 * np.pi)) ** (1 / 3)
    smin, smax = roughness.richardson_range(d, radius, grid.apix)
    rich = roughness.richardson(grid, mask, smin, smax)
    area = roughness.surface_area(grid, mask, d / 2)
    mass_da = params.mass_kda * 1e3
    m = torch.as_tensor(mask, device=grid.device)

    p_sig = grid.power_spectrum(signal)
    shells = slice(max(1, int(grid.n * grid.apix / radius)), grid.nyquist_shell + 1)
    log_ratio = torch.log10(p_sig[shells] / target_power[shells])
    fsc = grid.fsc(half1, half2)
    out = {
        "volume_A3": vol,
        "volume_target_A3": params.volume,
        "volume_ratio": vol / params.volume,
        "equivalent_radius_A": radius,
        "richardson_D": rich["D"],
        "richardson_sigma_range_A": [smin, smax],
        "surface_area_at_d_A2": area,
        "sphericity": roughness.sphericity(vol, area),
        "sasa_literature_A2": SASA_PREFACTOR * mass_da**SASA_EXPONENT,
        "surface_area_over_sasa": area / (SASA_PREFACTOR * mass_da**SASA_EXPONENT),
        "spectrum_log10_rms_error": float(torch.sqrt(torch.mean(log_ratio**2))),
        "spectrum_log10_max_error": float(log_ratio.abs().max()),
        "halfmap_fsc_resolution_A": fsc_resolution(grid, fsc),
        "histogram_signal": moments(signal[support]),
        "histogram_reference": moments(reference[support]),
        "chirality": chirality(grid, clean, m),
        "chirality_reference": chirality(grid, grid.cosine_lowpass(reference, 1 / d, 0.1 / d), m),
    }
    out.update({f"structure_tensor_{k}": v for k, v in structure_tensor_stats(grid, clean, m, d).items()})
    curves = {
        "shell_s": grid.shell_s.cpu().numpy(),
        "power_signal": p_sig.cpu().numpy(),
        "power_target": target_power.cpu().numpy(),
        "power_reference": grid.power_spectrum(reference).cpu().numpy(),
        "fsc": fsc.cpu().numpy(),
        "richardson": rich,
    }
    return out, curves
