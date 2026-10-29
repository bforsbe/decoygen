"""Surface roughness: Richardson-plot fractal dimension and the alpha -> D calibration.

Area at scale sigma is the total variation of the Gaussian-blurred indicator,
A(sigma) = int |grad(G_sigma * chi)| dV, which by the coarea formula averages the
areas of the level sets of the smoothed object. A(sigma) ~ sigma^(2 - D).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch

from .fourier import Grid

CALIBRATION_FILE = Path(__file__).parent / "data" / "alpha_to_D.json"


def surface_area(grid: Grid, mask: np.ndarray | torch.Tensor, sigma: float) -> float:
    m = torch.as_tensor(mask, dtype=torch.float32, device=grid.device)
    phi = grid.gaussian_filter(m, sigma)
    gz, gy, gx = grid.gradient(phi)
    return float(torch.sqrt(gz**2 + gy**2 + gx**2).sum().double()) * grid.apix**3


def equivalent_ball(grid: Grid, n_vox: int) -> np.ndarray:
    radius = (3 * n_vox / (4 * math.pi)) ** (1 / 3)
    c = grid.n // 2
    z, y, x = np.indices(grid.shape)
    return (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2


def richardson(grid: Grid, mask, sigma_min: float, sigma_max: float, n: int = 8) -> dict:
    """D = 2 - slope of log(A_mask / A_ball) vs log(sigma).

    Blurring a curved surface shrinks its level sets by ~sigma^2/R, so even a smooth
    ball loses area with sigma (raw D ~2.07-2.17 for R = 20-12 A). Dividing by an
    equal-volume ball measured the same way removes this to first order; the residual
    for smooth ellipsoids is ~+0.015 at 1.6x and ~+0.05 at 2.5x elongation.
    """
    mask = np.asarray(mask.cpu() if torch.is_tensor(mask) else mask)
    sigmas = np.geomspace(sigma_min, sigma_max, n)
    areas = np.array([surface_area(grid, mask, s) for s in sigmas])
    ball = equivalent_ball(grid, int(mask.sum()))
    ball_areas = np.array([surface_area(grid, ball, s) for s in sigmas])
    slope, intercept = np.polyfit(np.log(sigmas), np.log(areas / ball_areas), 1)
    raw_slope = np.polyfit(np.log(sigmas), np.log(areas), 1)[0]
    return {"D": 2.0 - slope, "D_raw": 2.0 - raw_slope, "sigmas": sigmas.tolist(), "areas": areas.tolist(),
            "ball_areas": ball_areas.tolist(), "fit": [float(slope), float(intercept)]}


def richardson_range(resolution: float, radius: float, apix: float) -> tuple[float, float]:
    """Blur scales between the resolution and the molecule radius."""
    return max(resolution / 2, apix), radius / 3


def theoretical_alpha(D: float) -> float:
    return 9.0 - 2.0 * D


def load_calibration(path: Path = CALIBRATION_FILE) -> dict | None:
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def alpha_for_D(D: float, calibration: dict | None = None) -> tuple[float, str]:
    """Map a target D to the GRF exponent. Returns (alpha, source)."""
    cal = calibration if calibration is not None else load_calibration()
    if cal is None:
        return theoretical_alpha(D), "theory"
    alphas = np.array(cal["alpha"])
    Ds = np.array(cal["D_mean"])
    order = np.argsort(Ds)
    Ds, alphas = Ds[order], alphas[order]
    if not Ds[0] <= D <= Ds[-1]:
        warnings.warn(f"roughness_D={D} outside calibrated range [{Ds[0]:.3f}, {Ds[-1]:.3f}]; clamping")
    return float(np.interp(D, Ds, alphas)), "calibration"


def sphericity(volume: float, area: float) -> float:
    return math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area
