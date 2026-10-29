"""Surface roughness: Richardson-plot fractal dimension and the (alpha, amplitude) -> D calibration.

Area at scale sigma is the total variation of the Gaussian-blurred indicator,
A(sigma) = int |grad(G_sigma * chi)| dV, which by the coarea formula averages the
areas of the level sets of the smoothed object. A(sigma) ~ sigma^(2 - D).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from .fourier import Grid

CALIBRATION_FILE = Path(__file__).parent / "data" / "roughness_calibration.json"


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


def load_calibration(path: Path | None = None) -> dict | None:
    """Calibration table: `path`, else $DECOYGEN_CALIBRATION, else the packaged file."""
    path = Path(path or os.environ.get("DECOYGEN_CALIBRATION") or CALIBRATION_FILE)
    if not path.exists():
        return None
    return json.loads(path.read_text())


DEFAULT_ROUGHNESS_AMPLITUDE = 0.3


def amplitude_for_D(D: float, alpha: float, calibration: dict | None = None) -> tuple[float, str]:
    """Fine-roughness amplitude giving Richardson D at the given alpha. Returns (amplitude, source).

    The table holds mean measured D on an (alpha x amplitude) grid. Rows are linearly
    interpolated in alpha, made monotone in amplitude, and inverted.
    """
    cal = calibration if calibration is not None else load_calibration()
    if cal is None:
        return DEFAULT_ROUGHNESS_AMPLITUDE, "uncalibrated"
    alphas = np.array(cal["alpha"])
    amps = np.array(cal["amplitude"])
    Dgrid = np.array(cal["D_mean"])              # shape (len(alphas), len(amps))
    a = float(np.clip(alpha, alphas[0], alphas[-1]))
    j = int(np.clip(np.searchsorted(alphas, a) - 1, 0, len(alphas) - 2))
    w = (a - alphas[j]) / (alphas[j + 1] - alphas[j])
    row = np.maximum.accumulate((1 - w) * Dgrid[j] + w * Dgrid[j + 1])
    if not row[0] <= D <= row[-1]:
        warnings.warn(f"roughness_D={D} outside the calibrated range [{row[0]:.3f}, {row[-1]:.3f}] "
                      f"at alpha={alpha:.2f}; clamping")
    # np.interp needs strictly increasing x; drop plateaus
    keep = np.concatenate([[True], np.diff(row) > 0])
    return float(np.interp(D, row[keep], amps[keep])), "calibration"


def roughness_parameters(D: float, alpha: float | None, amplitude: float | None,
                         calibration: dict | None = None) -> dict:
    """Resolve (alpha, amplitude) for a target D; explicit values win."""
    out = {"alpha": alpha if alpha is not None else theoretical_alpha(D),
           "alpha_source": "user" if alpha is not None else "theory"}
    if amplitude is not None:
        out.update(roughness_amplitude=amplitude, amplitude_source="user")
    else:
        amp, src = amplitude_for_D(D, out["alpha"], calibration)
        out.update(roughness_amplitude=amp, amplitude_source=src)
    return out


def sphericity(volume: float, area: float) -> float:
    return math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area
