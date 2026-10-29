"""Empirical (alpha, roughness amplitude) -> Richardson D table (brief section 4.1, step 5)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import roughness
from .envelope import EnvelopeError, make_envelope
from .fourier import Grid
from .params import DecoyParams

ENVELOPE_FIELDS = ("box", "apix", "resolution", "mass_kda", "compactness", "n_domains",
                   "elongation", "envelope_bias", "edge_clearance")


def calibrate(base: DecoyParams, alphas, amplitudes, seeds, device="cpu", log=print) -> dict:
    grid = Grid(base.box, base.apix, device)
    smin, smax = roughness.richardson_range(base.resolution, base.equivalent_radius, base.apix)
    D_mean = np.full((len(alphas), len(amplitudes)), np.nan)
    D_std = np.full_like(D_mean, np.nan)
    n_ok = np.zeros(D_mean.shape, dtype=int)
    for i, alpha in enumerate(alphas):
        for j, amp in enumerate(amplitudes):
            t = time.perf_counter()
            Ds = []
            for seed in seeds:
                try:
                    mask, _ = make_envelope(
                        grid, volume=base.volume, alpha=float(alpha), roughness_amplitude=float(amp),
                        compactness=base.compactness, n_domains=base.n_domains, elongation=base.elongation,
                        bias=base.envelope_bias, clearance=base.edge_clearance, seed=int(seed))
                except EnvelopeError:
                    continue
                Ds.append(roughness.richardson(grid, mask, smin, smax)["D"])
            n_ok[i, j] = len(Ds)
            if Ds:
                D_mean[i, j], D_std[i, j] = np.mean(Ds), np.std(Ds)
            log(f"alpha={alpha:.2f} amplitude={amp:.2f} D={D_mean[i, j]:.4f} +- {D_std[i, j]:.4f} "
                f"(n={len(Ds)}, {time.perf_counter() - t:.1f}s)", flush=True)
    return {
        "alpha": [float(a) for a in alphas],
        "amplitude": [float(a) for a in amplitudes],
        "D_mean": D_mean.tolist(), "D_std": D_std.tolist(), "n_seeds": len(seeds),
        "n_ok": n_ok.tolist(),
        "richardson_sigma_range_A": [smin, smax],
        "base_params": {k: v for k, v in dataclasses.asdict(base).items() if k in ENVELOPE_FIELDS},
        "note": "Rows: alpha; columns: roughness amplitude. Valid for base_params envelope settings.",
    }
