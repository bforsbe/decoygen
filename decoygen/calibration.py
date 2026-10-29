"""Empirical alpha -> D lookup table for the envelope pipeline (brief section 4.1, step 5)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import roughness
from .envelope import EnvelopeError, make_envelope
from .fourier import Grid
from .params import DecoyParams


def calibrate(base: DecoyParams, alphas, seeds, device="cpu", log=print) -> dict:
    grid = Grid(base.box, base.apix, device)
    smin, smax = roughness.richardson_range(base.resolution, base.equivalent_radius, base.apix)
    rows = []
    for alpha in alphas:
        Ds = []
        for seed in seeds:
            t = time.perf_counter()
            try:
                mask, _ = make_envelope(
                    grid, volume=base.volume, alpha=float(alpha), compactness=base.compactness,
                    n_domains=base.n_domains, elongation=base.elongation, bias=base.envelope_bias,
                    clearance=base.edge_clearance, seed=int(seed))
            except EnvelopeError:
                continue
            Ds.append(roughness.richardson(grid, mask, smin, smax)["D"])
            log(f"alpha={alpha:.2f} seed={seed} D={Ds[-1]:.4f} ({time.perf_counter() - t:.1f}s)")
        rows.append((float(alpha), float(np.mean(Ds)), float(np.std(Ds)), len(Ds)))
    base_dict = {k: v for k, v in dataclasses.asdict(base).items()
                 if k in ("box", "apix", "resolution", "mass_kda", "compactness", "n_domains",
                          "elongation", "envelope_bias", "edge_clearance")}
    return {
        "alpha": [r[0] for r in rows], "D_mean": [r[1] for r in rows],
        "D_std": [r[2] for r in rows], "n": [r[3] for r in rows],
        "richardson_sigma_range_A": [smin, smax],
        "base_params": base_dict,
        "note": "Valid for the base_params envelope settings; other settings shift the mapping.",
    }
