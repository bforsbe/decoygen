"""End-to-end v0 decoy generation."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import __version__, atoms, diagnostics, io, noise, projection, roughness, spectrum
from .constants import MASS_PER_HEAVY_ATOM
from .envelope import make_envelope
from .fourier import Grid
from .params import DecoyParams
from .turing import turing_field

SUPPORT_BLUR = 2.0         # A; the support extends past the envelope by the atomic spread
SUPPORT_LEVEL = 0.01


@dataclass
class Decoy:
    params: DecoyParams
    maps: dict            # name -> np.ndarray float32 [z, y, x]
    info: dict
    diagnostics: dict
    curves: dict = field(default_factory=dict)




def generate(params: DecoyParams, device: str = "cpu", with_diagnostics: bool = True) -> Decoy:
    params.validate()
    t0 = time.perf_counter()
    timings = {}

    def tick(name):
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        timings[name] = time.perf_counter() - t0 - sum(timings.values())

    grid = Grid(params.box, params.apix, device)
    rough = roughness.roughness_parameters(params.roughness_D, params.alpha, params.roughness_amplitude)
    mask, attempt = make_envelope(
        grid, volume=params.volume, alpha=rough["alpha"], roughness_amplitude=rough["roughness_amplitude"],
        compactness=params.compactness,
        n_domains=params.n_domains, elongation=params.elongation, bias=params.envelope_bias,
        clearance=params.edge_clearance, seed=params.seed)
    m = torch.as_tensor(mask, device=grid.device)
    tick("envelope")

    turing, turing_info = turing_field(
        grid, mask, wavelength=params.meso_wavelength, quotient=params.turing_quotient,
        fill_fraction=params.fill_fraction, iterations=params.turing_iterations,
        dt=params.turing_dt, seed=params.seed)
    intensity = ((1 - params.meso_strength) + params.meso_strength * turing) * m
    tick("turing")

    n_atoms = round(params.mass_kda * 1e3 / MASS_PER_HEAVY_ATOM)
    ref_pos, ref_el = atoms.sample_gas(grid, m.float(), n_atoms, params.seed, "reference")
    reference = atoms.render(grid, ref_pos, ref_el, params.bfactor)
    pos, el = atoms.sample_gas(grid, intensity, n_atoms, params.seed, "atoms")
    x0 = atoms.render(grid, pos, el, params.bfactor)
    support = grid.gaussian_filter(m.float(), SUPPORT_BLUR) > SUPPORT_LEVEL
    sorted_hist = torch.sort(reference[support]).values
    tick("atoms")

    target = spectrum.target_power(grid, grid.power_spectrum(reference), n_atoms, params)
    radius = params.equivalent_radius
    err_shells = slice(max(1, int(grid.n * grid.apix / radius)), grid.nyquist_shell + 1)
    if params.gs_iterations > 0:
        signal, gs_history = projection.gerchberg_saxton(
            grid, x0, target, support, sorted_hist, params.gs_iterations, err_shells)
    else:
        signal, gs_history = x0, []
    tick("projection")

    clean = grid.cosine_lowpass(signal, 1 / params.resolution, 0.1 / params.resolution)
    half1, half2, noise_info = noise.half_maps(
        grid, signal, params.resolution, params.snr_scale, params.noise_model, params.seed)
    full = (half1 + half2) / 2
    tick("render")

    diag, curves = ({}, {})
    if with_diagnostics:
        diag, curves = diagnostics.compute(
            grid, params, mask=mask, support=support, clean=clean, signal=signal,
            reference=reference, half1=half1, half2=half2, target_power=target)
        tick("diagnostics")

    info = {
        **rough, "envelope_attempt": attempt,
        "n_atoms": n_atoms, **turing_info, **noise_info,
        "gs_spectral_error_history": gs_history,
        "timings_s": timings, "device": str(grid.device),
        "versions": versions(),
        "notes": "v0: Poisson pseudo-atom gas + Gerchberg-Saxton; pseudo-atoms are not "
                 "the final map and are not written (milestone 2).",
    }
    maps = {
        "full": full, "half_1": half1, "half_2": half2, "clean": clean,
        "mask": m.float(), "turing": turing,
    }
    maps = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in maps.items()}
    return Decoy(params, maps, info, diag, curves)


def versions() -> dict:
    import gemmi
    import mrcfile
    import scipy

    return {"decoygen": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "gemmi": gemmi.__version__,
            "mrcfile": mrcfile.__version__}


OUTPUT_MAPS = {"full": "", "half_1": "_half_1", "half_2": "_half_2", "clean": "_clean", "mask": "_mask"}


def decoy_id(seed: int) -> str:
    return f"decoy_{seed:06d}"


def save(decoy: Decoy, outdir: str | Path, name: str | None = None, plots: bool = False,
         extra_maps: bool = False) -> Path:
    name = name or decoy_id(decoy.params.seed)
    d = Path(outdir) / name
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, suffix in OUTPUT_MAPS.items():
        p = d / f"{name}{suffix}.mrc"
        io.write_mrc(p, decoy.maps[key], decoy.params.apix)
        files[key] = p.name
    if extra_maps:
        p = d / f"{name}_turing.mrc"
        io.write_mrc(p, decoy.maps["turing"], decoy.params.apix)
        files["turing"] = p.name
    if plots:
        from .plots import summary_figure

        p = d / f"{name}_diagnostics.png"
        summary_figure(decoy, p)
        files["plot"] = p.name
    io.write_json(d / f"{name}.json", {
        "id": name, "params": decoy.params.to_dict(), "files": files,
        "info": decoy.info, "diagnostics": decoy.diagnostics,
    })
    return d
