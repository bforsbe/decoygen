import json
import os

import mrcfile
import numpy as np
import pytest
import torch

from decoygen import DecoyParams, diagnostics
from decoygen.fourier import Grid
from decoygen.io import check_mrc
from decoygen.pipeline import generate, save

SMALL = {"box": 64, "apix": 1.0, "resolution": 4.0, "mass_kda": 12.0, "edge_clearance": 5.0,
         "turing_iterations": 80, "gs_iterations": 10, "alpha": 4.4,
         "roughness_amplitude": 0.5}


@pytest.fixture(scope="module")
def decoy():
    return generate(DecoyParams(seed=7, **SMALL))


def test_deterministic(decoy):
    again = generate(DecoyParams(seed=7, **SMALL))
    for k in decoy.maps:
        assert np.array_equal(decoy.maps[k], again.maps[k]), k


def test_seed_changes_output(decoy):
    other = generate(DecoyParams(seed=8, **SMALL), with_diagnostics=False)
    assert not np.array_equal(decoy.maps["clean"], other.maps["clean"])


def test_volume(decoy):
    assert decoy.diagnostics["volume_ratio"] == pytest.approx(1.0, abs=0.02)


def test_spectrum_matches_target(decoy):
    assert decoy.diagnostics["spectrum_log10_rms_error"] < 0.05
    hist = decoy.info["gs_spectral_error_history"]
    assert hist[-1] <= hist[0]


def test_halfmap_fsc_at_resolution(decoy):
    assert decoy.diagnostics["halfmap_fsc_resolution_A"] == pytest.approx(SMALL["resolution"], rel=0.15)


def test_peaks_target_raises_power_at_peak():
    p = DecoyParams(seed=7, target_spectrum="physics+peaks", peak_d=(6.0,), peak_heights=(1.0,), **SMALL)
    d = generate(p)
    s = d.curves["shell_s"]
    i = int(np.argmin(np.abs(s - 1 / 6.0)))
    ratio = d.curves["power_signal"][i] / d.curves["power_reference"][i]
    assert ratio > 1.4


def test_tier3_requires_opt_in():
    with pytest.raises(ValueError):
        DecoyParams(target_spectrum="best", spectrum_file="x.txt").validate()


def test_chirality_flips_under_mirror():
    g = Grid(48, 1.0)
    x = g.gaussian_filter(torch.randn(g.shape, generator=torch.Generator().manual_seed(0)), 1.5)
    m = torch.ones(g.shape, dtype=torch.bool)
    c = diagnostics.chirality(g, x, m)
    c_mirror = diagnostics.chirality(g, torch.flip(x, dims=(2,)), m)
    assert c_mirror == pytest.approx(-c, rel=1e-4, abs=1e-6)


def test_outputs_and_headers(decoy, tmp_path):
    d = save(decoy, tmp_path, plots=True)
    name = d.name
    meta = json.loads((d / f"{name}.json").read_text())
    assert meta["params"]["seed"] == 7
    assert "richardson_D" in meta["diagnostics"]
    for suffix in ("", "_half_1", "_half_2", "_clean", "_mask"):
        path = d / f"{name}{suffix}.mrc"
        check_mrc(path, (64,) * 3, 1.0)
        with open(os.devnull, "w") as devnull:
            assert mrcfile.validate(str(path), print_file=devnull)
    assert (d / f"{name}_diagnostics.png").exists()
    with mrcfile.open(d / f"{name}_mask.mrc") as m:
        assert set(np.unique(m.data)) <= {0.0, 1.0}


def test_full_map_is_mean_of_halves(decoy):
    m = decoy.maps
    assert np.allclose(m["full"], (m["half_1"] + m["half_2"]) / 2, atol=1e-6)
