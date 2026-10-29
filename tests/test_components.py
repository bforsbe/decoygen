import math

import numpy as np
import pytest
import torch

from decoygen import atoms, roughness, turing
from decoygen.envelope import largest_component, make_envelope, periodic_label
from decoygen.fourier import Grid


@pytest.fixture(scope="module")
def grid():
    return Grid(64, 1.0)


def sphere(n, radius, centre=None):
    c = n / 2 if centre is None else centre
    z, y, x = np.indices((n, n, n))
    return (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2


def test_white_noise_spectrum_flat(grid):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(grid.shape, generator=g)
    p = grid.power_spectrum(x)[4:grid.nyquist_shell].numpy()
    assert np.allclose(p / grid.n**3, 1.0, atol=0.15)


def test_shell_mean_matches_bincount(grid):
    v = torch.rand(grid.s.shape, generator=torch.Generator().manual_seed(1))
    ref = np.bincount(grid.shell.reshape(-1).numpy(), v.reshape(-1).double().numpy()) / np.maximum(
        grid.shell_counts.numpy(), 1)
    assert np.allclose(grid.shell_mean(v).numpy(), ref)


def test_gaussian_filter_preserves_sum(grid):
    x = torch.from_numpy(sphere(64, 10).astype(np.float32))
    assert float(grid.gaussian_filter(x, 3.0).sum()) == pytest.approx(float(x.sum()), rel=1e-4)


def test_dog_peak_position():
    for wavelength, q in [(10.0, 4.0), (7.0, 2.5)]:
        g = Grid(128, 0.5)
        G = turing.growth_function(g, wavelength, q)
        s_peak = float(g.s.reshape(-1)[G.reshape(-1).argmax()])
        assert s_peak == pytest.approx(1 / wavelength, rel=0.05)


def test_dog_old_vermiculation_preset():
    # brief: sigma_s=1, sigma_l=4 px gives lambda ~ 10.3 px
    ss, sl = turing.dog_sigmas(10.3, 4.0)
    assert ss == pytest.approx(1.0, rel=0.01) and sl == pytest.approx(4.0, rel=0.01)


def test_turing_fill_fraction(grid):
    mask = sphere(64, 24)
    for fill in (0.3, 0.5):
        field, info = turing.turing_field(grid, mask, wavelength=10.0, quotient=4.0, fill_fraction=fill,
                                          iterations=150, dt=0.3, seed=0)
        assert info["turing_fill_realized"] == pytest.approx(fill, abs=0.03)
        assert float(field[~torch.from_numpy(mask)].abs().max()) == 0.0


def test_sphere_richardson_D(grid):
    r = roughness.richardson(grid, sphere(64, 20, centre=31), 1.5, 6.0)
    assert r["D"] == pytest.approx(2.0, abs=0.01)


def test_smooth_ellipsoid_richardson_D():
    g = Grid(96, 1.0)
    z, y, x = np.indices(g.shape) - 48
    e = 1.6
    m = (z / (18 * e ** (2 / 3))) ** 2 + (y / (18 / e ** (1 / 3))) ** 2 + (x / (18 / e ** (1 / 3))) ** 2 <= 1
    assert roughness.richardson(g, m, 1.75, 6.0)["D"] == pytest.approx(2.0, abs=0.03)


def test_sphere_area(grid):
    a = roughness.surface_area(grid, sphere(64, 20), 1.0)
    assert a == pytest.approx(4 * math.pi * 20**2, rel=0.05)


def test_periodic_label_merges_across_faces():
    m = np.zeros((16, 16, 16), bool)
    m[0:2, 4:6, 4:6] = True
    m[-2:, 4:6, 4:6] = True
    m[8, 8, 8] = True
    _, sizes = periodic_label(m)
    assert sorted(sizes[sizes > 0].tolist()) == [1, 16]
    assert largest_component(m).sum() == 16


def test_envelope_volume_clearance_connectivity(grid):
    volume = 25000.0
    mask, _ = make_envelope(grid, volume=volume, alpha=4.5, roughness_amplitude=0.6, compactness=8.0,
                            n_domains=1, elongation=1.0, bias=1.5, clearance=6.0, seed=3)
    assert mask.sum() == pytest.approx(volume, rel=0.02)
    assert not mask[:6].any() and not mask[-6:].any()
    _, sizes = periodic_label(mask)
    assert (sizes > 0).sum() == 1


def test_envelope_has_no_cavities(grid):
    from scipy import ndimage

    mask, _ = make_envelope(grid, volume=15000.0, alpha=4.0, roughness_amplitude=1.5, compactness=0.0,
                            n_domains=1, elongation=1.0, bias=1.5, clearance=3.0, seed=1)
    assert np.array_equal(mask, ndimage.binary_fill_holes(mask))


def test_larger_roughness_amplitude_gives_higher_D(grid):
    Ds = []
    for amp in (0.0, 1.0):
        mask, _ = make_envelope(grid, volume=15000.0, alpha=4.4, roughness_amplitude=amp, compactness=0.0,
                                n_domains=1, elongation=1.0, bias=1.5, clearance=3.0, seed=0)
        Ds.append(roughness.richardson(grid, mask, 1.75, 5.0)["D"])
    assert Ds[1] > Ds[0] + 0.03


def test_amplitude_for_D_inverts_table():
    cal = {"alpha": [4.0, 5.0], "amplitude": [0.0, 0.5, 1.0],
           "D_mean": [[2.05, 2.20, 2.40], [2.03, 2.10, 2.20]]}
    amp, src = roughness.amplitude_for_D(2.30, 4.0, cal)
    assert src == "calibration" and amp == pytest.approx(0.75)
    amp, _ = roughness.amplitude_for_D(2.15, 4.5, cal)     # row = [2.04, 2.15, 2.30]
    assert amp == pytest.approx(0.5)
    with pytest.warns(UserWarning):
        roughness.amplitude_for_D(2.6, 4.0, cal)
    p = roughness.roughness_parameters(2.25, None, None, cal)
    assert p["alpha"] == pytest.approx(4.5) and p["alpha_source"] == "theory"


def test_render_integral_equals_f0(grid):
    pos = torch.tensor([[20.3, 31.7, 12.1], [40.0, 40.5, 41.2]])
    el = torch.tensor([atoms.ELEMENTS.index("C"), atoms.ELEMENTS.index("O")])
    rho = atoms.render(grid, pos, el, bfactor=30.0)
    f0 = sum(sum(atoms.form_factor_coefs(e)[0]) for e in ("C", "O"))
    assert float(rho.double().sum()) * grid.apix**3 == pytest.approx(f0, rel=1e-3)


def test_render_matches_analytic_form_factor():
    g = Grid(64, 0.5)
    pos = torch.tensor([[16.0, 16.0, 16.0]])
    el = torch.tensor([atoms.ELEMENTS.index("N")])
    rho = atoms.render(g, pos, el, bfactor=20.0)
    F = (g.rfft(rho).abs() * g.apix**3)
    s = g.s
    f = atoms.form_factor("N", s, 20.0)
    sel = s < 0.6
    assert torch.allclose(F[sel], f[sel], rtol=2e-3, atol=1e-4)


def test_render_against_gemmi():
    gemmi = pytest.importorskip("gemmi")
    n, apix, B = 48, 1.0, 25.0
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(n * apix, n * apix, n * apix, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "UNX"
    res.het_flag = "H"
    res.seqid = gemmi.SeqId(1, " ")
    xyz = [(10.2, 20.7, 30.1), (25.5, 24.0, 18.3)]
    for i, (p, e) in enumerate(zip(xyz, ("C", "O"))):
        a = gemmi.Atom()
        a.name = f"{e}{i}"
        a.element = gemmi.Element(e)
        a.pos = gemmi.Position(*p)
        a.b_iso = B
        a.occ = 1.0
        res.add_atom(a)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    dc = gemmi.DensityCalculatorE()
    dc.d_min = 2 * apix
    dc.rate = 1.0
    dc.blur = 0.0
    dc.cutoff = 1e-6
    dc.grid.setup_from(st)
    dc.grid.set_size(n, n, n)
    dc.put_model_density_on_grid(st[0])
    ref = np.array(dc.grid, copy=False)            # indexed [x, y, z]
    g = Grid(n, apix)
    # our arrays are [z, y, x] with position (z, y, x)
    pos = torch.tensor([[p[2], p[1], p[0]] for p in xyz], dtype=torch.float32)
    el = torch.tensor([atoms.ELEMENTS.index("C"), atoms.ELEMENTS.index("O")])
    ours = atoms.render(g, pos, el, B).numpy().transpose(2, 1, 0)
    assert np.allclose(ours, ref, atol=2e-3 * ref.max())
