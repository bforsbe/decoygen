import numpy as np
import pytest

from decoygen.envelope import EnvelopeError
from decoygen.fourier import Grid
from decoygen.symmetry import cyclic, make_symmetric_envelope, neighbour_distance, point_group, rotation_from_z

KW = {"alpha": 4.5, "roughness_amplitude": 0.6, "compactness": 4.0, "bias": 1.5, "clearance": 4.0, "seed": 0}


@pytest.fixture(scope="module")
def grid():
    return Grid(64, 1.0, "cpu")


def build(grid, sym, **kw):
    return make_symmetric_envelope(grid, symmetry=sym, volume=24_000.0, **{**KW, **kw})


def test_groups_are_rotations_closed_and_general():
    for n in (1, 2, 3, 7):
        g = point_group(f"C{n}")
        assert np.allclose(g @ g.transpose(0, 2, 1), np.eye(3)) and np.allclose(np.linalg.det(g), 1)
        prods = (g[:, None] @ g[None]).reshape(-1, 3, 3)                 # closed under composition
        assert all(np.abs(g - h).max(axis=(1, 2)).min() < 1e-9 for h in prods)
    d, _ = neighbour_distance(cyclic(5), np.array([0, 1.0, 0]))
    assert np.isclose(d, 2 * np.sin(np.pi / 5))
    with pytest.raises(NotImplementedError, match="dihedral"):
        point_group("D2")
    a = np.array([0.3, -0.5, 0.8])
    assert np.allclose(rotation_from_z(a) @ [1, 0, 0], a / np.linalg.norm(a))


def test_c3_subunits_are_equal_whole_and_touching(grid):
    e = build(grid, "C3")
    i = e.info
    v = np.array(i["subunit_volumes_A3"])
    assert abs(i["volume_A3"] / 24_000 - 1) < 0.02
    assert v.max() / v.min() < 1.03 and i["subunit_pieces"] == [1, 1, 1]
    area = np.array(i["interface_A2"])
    assert (area[~np.eye(3, dtype=bool)] > 0).all()                         # every pair of a C3 ring touches
    assert i["symmetry_mismatch_fraction_max"] < 0.03


def test_c4_on_grid_axes_is_exactly_symmetric(grid):
    e = build(grid, "C4")
    assert e.info["symmetry_mismatch_fraction_max"] == 0.0
    assert len(np.unique(e.labels)) == 5                                     # solvent + 4 subunits


def test_off_centre_oblique_axis(grid):
    e = build(grid, "C3", axis=np.array([0.4, 0.2, 0.9]), shift=np.array([2.0, -2.0, 1.0]))
    assert e.info["subunit_pieces"] == [1, 1, 1]
    assert np.allclose(e.centre, 32 + np.array([2.0, -2.0, 1.0]))
    for g in e.ops:                                                           # the operators keep the axis
        assert np.allclose(g @ np.array(e.info["axis_zyx"]), e.info["axis_zyx"])


def test_copies_that_do_not_touch_are_an_error(grid):
    with pytest.raises(EnvelopeError, match="do not touch"):
        build(grid, "C3", contact=-1.5)


def test_swap_needs_a_partner(grid):
    with pytest.raises(ValueError, match="partner"):
        build(grid, "C1", swap=1.0)
