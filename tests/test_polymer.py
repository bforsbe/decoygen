import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from decoygen import polymer


@pytest.fixture(scope="module")
def ref():
    return polymer.load_reference()


@pytest.fixture(scope="module")
def blob():
    """~300 points at protein heavy-atom density (1 per ~17 A^3) in a ball."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-11, 11, size=(1500, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < 6.8 * 1.6]
    return pts[:300] + 20.0


def degrees(edges, n):
    return np.bincount(edges.ravel(), minlength=n)


def test_reference_geometry(ref):
    q = ref["quantiles"]
    assert 1.5 < q["bond"]["C-C"][50] < 1.56
    assert 1.2 < q["bond"]["C-O"][50] < 1.28
    assert 2.35 < q["angle"]["*"][50] < 2.5
    assert q["nonbonded"]["N-O"][1] < q["nonbonded"]["C-C"][1]       # hydrogen bonds
    frac = polymer.degree_fractions(ref)
    assert frac[0] == 0 and abs(frac[1:].sum() - 1) < 1e-9


def test_graph_shells_on_a_chain():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    _, p13, p14 = polymer.graph_shells(5, edges)
    assert {tuple(p) for p in p13} == {(0, 2), (1, 3), (2, 4)}
    assert {tuple(p) for p in p14} == {(0, 3), (1, 4)}


def test_forest_is_a_capped_tree(blob):
    e = polymer.spanning_forest(blob, seed=1)
    n = len(blob)
    assert degrees(e, n).max() <= polymer.MAX_DEGREE
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp = connected_components(a, directed=False)[0]
    assert n_comp == n - len(e)                                     # a forest: no cycles
    assert n_comp <= 3                                              # nearly one tree


def test_branch_weight_increases_branching(blob):
    n = len(blob)
    d1 = degrees(polymer.spanning_forest(blob, branch_weight=1.0), n)
    d2 = degrees(polymer.spanning_forest(blob, branch_weight=0.8), n)
    assert (d2 == 3).sum() > (d1 == 3).sum()


def test_relaxation_matches_bonds_and_clears_clashes(blob, ref):
    pos = torch.tensor(blob, dtype=torch.float32)
    p, _, edges, info = polymer.build_polymer(pos, ref, steps=600)
    assert info["bond_rmsd_A"] < 0.1
    assert info["angle_rmsd_A"] < 0.1
    assert info["steric_violations_gt_0.2A"] <= 2
    d = (p[edges[:, 0]] - p[edges[:, 1]]).norm(dim=1)
    assert 1.2 < float(d.min()) and float(d.max()) < 1.95
