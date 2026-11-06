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


def test_element_caps_respected(blob):
    els = np.random.default_rng(3).choice(4, size=len(blob), p=[0.63, 0.17, 0.19, 0.01])
    caps = polymer.element_caps(els)
    e = polymer.spanning_forest(blob, max_degree=caps)
    deg = degrees(e, len(blob))
    assert (deg <= caps).all()
    assert deg[caps == 1].max() <= 1                                 # oxygens stay terminal


def test_rewiring_keeps_targets_of_surviving_bonds(blob, ref):
    els = np.zeros(len(blob), dtype=np.int64)
    e1 = polymer.spanning_forest(blob, seed=0)
    e2 = polymer.spanning_forest(blob, seed=1)
    r1 = polymer.Restraints(els, e1, ref, "cpu", seed=0)
    r2 = polymer.Restraints(els, e2, ref, "cpu", seed=5, previous=r1)
    shared, i1, i2 = np.intersect1d(r1.keys12, r2.keys12, return_indices=True)
    assert len(shared) > 0
    assert np.array_equal(r1.b0_np[i1], r2.b0_np[i2])


def test_soft_bonds_recover_a_relaxed_polymer_and_can_rewire(blob, ref):
    els = np.zeros(len(blob), dtype=np.int64)                  # all carbon
    edges = polymer.spanning_forest(blob, max_degree=polymer.element_caps(els))
    x, _ = polymer.relax(torch.tensor(blob, dtype=torch.float32), els, edges, ref, steps=1500)
    soft = polymer.SoftChemistry(els, degrees(edges, len(els)), ref, "cpu")
    stats = polymer.topology_stats(len(els), soft.bonds(x), edges)
    assert stats["bonds_new"] == 0 and stats["bonds_lost"] <= 2   # a relaxed tree reads back as itself

    y = x.clone()
    y[:30] += 1.0 * torch.nn.functional.normalize(torch.randn(30, 3, generator=torch.Generator().manual_seed(0)), dim=1)
    y.requires_grad_(True)
    opt = torch.optim.Adam([y], lr=0.01)
    for i in range(300):
        if i % 10 == 0:
            soft.update_neighbours(y.detach())
        loss = sum(soft.terms(y, 0.05).values()) / len(els)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = soft.bonds(y.detach())
    assert polymer.topology_stats(len(els), after, edges)["bonds_new"] > 0
    assert (degrees(after, len(els)) <= 3).all()
