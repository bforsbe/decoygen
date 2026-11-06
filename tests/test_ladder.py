import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from decoygen import cli, ladder, peptide, sidechains
from decoygen.model import Model, RungError, expand, read, require, write
from decoygen.params import DecoyParams

SMALL = {"box": 64, "apix": 1.0, "mass_kda": 12.0, "edge_clearance": 5.0, "turing_iterations": 80,
         "alpha": 4.4, "roughness_amplitude": 0.5}


def pieces(m: Model) -> int:
    a = sparse.coo_matrix((np.ones(len(m.edges)), (m.edges[:, 0], m.edges[:, 1])), shape=(m.n, m.n))
    return connected_components(a, directed=False)[0]


@pytest.fixture(scope="module")
def rungs():
    m1 = ladder.gas(DecoyParams(seed=3, **SMALL))
    m2, _ = ladder.build_polymer(m1, steps=300)
    m3, info3 = ladder.thread_chain(m2, relax=300)
    m4, info4 = ladder.make_peptide(m3, steps=2000)
    return m1, m2, m3, m4, info4, info3


@pytest.fixture(scope="module")
def rung5(rungs):
    return ladder.make_sidechains(rungs[3], steps=900, stages=3)


def toy(rung, mirrored=False):
    return Model(pos=np.zeros((1, 3)), els=np.zeros(1, dtype=np.int64), edges=np.zeros((0, 2), dtype=np.int64),
                 box=10.0, rung=rung, mirrored=mirrored)


def test_require_refuses_skipped_repeated_mirrored_and_foreign_models():
    require(toy(3), 3, "peptide")
    with pytest.raises(RungError, match="run `decoygen thread` first"):
        require(toy(2), 3, "peptide")
    with pytest.raises(RungError, match="`decoygen polymer`, `decoygen thread`"):
        require(toy(1), 3, "peptide")
    with pytest.raises(RungError, match="already rung 4"):
        require(toy(4), 3, "peptide")
    with pytest.raises(RungError, match="mirrored"):
        require(toy(3, mirrored=True), 3, "peptide")
    with pytest.raises(RungError, match="no decoygen header"):
        require(toy(None), 3, "peptide")
    require(toy(None), 3, "peptide", force=True)


def test_rungs_and_history(rungs):
    m1, m2, m3, m4, _, _ = rungs
    assert [m.rung for m in rungs[:4]] == [1, 2, 3, 4]
    assert [h.split()[0] for h in m4.history] == ["gas", "polymer", "thread", "peptide"]
    assert len(m1.edges) == 0 and len(m2.edges) > 0.9 * m2.n
    assert pieces(m3) == 1 and pieces(m4) == 1                       # one molecule from rung 3 on


def test_model_file_round_trip(rungs, tmp_path):
    m4 = rungs[3]
    back = read(write(tmp_path / "m.pdb", m4))
    assert (back.rung, back.mirrored, back.history, back.box) == (4, False, m4.history, m4.box)
    assert back.n == m4.n and len(back.edges) == len(m4.edges)
    with pytest.raises(FileExistsError):
        write(tmp_path / "m.pdb", m4)


def test_peptide_is_trans_planar_and_L(rungs):
    _, _, m3, m4, info, _ = rungs
    assert m4.n == m3.n                                               # carbonyl O paid for by side atoms
    assert info["L_fraction"] == 1.0
    assert info["omega_trans_fraction_gt150"] > 0.97
    assert abs(info["CO_bond_median_A"] - 1.23) < 0.03
    assert info["steric_violations_gt_0.2A"] == 0
    names = m4.names
    assert sum(x == "CA" for x in names) == info["residues"] == sum(x == "O" for x in names)


def test_mirror_flips_handedness_and_is_marked(rungs):
    m4 = rungs[3]
    mm = ladder.mirror(m4)
    assert mm.mirrored and mm.rung == 4 and mm.history[-1] == "mirror"
    atom = {(r, nm): i for i, (r, nm) in enumerate(zip(m4.resseq, m4.names))}
    with_cb = sorted(r for r, nm in atom if nm == "CB")
    assert with_cb

    def chirality(m):
        return peptide.chiral_volume(*(m.pos[[atom[(r, nm)] for r in with_cb]] for nm in ("N", "CA", "C", "CB")))

    assert (chirality(m4) > 0).all() and (chirality(mm) < 0).all()


def test_cli_refuses_the_wrong_rung(rungs, tmp_path, capsys):
    p = write(tmp_path / "r2.pdb", rungs[1])
    with pytest.raises(SystemExit) as e:
        cli.main(["peptide", str(p), "--device", "cpu"])
    assert e.value.code == 2
    assert "run `decoygen thread` first" in capsys.readouterr().err


def test_sidechains_thin_to_protein_mass_per_residue_and_keep_the_mass(rungs, rung5):
    m4 = rungs[3]
    m5, info = rung5
    assert m5.rung == 5 and pieces(m5) == 1
    assert m5.n == m4.n                                               # mass preserved exactly
    assert 7.3 < info["atoms_per_residue"] < 8.3                     # protein: ~7.8
    for p in info["thinning_passes"]:                                 # at most 1 in 5 per pass, joins healed
        assert p["junction_CA_CA_after_max_A"] < 4.2
    assert info["L_fraction"] == 1.0 and info["omega_trans_fraction_gt150"] > 0.97
    assert info["steric_violations_gt_0.2A"] == 0
    assert set(m5.resnames) <= set(sidechains.TOPOLOGY) and len(set(m5.resnames)) > 10


def test_library_and_placement_reproduce_real_side_chains():
    lib = sidechains.load_library()
    assert set(lib) == set(sidechains.TOPOLOGY) - {"GLY"}
    n, ca, c = np.array([1.458, 0, 0]), np.zeros(3), np.array([-0.55, 1.42, 0])
    phe = sidechains.place(lib["PHE"][0], n, ca, c)
    names = sidechains.TOPOLOGY["PHE"][0]
    assert abs(np.linalg.norm(phe[names.index("CB")] - ca) - 1.53) < 0.05
    ring = phe[[names.index(x) for x in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]]
    assert np.linalg.svd(ring - ring.mean(0))[1][-1] < 0.1               # planar ring


def test_thread_pays_for_its_bridge_atoms(rungs):
    m2, m3, info3 = rungs[1], rungs[2], rungs[5]
    assert info3["added"] > 0 and info3["removed_for_bridges"] == info3["added"]
    assert m3.n == m2.n - info3["pruned"]                             # no net mass from the bridges


@pytest.fixture(scope="module")
def sym_rungs():
    m1 = ladder.gas(DecoyParams(seed=3, **SMALL), symmetry="C3")
    m2, i2 = ladder.build_polymer(m1, steps=300)
    m3, i3 = ladder.thread_chain(m2, relax=300)
    return m1, m2, m3, i2, i3


def test_symmetric_gas_is_one_unit_with_operators(sym_rungs):
    m1 = sym_rungs[0]
    plain = ladder.gas(DecoyParams(seed=3, **SMALL))
    assert m1.symmetry["group"] == "C3" and len(m1.symmetry["ops"]) == 3
    assert abs(3 * m1.n - plain.n) <= 3                                # the unit carries a third of the mass
    full = expand(m1)
    assert full.n == 3 * m1.n and len(np.unique((full.resseq - 1) // 9999)) == 3   # one chain per copy


def test_symmetric_steps_relax_against_their_images(sym_rungs):
    _, _, m3, i2, i3 = sym_rungs
    for info in (i2, i3):
        assert info["image_contacts"] > 0 and info["image_violations_gt_0.2A"] == 0
    assert m3.symmetry["group"] == "C3" and pieces(m3) == 1           # each copy is one chain


def test_symmetry_survives_files_and_mirroring(sym_rungs, tmp_path):
    m3 = sym_rungs[2]
    back = read(write(tmp_path / "s.pdb", m3))
    assert back.symmetry["group"] == "C3"
    assert np.allclose(back.symmetry["ops"], m3.symmetry["ops"], atol=1e-6)
    assert np.allclose(back.symmetry["centre"], m3.symmetry["centre"], atol=1e-3)
    assert "REMARK 350   BIOMT1   3" in (tmp_path / "s.pdb").read_text()
    mm = ladder.mirror(m3)
    assert np.allclose(np.linalg.det(mm.symmetry["ops"]), 1.0)          # still proper rotations
    a, b = expand(m3).pos, expand(mm).pos                               # the mirror of the whole assembly
    a = a.copy()
    a[:, 2] = m3.box - a[:, 2]
    assert np.allclose(a, b, atol=1e-6)
