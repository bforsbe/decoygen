"""Real side chains (rung 4 -> rung 5 of the ladder).

1. Thinning. A peptide made from the threaded chain has far more residues than a protein of
   its mass (about 4.3 atoms per residue against 7.7). Residues are excised in passes (at most
   `fraction` of them per pass, never two neighbours, the most bent first: smallest
   CA(i-1)-CA(i+1) distance). An excised residue's atoms stay where they are and join the side
   group of its lighter neighbour; its two neighbours are bonded to each other. The new bonds,
   angles and peptide terms are switched on gradually (targets slide from the current distances
   to the ideal ones over `stages` relaxations), so the backbone draws together around atoms
   that never move away: no shrinking and re-expanding of the volume.
2. Sequence. A random sequence with natural amino-acid frequencies is drawn and matched to the
   residues by rank of side-group size (largest group gets the largest side chain), so the
   composition is protein-like and the local mass follows the atoms being replaced.
3. Placement. Each side group is replaced by a real side-chain conformation from the library
   (superposed on N, CA, C), the one that best covers the group's atoms and avoids the backbone;
   then everything relaxes with peptide restraints and the library's own side-chain geometry.

The library (data/sidechain_library.json) holds up to 100 conformations per amino acid from
the reference proteins of protein_reference.json; build_library() rebuilds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LIBRARY_FILE = Path(__file__).parent / "data" / "sidechain_library.json"
BACKBONE = ("N", "CA", "C", "O")

# side-chain heavy atoms and bonds (CA-CB implied for all but GLY; PRO also CD-N)
TOPOLOGY = {
    "GLY": ([], []),
    "ALA": (["CB"], []),
    "SER": (["CB", "OG"], [("CB", "OG")]),
    "CYS": (["CB", "SG"], [("CB", "SG")]),
    "VAL": (["CB", "CG1", "CG2"], [("CB", "CG1"), ("CB", "CG2")]),
    "THR": (["CB", "OG1", "CG2"], [("CB", "OG1"), ("CB", "CG2")]),
    "PRO": (["CB", "CG", "CD"], [("CB", "CG"), ("CG", "CD"), ("CD", "N")]),
    "ASP": (["CB", "CG", "OD1", "OD2"], [("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")]),
    "ASN": (["CB", "CG", "OD1", "ND2"], [("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")]),
    "ILE": (["CB", "CG1", "CG2", "CD1"], [("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")]),
    "LEU": (["CB", "CG", "CD1", "CD2"], [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")]),
    "MET": (["CB", "CG", "SD", "CE"], [("CB", "CG"), ("CG", "SD"), ("SD", "CE")]),
    "GLU": (["CB", "CG", "CD", "OE1", "OE2"], [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")]),
    "GLN": (["CB", "CG", "CD", "OE1", "NE2"], [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")]),
    "LYS": (["CB", "CG", "CD", "CE", "NZ"], [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")]),
    "HIS": (["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
            [("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")]),
    "PHE": (["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            [("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"),
             ("CD2", "CG")]),
    "TYR": (["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
            [("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"),
             ("CD2", "CG"), ("CZ", "OH")]),
    "TRP": (["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            [("CB", "CG"), ("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CG"),
             ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"), ("CZ3", "CE3"), ("CE3", "CD2")]),
    "ARG": (["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
            [("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")]),
}
# natural frequencies (UniProtKB/Swiss-Prot, %)
FREQUENCY = {"ALA": 8.25, "ARG": 5.53, "ASN": 4.06, "ASP": 5.45, "CYS": 1.37, "GLN": 3.93, "GLU": 6.75,
             "GLY": 7.07, "HIS": 2.27, "ILE": 5.96, "LEU": 9.66, "LYS": 5.84, "MET": 2.42, "PHE": 3.86,
             "PRO": 4.70, "SER": 6.56, "THR": 5.34, "TRP": 1.08, "TYR": 2.92, "VAL": 6.87}
SIZE = {aa: len(TOPOLOGY[aa][0]) for aa in TOPOLOGY}
MEAN_SIZE = sum(FREQUENCY[a] * SIZE[a] for a in FREQUENCY) / sum(FREQUENCY.values())


def element(name: str) -> str:
    return name[0]


def local_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rows: orthonormal axes of the residue frame (x along CA->C, z normal to N-CA-C)."""
    e1 = (c - ca) / np.linalg.norm(c - ca)
    e3 = np.cross(e1, n - ca)
    e3 /= np.linalg.norm(e3)
    return np.stack([e1, np.cross(e3, e1), e3])


def build_library(paths, max_per_aa: int = 100, seed: int = 0) -> dict:
    """Side-chain conformations of the given structures, in each residue's N-CA-C frame."""
    import gemmi

    found = {aa: [] for aa in TOPOLOGY if aa != "GLY"}
    for path in paths:
        st = gemmi.read_structure(str(path))
        st.remove_hydrogens()
        st.remove_alternative_conformations()
        st.remove_ligands_and_waters()
        for ch in st[0]:
            for r in ch:
                if r.name not in found:
                    continue
                names = ["N", "CA", "C", *TOPOLOGY[r.name][0]]
                atoms = {a.name: np.array(a.pos.tolist()) for a in r}
                if not all(nm in atoms for nm in names):
                    continue
                rot = local_frame(atoms["N"], atoms["CA"], atoms["C"])
                found[r.name].append([((atoms[nm] - atoms["CA"]) @ rot.T).round(3).tolist()
                                      for nm in TOPOLOGY[r.name][0]])
    rng = np.random.default_rng(seed)
    lib = {}
    for aa, confs in found.items():
        if len(confs) > max_per_aa:
            confs = [confs[k] for k in sorted(rng.choice(len(confs), max_per_aa, replace=False))]
        lib[aa] = confs
    return {"sources": [Path(p).stem for p in paths], "atoms": {aa: TOPOLOGY[aa][0] for aa in lib},
            "conformations": lib}


def load_library(path: Path | None = None) -> dict:
    lib = json.loads(Path(path or LIBRARY_FILE).read_text())
    return {aa: np.array(c, dtype=np.float64) for aa, c in lib["conformations"].items()}


def place(conf: np.ndarray, n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Library conformation(s) (..., m, 3) in local coordinates -> positions on this backbone."""
    return ca + conf @ local_frame(n, ca, c)


def draw_sequence(n_res: int, rng: np.random.Generator) -> list[str]:
    aas = sorted(FREQUENCY)
    p = np.array([FREQUENCY[a] for a in aas])
    return list(rng.choice(aas, size=n_res, p=p / p.sum()))


# ---------------------------------------------------------------- the rung-5 step

CAPS = {"C": 3, "N": 3, "O": 1, "S": 2}
CARBON = 0                                                   # atoms.ELEMENTS.index("C")


class Chain:
    """A peptide being thinned: atom indices never change; residues are dropped from the lists."""

    def __init__(self, pos, els, edges, resseq, names):
        from . import atoms

        self.pos, self.els = np.asarray(pos, dtype=np.float64).copy(), np.asarray(els).copy()
        self.sym = [atoms.ELEMENTS[e] for e in self.els]
        self.edges = set()
        self.deg = np.zeros(len(self.pos), dtype=np.int64)
        for i, j in edges:
            self.link(int(i), int(j))
        order = sorted({int(r) for r in resseq})
        at = {(int(r), nm): i for i, (r, nm) in enumerate(zip(resseq, names))}
        self.n = [at[(r, "N")] for r in order]
        self.ca = [at[(r, "CA")] for r in order]
        self.c = [at[(r, "C")] for r in order]
        self.o = [at[(r, "O")] for r in order]
        self.cb = [at.get((r, "CB"), -1) for r in order]
        slot = {r: k for k, r in enumerate(order)}
        self.group = [[] for _ in order]                     # side atoms (incl. CB) per residue
        for i, (r, nm) in enumerate(zip(resseq, names)):
            if nm not in BACKBONE:
                self.group[slot[int(r)]].append(i)

    @property
    def n_res(self) -> int:
        return len(self.n)

    def link(self, i: int, j: int) -> None:
        e = (min(i, j), max(i, j))
        if e not in self.edges:
            self.edges.add(e)
            self.deg[[i, j]] += 1

    def unlink(self, i: int, j: int) -> None:
        e = (min(i, j), max(i, j))
        if e in self.edges:
            self.edges.remove(e)
            self.deg[[i, j]] -= 1

    def pep(self) -> dict:
        return {"pos": self.pos, "els": self.els, "edges": np.array(sorted(self.edges), dtype=np.int64),
                **{k: np.array(getattr(self, k)) for k in ("n", "ca", "c", "o", "cb")}}

    def excise(self, k: int) -> tuple[int, bool]:
        """Give residue k's atoms to its lighter neighbour and join the neighbours.
        Returns (host residue index before removal, whether the host got a new CB)."""
        from .peptide import chiral_volume

        a, b = k - 1, k + 1
        host = a if len(self.group[a]) <= len(self.group[b]) else b
        if len(self.group[a]) == len(self.group[b]):
            d = [np.linalg.norm(self.pos[self.ca[h]] - self.pos[self.ca[k]]) for h in (a, b)]
            host = (a, b)[int(np.argmin(d))]
        self.unlink(self.c[a], self.n[k])
        self.unlink(self.c[k], self.n[b])
        self.link(self.c[a], self.n[b])
        moved = [self.n[k], self.ca[k], self.c[k], self.o[k], *self.group[k]]
        free = [i for i in moved if self.deg[i] < CAPS[self.sym[i]]]
        new_cb = self.cb[host] < 0
        if new_cb:                                           # host was glycine: its new CB, on the L side
            n_, ca_, c_ = (self.pos[x[host]] for x in (self.n, self.ca, self.c))
            cand = free or moved
            m = len(cand)
            vol = chiral_volume(np.repeat(n_[None], m, 0), np.repeat(ca_[None], m, 0), np.repeat(c_[None], m, 0),
                                self.pos[cand])
            dist = np.linalg.norm(self.pos[cand] - ca_, axis=1)
            pick = cand[int(np.argmin(np.where(vol > 0, dist, dist + 100)))]
            self.link(self.ca[host], pick)
            self.cb[host] = pick
            self.els[pick], self.sym[pick] = CARBON, "C"
        else:
            hosts = [i for i in self.group[host] if self.deg[i] < CAPS[self.sym[i]]] or self.group[host]
            cand = free or moved
            d = np.linalg.norm(self.pos[hosts][:, None] - self.pos[cand][None], axis=2)
            h, m = np.unravel_index(int(np.argmin(d)), d.shape)
            self.link(hosts[h], cand[m])
        self.group[host] += moved
        for lst in (self.n, self.ca, self.c, self.o, self.cb, self.group):
            del lst[k]
        return host, new_cb


def choose_excisions(ch: Chain, count: int) -> list[int]:
    """Up to `count` interior residues, no two adjacent, most bent first (CA(i-1)-CA(i+1))."""
    k = np.arange(1, ch.n_res - 1)
    span = np.linalg.norm(ch.pos[np.array(ch.ca)[k - 1]] - ch.pos[np.array(ch.ca)[k + 1]], axis=1)
    taken = np.zeros(ch.n_res, dtype=bool)
    out = []
    for j in k[np.argsort(span, kind="stable")]:
        if len(out) >= count:
            break
        if not (taken[j - 1] or taken[j] or taken[j + 1]):
            taken[j] = True
            out.append(int(j))
    return out


def fit_sequence(sizes: np.ndarray, n_atoms_side: int, rng: np.random.Generator, tries: int = 400) -> list[str]:
    """A natural-frequency sequence whose side-chain atoms total close to n_atoms_side, matched
    to residues by rank of their side-group size (ties broken at random)."""
    best, best_err = None, np.inf
    for _ in range(tries):
        seq = draw_sequence(len(sizes), rng)
        err = abs(sum(SIZE[a] for a in seq) - n_atoms_side)
        if err < best_err:
            best, best_err = seq, err
    by_size = sorted(best, key=lambda a: (SIZE[a], rng.random()))
    order = np.lexsort((rng.random(len(sizes)), sizes))
    out = [""] * len(sizes)
    for slot, aa in zip(order, by_size):
        out[slot] = aa
    return out


def _pair_keys(n: int, edges) -> tuple[set, set]:
    from .polymer import graph_shells

    p12, p13, _ = graph_shells(n, np.array(sorted(edges), dtype=np.int64).reshape(-1, 2))
    return set((p12[:, 0] * n + p12[:, 1]).tolist()), set((p13[:, 0] * n + p13[:, 1]).tolist())


def _key_dist(pos: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Distances of atom pairs given as keys i * n + j."""
    n = len(pos)
    return np.linalg.norm(pos[keys // n] - pos[keys % n], axis=1)


def thin(ch: Chain, n_target: int, *, fraction: float, steps: int, stages: int, envelope, ref, device,
         seed: int = 0, log=None) -> list[dict]:
    """Excise residues in passes of at most `fraction` until n_target remain; after each pass the
    new bonds, angles and peptide terms are switched on over `stages` relaxations."""
    import torch

    from . import peptide, polymer

    passes = []
    while ch.n_res > n_target:
        count = min(max(1, round(fraction * ch.n_res)), ch.n_res - n_target)
        old12, old13 = _pair_keys(len(ch.pos), ch.edges)
        chosen = choose_excisions(ch, count)
        spans = [float(np.linalg.norm(ch.pos[ch.ca[k - 1]] - ch.pos[ch.ca[k + 1]])) for k in chosen]
        junction_c, new_cb_ca = set(), set()
        for k in sorted(chosen, reverse=True):               # from the end, so lower indices stay valid
            junction_c.add(ch.c[k - 1])
            host, new_cb = ch.excise(k)
            if new_cb:
                new_cb_ca.add(ch.ca[host if host < k else host - 1])
        pep = ch.pep()
        lw, cw = np.ones(ch.n_res - 1), np.ones(ch.n_res)
        r, extra = peptide.restraints(pep, ref, device, seed=seed, link_weight=lw, chiral_weight=cw)
        links = np.flatnonzero(np.isin(pep["c"][:-1], list(junction_c)))
        cbs = np.flatnonzero(np.isin(pep["ca"], list(new_cb_ca)))
        new12 = np.flatnonzero(~np.isin(r.keys12, list(old12)))
        new13 = np.flatnonzero(~np.isin(r.keys13, list(old13)))
        b_end, a_end = r.b0_np[new12].copy(), r.a0_np[new13].copy()
        b_start, a_start = _key_dist(ch.pos, r.keys12[new12]), _key_dist(ch.pos, r.keys13[new13])
        x = torch.as_tensor(ch.pos, dtype=torch.float32, device=device)
        for stage in range(1, stages + 1):
            s = stage / stages
            lw[links], cw[cbs] = s, s
            r.b0_np[new12] = b_start + s * (b_end - b_start)
            r.a0_np[new13] = a_start + s * (a_end - a_start)
            r.b0 = torch.as_tensor(r.b0_np, dtype=torch.float32, device=device)
            r.a0 = torch.as_tensor(r.a0_np, dtype=torch.float32, device=device)
            x, info = polymer.relax(x, ch.els, pep["edges"], ref, envelope=envelope, steps=max(1, steps // stages),
                                    seed=seed, restraints=r, extra=extra, log=log)
        ch.pos = x.cpu().numpy().astype(np.float64)
        after = np.linalg.norm(ch.pos[pep["ca"][links]] - ch.pos[pep["ca"][links + 1]], axis=1)
        rec = {"excised": len(chosen), "residues_left": ch.n_res,
               "span_before_median_A": float(np.median(spans)), "span_before_max_A": float(np.max(spans)),
               "junction_CA_CA_after_median_A": float(np.median(after)),
               "junction_CA_CA_after_max_A": float(np.max(after)),
               "bond_rmsd_A": info["bond_rmsd_A"], "steric_violations_gt_0.2A": info["steric_violations_gt_0.2A"],
               **{f"after_{k}": v for k, v in peptide.report(pep, ch.pos).items()}}
        passes.append(rec)
        if log:
            log({"step": len(passes), **{k: v for k, v in rec.items() if isinstance(v, (int, float))}})
    return passes


def place_sidechains(ch: Chain, lib: dict, seq: list[str], clash_radius: float = 3.0) -> dict:
    """Replace every side group by the library conformation of its amino acid that best covers
    the group's atoms (symmetric mean nearest distance) with fewest backbone clashes."""
    from scipy.spatial import cKDTree

    from . import atoms

    bb = np.array([ch.n, ch.ca, ch.c, ch.o]).T                         # (n_res, 4)
    tree = cKDTree(ch.pos[bb.ravel()])
    owner = np.repeat(np.arange(ch.n_res), 4)
    pos, names, resnames, resseq, els, edges, rigid, template = [], [], [], [], [], [], [], []
    idx_n, idx_ca, idx_c, idx_o, idx_cb = [], [], [], [], []
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}

    def add(p, name, k, aa):
        pos.append(p)
        names.append(name)
        resnames.append(aa)
        resseq.append(k + 1)
        els.append(lookup[element(name)])
        return len(pos) - 1

    for k, aa in enumerate(seq):
        n_, ca_, c_ = (ch.pos[x[k]] for x in (ch.n, ch.ca, ch.c))
        here = {nm: add(ch.pos[x[k]], nm, k, aa) for nm, x in zip(BACKBONE, (ch.n, ch.ca, ch.c, ch.o))}
        idx_n.append(here["N"])
        idx_ca.append(here["CA"])
        idx_c.append(here["C"])
        idx_o.append(here["O"])
        edges += [(here["N"], here["CA"]), (here["CA"], here["C"]), (here["C"], here["O"])]
        if k:
            edges.append((idx_c[k - 1], here["N"]))
        if aa == "GLY":
            idx_cb.append(-1)
            continue
        cand = place(lib[aa], n_, ca_, c_)                              # (K, m, 3)
        g = ch.pos[ch.group[k]] if ch.group[k] else None
        if g is not None:
            d = np.linalg.norm(cand[:, :, None] - g[None, None], axis=3)
            fit = d.min(2).mean(1) + d.min(1).mean(1)
        else:
            fit = np.zeros(len(cand))
        hits = tree.query_ball_point(cand.reshape(-1, 3), clash_radius)
        near = np.array([sum(1 for h in hs if abs(owner[h] - k) > 1) for hs in hits]).reshape(cand.shape[:2])
        best = int(np.argmin(fit + near.sum(1)))
        side = TOPOLOGY[aa]
        for nm, p in zip(side[0], cand[best]):
            here[nm] = add(p, nm, k, aa)
        idx_cb.append(here["CB"])
        edges.append((here["CA"], here["CB"]))
        edges += [(here[a], here[b]) for a, b in side[1]]
        members = [here[nm] for nm in ("N", "CA", "C", *side[0])]
        template.append(members)
        if aa in ("HIS", "PHE", "TYR", "TRP", "PRO"):
            rigid.append([here[nm] for nm in ("CA", *side[0])])
    e = np.sort(np.array(edges, dtype=np.int64), axis=1)
    return {"pos": np.array(pos), "els": np.array(els, dtype=np.int64), "edges": np.unique(e, axis=0),
            "resseq": np.array(resseq), "names": names, "resnames": resnames,
            "n": np.array(idx_n), "ca": np.array(idx_ca), "c": np.array(idx_c), "o": np.array(idx_o),
            "cb": np.array(idx_cb), "template": template, "rigid": rigid}


def final_restraints(pep: dict, ref: dict, device, seed: int = 0):
    """Peptide restraints, with every within-residue bond and 1-3 distance involving side-chain
    atoms held at the library conformation's value, and ring residues (HIS PHE TYR TRP PRO) rigid."""
    import torch

    from . import peptide

    r, extra = peptide.restraints(pep, ref, device, seed=seed)
    pos = pep["pos"]
    table = []
    for members in pep["template"]:
        m = np.array(members)
        i, j = np.triu_indices(len(m), 1)
        a, b = np.minimum(m[i], m[j]), np.maximum(m[i], m[j])
        side = np.isin(a, m[3:]) | np.isin(b, m[3:])
        pr = np.stack([a[side], b[side]], 1)
        table.append((pr, np.linalg.norm(pos[pr[:, 0]] - pos[pr[:, 1]], axis=1)))
    peptide.set_targets(r, table)
    rp = [np.zeros((0, 2), dtype=np.int64)]
    for members in pep["rigid"]:                          # every pair within a ring residue
        m = np.array(members)
        i, j = np.triu_indices(len(m), 1)
        rp.append(np.stack([m[i], m[j]], 1))
    rp = np.concatenate(rp)
    rd = torch.as_tensor(np.linalg.norm(pos[rp[:, 0]] - pos[rp[:, 1]], axis=1), dtype=torch.float32, device=device)
    rp_t = torch.as_tensor(rp, device=device)

    def extra_all(x):
        out = extra(x)
        out["rings"] = (((x[rp_t[:, 0]] - x[rp_t[:, 1]]).norm(dim=1) - rd) ** 2).sum()
        return out

    return r, extra_all
