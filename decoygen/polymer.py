"""Covalent pseudo-polymer: a branched, self-avoiding tree of heavy atoms (milestone 2).

The v0 Poisson gas has no short-range order: atoms overlap, and the 1.2-1.6 A bond and
2.4-2.5 A angle peaks of real protein are missing. Here the gas is only the starting
point:

1. Topology: a degree-capped minimum spanning forest over the gas (Prim on a
   k-nearest-neighbour graph). Attaching to an atom that already has two bonds is
   cost-weighted by `branch_weight`, tuned so the degree distribution matches protein
   (about 29% one bond, 37% two, 35% three). The result is a main chain with side
   branches, but no residues.
2. Elements are re-drawn from the reference P(element | degree), so oxygens are mostly
   terminal, as in carbonyls and hydroxyls.
3. Relaxation (Adam on positions), with graph distance g along the bonds:
     g = 1   harmonic to per-bond targets drawn from the reference distribution for
             the element pair
     g = 2   the same for 1-3 distances (bond angles)
     g = 3   one-sided lower bound (torsions are free)
     g >= 4  one-sided steric lower bound per element pair, ramped in
   plus a smooth penalty on the distance outside the envelope.

Torsions are left free and symmetric, so the polymer has no handedness, secondary
structure or residue repeat: it is protein-like below ~3 A and noise-like above.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from . import atoms

REFERENCE_FILE = Path(__file__).parent / "data" / "protein_reference.json"
ELEMENTS = atoms.ELEMENTS
MAX_DEGREE = 3
# bonds per heavy atom in protein (reference): O is terminal in 98%, S has at most 2, C and N at most 3
ELEMENT_MAX_DEGREE = {"C": 3, "N": 3, "O": 1, "S": 2}
N_QUANTILES = 101


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


PAIR_KEYS = sorted({pair_key(a, b) for a in ELEMENTS for b in ELEMENTS})


# ---------------------------------------------------------------- graph helpers

def bond_cutoff(a: str, b: str) -> float:
    if a == b == "S":
        return 2.2            # disulfide 2.03 A
    if "S" in (a, b):
        return 1.95           # C-S 1.81 A
    return 1.9


def graph_shells(n: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atom pairs (i < j) at shortest-path distance 1, 2 and 3 along the bonds."""
    a = sparse.coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.int32).tocsr()
    eye = sparse.identity(n, dtype=np.int32, format="csr")
    seen = (a + eye) > 0
    shells = [a]
    frontier = a
    for _ in range(2):
        frontier = ((frontier @ a) > 0).astype(np.int32)
        new = (frontier > seen).astype(np.int32)
        seen = (seen + new) > 0
        shells.append(new)
    out = []
    for s in shells:
        s = sparse.triu(s, k=1).tocoo()
        out.append(np.stack([s.row, s.col], 1).astype(np.int64))
    return out[0], out[1], out[2]


# ---------------------------------------------------------------- reference statistics

def _heavy_atoms(path: Path):
    import gemmi

    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    els, xyz = [], []
    for ch in st[0]:
        for r in ch:
            for at in r:
                if at.element.name in ELEMENTS:
                    els.append(at.element.name)
                    xyz.append(at.pos.tolist())
    return np.array(xyz), np.array(els)


def protein_reference(paths) -> dict:
    """Short-range geometry of real proteins, by element pair and bond-graph distance."""
    cats = {"bond": {}, "angle": {}, "torsion": {}, "nonbonded": {}}
    degree_counts = np.zeros((MAX_DEGREE + 2, len(ELEMENTS)), dtype=np.int64)
    pair_hist = np.zeros(80)
    n_total = 0
    for path in paths:
        xyz, els = _heavy_atoms(path)
        n = len(xyz)
        tree = cKDTree(xyz)
        close = tree.query_pairs(2.2, output_type="ndarray")
        d = np.linalg.norm(xyz[close[:, 0]] - xyz[close[:, 1]], axis=1)
        cut = np.array([bond_cutoff(els[i], els[j]) for i, j in close])
        edges = close[d < cut]
        deg = np.bincount(edges.ravel(), minlength=n)
        for k, e in enumerate(ELEMENTS):
            degree_counts[:, k] += np.bincount(np.minimum(deg[els == e], MAX_DEGREE + 1),
                                               minlength=MAX_DEGREE + 2)
        p12, p13, p14 = graph_shells(n, edges)
        near = tree.query_pairs(4.5, output_type="ndarray")
        near_keys = near[:, 0] * n + near[:, 1]
        bonded_keys = np.concatenate([p[:, 0] * n + p[:, 1] for p in (p12, p13, p14)])
        pnb = near[~np.isin(near_keys, bonded_keys)]
        for cat, pairs in (("bond", p12), ("angle", p13), ("torsion", p14), ("nonbonded", pnb)):
            dist = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
            keys = np.array([pair_key(els[i], els[j]) for i, j in pairs])
            for key in PAIR_KEYS:
                cats[cat].setdefault(key, []).extend(dist[keys == key].tolist())
        allp = tree.query_pairs(8.0, output_type="ndarray")
        dist = np.linalg.norm(xyz[allp[:, 0]] - xyz[allp[:, 1]], axis=1)
        pair_hist += np.histogram(dist, bins=80, range=(0, 8))[0]
        n_total += n

    q = np.linspace(0, 1, N_QUANTILES)
    ref = {"sources": [Path(p).stem for p in paths], "n_atoms": n_total, "quantiles": {}, "counts": {}}
    for cat, by_key in cats.items():
        ref["quantiles"][cat], ref["counts"][cat] = {}, {}
        for key, v in by_key.items():
            ref["counts"][cat][key] = len(v)
            if len(v) >= 20:
                ref["quantiles"][cat][key] = np.quantile(v, q).round(4).tolist()
        allv = np.concatenate([np.asarray(v) for v in by_key.values()])
        ref["quantiles"][cat]["*"] = np.quantile(allv, q).round(4).tolist()
    ref["degree_by_element"] = {"elements": list(ELEMENTS), "degrees": list(range(MAX_DEGREE + 2)),
                                "counts": degree_counts.tolist()}
    r = np.arange(80) * 0.1 + 0.05
    ref["pair_density"] = {"r": r.round(3).tolist(),
                           "density": (2 * pair_hist / n_total / (4 * np.pi * r**2 * 0.1)).round(5).tolist()}
    return ref


def load_reference(path: Path | None = None) -> dict:
    return json.loads(Path(path or REFERENCE_FILE).read_text())


def degree_fractions(ref: dict) -> np.ndarray:
    """Fraction of atoms with 0..MAX_DEGREE bonds (degrees above the cap folded in)."""
    c = np.array(ref["degree_by_element"]["counts"]).sum(1).astype(float)
    c[MAX_DEGREE] += c[MAX_DEGREE + 1:].sum()
    c = c[:MAX_DEGREE + 1]
    return c / c.sum()


# ---------------------------------------------------------------- topology

def spanning_forest(pos: np.ndarray, *, branch_weight: float = 1.0, k: int = 16,
                    max_degree: int | np.ndarray = MAX_DEGREE, seed: int = 0) -> np.ndarray:
    """Degree-capped minimum spanning forest (Prim). Returns edges (E, 2).

    Edge cost is length, times branch_weight when it would give the attached-to atom its
    third bond; branch_weight < 1 makes branch points (and leaves) more common.
    max_degree is one cap for all atoms or a per-atom array (see element_caps).
    """
    n = len(pos)
    cap = np.broadcast_to(np.asarray(max_degree), (n,))
    rng = np.random.default_rng(seed)
    dist, nbr = cKDTree(pos).query(pos, k=min(k + 1, n))
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    deg = np.zeros(n, dtype=np.int64)
    in_tree = np.zeros(n, dtype=bool)
    edges, heap = [], []

    def cost(src, d):
        return d * (branch_weight if deg[src] == 2 else 1.0)

    def push(i):
        for d, j in zip(dist[i], nbr[i]):
            if not in_tree[j]:
                heapq.heappush(heap, (cost(i, d), d, int(deg[i]), i, int(j)))

    def add(i):
        in_tree[i] = True
        if deg[i] < cap[i]:
            push(i)

    order = rng.permutation(n)
    next_root = 0
    while True:
        while heap:
            _, _, deg_at_push, i, j = heapq.heappop(heap)
            if in_tree[j] or deg[i] >= cap[i] or deg[i] != deg_at_push:
                continue                         # stale: i was re-pushed when its degree changed
            edges.append((i, j))
            deg[i] += 1
            deg[j] += 1
            add(j)
            if deg[i] < cap[i]:
                push(i)                          # i's remaining edges at their new cost
        while next_root < n and in_tree[order[next_root]]:
            next_root += 1
        if next_root == n:
            break
        add(int(order[next_root]))
    return np.array(edges, dtype=np.int64).reshape(-1, 2)


def element_caps(els: np.ndarray) -> np.ndarray:
    """Per-atom bond caps from ELEMENT_MAX_DEGREE, for re-wiring with fixed elements."""
    return np.array([ELEMENT_MAX_DEGREE[ELEMENTS[e]] for e in els], dtype=np.int64)


def fit_branch_weight(pos: np.ndarray, target: np.ndarray, seed: int = 0,
                      weights=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.8, 0.7)) -> tuple[float, np.ndarray]:
    """branch_weight whose forest's degree distribution is closest (L1) to `target`."""
    best = None
    for w in weights:
        e = spanning_forest(pos, branch_weight=w, seed=seed)
        frac = np.bincount(np.bincount(e.ravel(), minlength=len(pos)), minlength=MAX_DEGREE + 1)[:MAX_DEGREE + 1]
        frac = frac / len(pos)
        err = np.abs(frac - target).sum()
        if best is None or err < best[0]:
            best = (err, w, e)
    return best[1], best[2]


def assign_elements(degree: np.ndarray, ref: dict, rng: np.random.Generator) -> np.ndarray:
    """Element indices drawn from the reference P(element | degree)."""
    counts = np.array(ref["degree_by_element"]["counts"], dtype=float)
    out = np.empty(len(degree), dtype=np.int64)
    for d in range(MAX_DEGREE + 1):
        sel = degree == d
        p = counts[max(d, 1)]                    # isolated atoms: as terminal atoms
        out[sel] = rng.choice(len(ELEMENTS), size=int(sel.sum()), p=p / p.sum())
    return out


# ---------------------------------------------------------------- relaxation

TARGET_TAIL = 0.02          # targets come from the central 96%: the reference has a few junk contacts


def _sample_targets(pairs: np.ndarray, els: np.ndarray, table: dict, rng) -> np.ndarray:
    u = TARGET_TAIL + (1 - 2 * TARGET_TAIL) * rng.random(len(pairs))
    q = np.linspace(0, 1, N_QUANTILES)
    out = np.empty(len(pairs))
    keys = np.array([pair_key(ELEMENTS[els[i]], ELEMENTS[els[j]]) for i, j in pairs])
    for key in np.unique(keys):
        sel = keys == key
        out[sel] = np.interp(u[sel], q, table.get(key, table["*"]))
    return out


def _lower_bounds(table: dict, quantile_index: int) -> torch.Tensor:
    """(n_el, n_el) lower bound per element pair from a reference quantile."""
    lb = torch.zeros(len(ELEMENTS), len(ELEMENTS))
    for a, ea in enumerate(ELEMENTS):
        for b, eb in enumerate(ELEMENTS):
            lb[a, b] = table.get(pair_key(ea, eb), table["*"])[quantile_index]
    return lb


class Envelope:
    """Smooth confinement: squared distance (A) outside the mask, trilinearly interpolated."""

    def __init__(self, mask: np.ndarray, apix: float, device):
        self.apix = apix
        out = ndimage.distance_transform_edt(~mask) * apix
        self.field = torch.as_tensor(out, dtype=torch.float32, device=device)[None, None]
        self.n = mask.shape[0]

    def outside(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance outside the envelope for each position (0 inside)."""
        g = pos / self.apix / (self.n - 1) * 2 - 1           # [z, y, x] -> grid_sample's [x, y, z]
        g = g.flip(-1).view(1, -1, 1, 1, 3)
        return torch.nn.functional.grid_sample(self.field, g, align_corners=True).view(-1)


def _carry_targets(new_keys: np.ndarray, old_keys: np.ndarray, old_vals: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Keep the previous target for pairs that already had one."""
    if len(old_keys) == 0 or len(new_keys) == 0:
        return vals
    order = np.argsort(old_keys)
    ko, vo = old_keys[order], old_vals[order]
    idx = np.clip(np.searchsorted(ko, new_keys), 0, len(ko) - 1)
    hit = ko[idx] == new_keys
    vals = vals.copy()
    vals[hit] = vo[idx[hit]]
    return vals


class Restraints:
    """Covalent geometry and sterics of a bonded forest, by bond-graph distance g:

    g = 1, 2   harmonic to per-pair targets drawn from the reference (bond, 1-3 distances)
    g = 3      one-sided lower bound (torsions free)
    g >= 4     one-sided steric lower bound per element pair (call update_neighbours)

    `previous` (a Restraints on the same atoms) carries over targets of pairs present in
    both, so re-wiring changes only the new bonds and angles.
    """

    def __init__(self, els: np.ndarray, edges: np.ndarray, ref: dict, device, *,
                 nonbonded_quantile: float = 0.01, seed: int = 0, previous: Restraints | None = None):
        n = len(els)
        self.n, self.edges = n, edges
        rng = np.random.default_rng(seed)
        p12, p13, p14 = graph_shells(n, edges)
        self.keys12, self.keys13 = p12[:, 0] * n + p12[:, 1], p13[:, 0] * n + p13[:, 1]
        b0 = _sample_targets(p12, els, ref["quantiles"]["bond"], rng)
        a0 = _sample_targets(p13, els, ref["quantiles"]["angle"], rng)
        if previous is not None:
            b0 = _carry_targets(self.keys12, previous.keys12, previous.b0_np, b0)
            a0 = _carry_targets(self.keys13, previous.keys13, previous.a0_np, a0)
        self.b0_np, self.a0_np = b0, a0
        self.b0 = torch.as_tensor(b0, dtype=torch.float32, device=device)
        self.a0 = torch.as_tensor(a0, dtype=torch.float32, device=device)
        qi = round(nonbonded_quantile * (N_QUANTILES - 1))
        self.lb14 = _lower_bounds(ref["quantiles"]["torsion"], qi).to(device)
        self.lbnb = _lower_bounds(ref["quantiles"]["nonbonded"], qi).to(device)
        self.el = torch.as_tensor(els, device=device)
        self.t12, self.t13, self.t14 = (torch.as_tensor(p, device=device) for p in (p12, p13, p14))
        self.excluded = torch.sort(torch.cat([t[:, 0] * n + t[:, 1] for t in (self.t12, self.t13, self.t14)])).values
        self.reach = float(self.lbnb.max()) + 0.5
        self.pnb = torch.zeros((0, 2), dtype=torch.long, device=device)

    @staticmethod
    def dist(x, p):
        return (x[p[:, 0]] - x[p[:, 1]]).norm(dim=1)

    @torch.no_grad()
    def update_neighbours(self, x: torch.Tensor) -> None:
        n, cand = self.n, []
        for s in range(0, n, 4096):                       # chunked to bound cdist memory
            d = torch.cdist(x[s:s + 4096], x, compute_mode="donot_use_mm_for_euclid_dist")
            i, j = torch.nonzero(d < self.reach, as_tuple=True)
            i = i + s
            keep = i < j
            cand.append(torch.stack([i[keep], j[keep]], 1))
        cand = torch.cat(cand)
        if len(self.excluded):
            keys = cand[:, 0] * n + cand[:, 1]
            hit = torch.searchsorted(self.excluded, keys).clamp(max=len(self.excluded) - 1)
            cand = cand[self.excluded[hit] != keys]
        self.pnb = cand

    def terms(self, x: torch.Tensor, steric_scale: float = 1.0) -> dict:
        el, d = self.el, self.dist
        return {
            "bond": ((d(x, self.t12) - self.b0) ** 2).sum(),
            "angle": ((d(x, self.t13) - self.a0) ** 2).sum(),
            "torsion": (torch.relu(steric_scale * self.lb14[el[self.t14[:, 0]], el[self.t14[:, 1]]]
                                   - d(x, self.t14)) ** 2).sum(),
            "steric": (torch.relu(steric_scale * self.lbnb[el[self.pnb[:, 0]], el[self.pnb[:, 1]]]
                                  - d(x, self.pnb)) ** 2).sum(),
        }

    @torch.no_grad()
    def report(self, x: torch.Tensor) -> dict:
        self.update_neighbours(x)
        viol = self.lbnb[self.el[self.pnb[:, 0]], self.el[self.pnb[:, 1]]] - self.dist(x, self.pnb)
        return {"n_bonds": len(self.t12), "n_angles": len(self.t13), "n_torsions": len(self.t14),
                "bond_rmsd_A": float(((self.dist(x, self.t12) - self.b0) ** 2).mean().sqrt()),
                "angle_rmsd_A": float(((self.dist(x, self.t13) - self.a0) ** 2).mean().sqrt()),
                "steric_violations_gt_0.2A": int((viol > 0.2).sum()),
                "steric_violation_max_A": float(viol.max()) if len(viol) else 0.0}


def relax(pos: torch.Tensor, els: np.ndarray, edges: np.ndarray, ref: dict, *,
          envelope: Envelope | None = None, steps: int = 3000, lr: float = 0.02,
          nonbonded_quantile: float = 0.01, ramp: float = 0.6, seed: int = 0, log=None,
          restraints: Restraints | None = None, extra=None) -> tuple[torch.Tensor, dict]:
    """Restrained relaxation of a bonded forest. Returns (positions, info).

    restraints: use these instead of Restraints(els, edges, ref) (e.g. with targets overridden);
    extra: callable x -> dict of further energy terms (summed in, reported under their names).
    """
    n = len(pos)
    r = restraints or Restraints(els, edges, ref, pos.device, nonbonded_quantile=nonbonded_quantile, seed=seed)
    x = pos.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.05 * lr)
    ramp_steps = max(1, int(ramp * steps))
    history = []
    for step in range(steps):
        if step % 10 == 0:
            r.update_neighbours(x.detach())
        e = r.terms(x, steric_scale=min(1.0, 0.5 + 0.5 * step / ramp_steps))   # sterics ramp 0.5 -> 1
        e_env = (envelope.outside(x) ** 2).sum() if envelope is not None else x.new_zeros(())
        e_extra = extra(x) if extra is not None else {}
        loss = (sum(e.values()) + e_env + sum(e_extra.values(), x.new_zeros(()))) / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss), "bond": float(e["bond"] / max(len(r.t12), 1)),
                   "angle": float(e["angle"] / max(len(r.t13), 1)),
                   "steric": float((e["torsion"] + e["steric"]) / n), "envelope": float(e_env / n),
                   **{k: float(v / n) for k, v in e_extra.items()}}
            history.append(rec)
            if log:
                log(rec)
    x = x.detach()
    info = {**r.report(x),
            "outside_envelope_gt_0.5A": int((envelope.outside(x) > 0.5).sum()) if envelope is not None else 0,
            "history": history}
    return x, info


def build_polymer(pos0: torch.Tensor, ref: dict, *, envelope: Envelope | None = None, seed: int = 0,
                  steps: int = 3000, log=None) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """Gas positions -> (positions, element indices, edges, info)."""
    p = pos0.detach().cpu().numpy().astype(np.float64)
    target = degree_fractions(ref)
    w, edges = fit_branch_weight(p, target, seed=seed)
    degree = np.bincount(edges.ravel(), minlength=len(p))
    els = assign_elements(degree, ref, np.random.default_rng(seed))
    pos, info = relax(pos0, els, edges, ref, envelope=envelope, steps=steps, seed=seed, log=log)
    frac = np.bincount(degree, minlength=MAX_DEGREE + 1)[:MAX_DEGREE + 1] / len(p)
    info.update(branch_weight=w, degree_fractions=frac.tolist(), degree_target=target.tolist(),
                n_components=len(p) - len(edges),
                composition={e: float((els == k).mean()) for k, e in enumerate(ELEMENTS)})
    return pos, torch.as_tensor(els, device=pos.device), edges, info


# ---------------------------------------------------------------- soft bonds

def _median_table(table: dict) -> torch.Tensor:
    return _lower_bounds(table, (N_QUANTILES - 1) // 2)


class SoftChemistry:
    """Covalent geometry without a fixed bond list: every atom pair within reach is classified
    by distance, with smooth (sigmoid) boundaries of width `width`, as

        bonded     r < bond_cutoff (1.9 A, more with S)   harmonic to the element-pair median bond
        1-3        bond_cutoff <= r < angle_cutoff        harmonic to the median 1-3 distance
        free       beyond                                 steric lower bound per element pair

    and two valence counts per atom hold the chemistry together: its soft bond count must
    equal its target degree (`degree`, normally the degree it had in the bonded polymer, so
    P(element | degree) is kept), and its soft 1-3 count must be at least the number of
    two-bond paths through its soft partners, so every pair of bonds comes with a bond angle
    (one-sided: close 1-4 and O/N contacts also fall in the 1-3 class). Bonds can then break and form wherever the pull on the
    atoms wins over the count penalty. As width -> 0 the classes become hard and
    `harden()` reads off an ordinary bond list for `Restraints`.
    """

    def __init__(self, els: np.ndarray, degree: np.ndarray, ref: dict, device, *,
                 angle_cutoff: float = 2.65, nonbonded_quantile: float = 0.01, count_weight: float = 1.0):
        self.n = len(els)
        self.el = torch.as_tensor(els, device=device)
        self.degree = torch.as_tensor(degree, dtype=torch.float32, device=device)
        self.caps = torch.as_tensor(element_caps(els), device=device)
        self.b0 = _median_table(ref["quantiles"]["bond"]).to(device)
        self.a0 = _median_table(ref["quantiles"]["angle"]).to(device)
        qi = round(nonbonded_quantile * (N_QUANTILES - 1))
        self.lbnb = _lower_bounds(ref["quantiles"]["nonbonded"], qi).to(device)
        self.rb = torch.tensor([[bond_cutoff(a, b) for b in ELEMENTS] for a in ELEMENTS], device=device)
        self.ra = angle_cutoff
        self.count_weight = count_weight
        self.reach = max(float(self.lbnb.max()), angle_cutoff) + 0.5
        self.pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

    @torch.no_grad()
    def update_neighbours(self, x: torch.Tensor) -> None:
        cand = []
        for s in range(0, self.n, 4096):
            d = torch.cdist(x[s:s + 4096], x, compute_mode="donot_use_mm_for_euclid_dist")
            i, j = torch.nonzero(d < self.reach, as_tuple=True)
            i = i + s
            keep = i < j
            cand.append(torch.stack([i[keep], j[keep]], 1))
        self.pairs = torch.cat(cand)

    def classify(self, x: torch.Tensor, width: float):
        p, el = self.pairs, self.el
        ea, eb = el[p[:, 0]], el[p[:, 1]]
        r = (x[p[:, 0]] - x[p[:, 1]]).norm(dim=1)
        wb = torch.sigmoid((self.rb[ea, eb] - r) / width)
        wa = (1 - wb) * torch.sigmoid((self.ra - r) / width)
        return r, wb, wa, 1 - wb - wa, ea, eb

    def terms(self, x: torch.Tensor, width: float) -> dict:
        p = self.pairs
        r, wb, wa, wf, ea, eb = self.classify(x, width)
        nb = x.new_zeros(self.n).index_add(0, p[:, 0], wb).index_add(0, p[:, 1], wb)
        na = x.new_zeros(self.n).index_add(0, p[:, 0], wa).index_add(0, p[:, 1], wa)
        # two-bond paths through i's partners j: sum_j wb_ij (nb_j - wb_ij)
        paths = (x.new_zeros(self.n).index_add(0, p[:, 0], wb * (nb[p[:, 1]] - wb))
                 .index_add(0, p[:, 1], wb * (nb[p[:, 0]] - wb)))
        return {
            "bond": (wb * (r - self.b0[ea, eb]) ** 2).sum(),
            "angle": (wa * (r - self.a0[ea, eb]) ** 2).sum(),
            "steric": (wf * torch.relu(self.lbnb[ea, eb] - r) ** 2).sum(),
            "valence": self.count_weight * ((nb - self.degree) ** 2).sum(),
            "angle_count": self.count_weight * (torch.relu(paths - na) ** 2).sum(),
        }

    @torch.no_grad()
    def bonds(self, x: torch.Tensor) -> np.ndarray:
        """Hard bond list: pairs inside the bond cutoff, shortest first, within element caps."""
        self.update_neighbours(x)
        r, wb, *_ = self.classify(x, 1e-3)
        sel = wb > 0.5
        p, r = self.pairs[sel].cpu().numpy(), r[sel].cpu().numpy()
        caps, deg, out = self.caps.cpu().numpy(), np.zeros(self.n, dtype=int), []
        for k in np.argsort(r):
            i, j = p[k]
            if deg[i] < caps[i] and deg[j] < caps[j]:
                deg[i] += 1
                deg[j] += 1
                out.append((i, j))
        return np.array(out, dtype=np.int64).reshape(-1, 2)


def topology_stats(n: int, edges: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """Degree fractions, cycles and (optionally) bonds not in `reference`."""
    deg = np.bincount(edges.ravel(), minlength=n)
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_comp = csgraph.connected_components(adj, directed=False)[0]
    out = {"n_bonds": len(edges), "degree_fractions": [float((deg == d).mean()) for d in range(5)],
           "cycles": int(len(edges) - n + n_comp), "components": int(n_comp)}
    if reference is not None:
        a = {tuple(e) for e in np.sort(edges, axis=1)}
        b = {tuple(e) for e in np.sort(reference, axis=1)}
        out["bonds_new"], out["bonds_lost"] = len(a - b), len(b - a)
    return out
