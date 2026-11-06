"""Peptide backbone (rung 3 -> rung 4 of the ladder).

The threaded chain's backbone atoms are read three at a time as N, CA, C of one residue.
Each carbonyl C gets an O in the plane of its CA and the next N (on the outer bisector),
and every residue's side atoms (whatever hung off its N, CA or C) are regrouped into one
branch rooted at CB on CA, because N and the carbonyl C have no free valence in a peptide.
Glycine = a residue with nothing hanging off it. CB is placed on the L side (the side atoms
are reflected through the N-CA-C plane where needed), and relaxation then holds:

  bonds and angles     Engh & Huber targets for N-CA, CA-C, C-N, C=O, CA-CB and their 1-3 distances
  trans, planar        CA(i)-CA(i+1) = 3.80 A, O(i)-CA(i+1) = 2.76 A and a torsion term on
                       omega = CA-C-N-CA (minimum at 180)
  L at CA              chiral volume (N-CA).((C-CA)x(CB-CA)) = +2.50 A^3 (6CVM: +2.50, all > 0)
  everything else      as polymer.relax: element-pair geometry, sterics, the input's envelope

Phi and psi are left as they fall (random coil comes at a later rung). The new O atoms are
paid for by trimming side-branch leaves, so the atom count (and map mass) stays that of the
input unless keep_atoms=False.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from . import atoms, polymer

# Engh & Huber (1991, 2001) targets
BOND = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "N+"): 1.329, ("C", "O"): 1.231, ("CA", "CB"): 1.530}
ANGLE_DEG = {("N", "CA", "C"): 111.0, ("CA", "C", "N+"): 116.2, ("C", "N+", "CA+"): 121.7,
             ("CA", "C", "O"): 120.1, ("O", "C", "N+"): 122.7, ("N", "CA", "CB"): 110.5,
             ("C", "CA", "CB"): 110.1}
CA_CA = 3.80              # trans peptide
O_CA_NEXT = 2.76
CHIRAL_VOLUME = 2.50      # L-amino acids
OMEGA_WEIGHT = 10.0       # A^2 per peptide: w (1 + cos omega), ~ w/2 (omega - 180)^2 near trans
GREEK = "BGDEZH"
E = {e: k for k, e in enumerate(atoms.ELEMENTS)}


def _third_side(a: float, b: float, angle_deg: float) -> float:
    return float(np.sqrt(a * a + b * b - 2 * a * b * np.cos(np.radians(angle_deg))))


def _bond(x: str, y: str) -> float:
    if x.endswith("+") and y.endswith("+"):                       # both in the next residue
        x, y = x[:-1], y[:-1]
    return BOND.get((x, y)) or BOND[(y, x)]


def chiral_volume(n, ca, c, cb):
    """(N-CA).((C-CA)x(CB-CA)) for (k, 3) arrays or tensors; > 0 for L."""
    if isinstance(n, torch.Tensor):
        return ((n - ca) * torch.linalg.cross(c - ca, cb - ca)).sum(-1)
    return np.einsum("ij,ij->i", n - ca, np.cross(c - ca, cb - ca))


def cos_dihedral(a, b, c, d):
    """cos of the dihedral a-b-c-d for (k, 3) tensors."""
    n1 = torch.linalg.cross(b - a, c - b)
    n2 = torch.linalg.cross(c - b, d - c)
    return (n1 * n2).sum(-1) / (n1.norm(dim=-1) * n2.norm(dim=-1) + 1e-8)


def dihedral(a, b, c, d) -> np.ndarray:
    """Dihedral angles (degrees) of (k, 3) arrays."""
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.einsum("ij,ij->i", b0, b1)[:, None] * b1
    w = b2 - np.einsum("ij,ij->i", b2, b1)[:, None] * b1
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def _reflect(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm)
    return p - 2 * ((p - a) @ nrm)[:, None] * nrm


def build(pos: np.ndarray, els: np.ndarray, edges: np.ndarray, backbone: np.ndarray, *,
          keep_atoms: bool = True, seed: int = 0) -> dict:
    """Topology and starting positions of the peptide. `backbone`: backbone atom indices in chain order.

    Returns dict with pos, els, edges, resseq, names, resnames and the index arrays
    n/ca/c/o (per residue) and cb (per residue, -1 for glycine), plus stats.
    """
    rng = np.random.default_rng(seed)
    n_res = len(backbone) // 3
    bb = np.asarray(backbone[:3 * n_res]).reshape(n_res, 3)          # N, CA, C per residue
    n_at = len(pos)
    role = np.full(n_at, -1)                                          # 0 N, 1 CA, 2 C; -1 side atom
    res = np.full(n_at, -1)
    for k in range(3):
        role[bb[:, k]], res[bb[:, k]] = k, np.arange(n_res)
    adj = [[] for _ in range(n_at)]
    for i, j in edges:
        adj[i].append(int(j))
        adj[j].append(int(i))

    # side atoms: parent towards the backbone, residue of the backbone atom they hang off
    parent = np.full(n_at, -1)
    queue = deque(int(a) for a in bb.ravel())
    seen = np.zeros(n_at, dtype=bool)
    seen[bb.ravel()] = True
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if not seen[j]:
                seen[j], parent[j], res[j] = True, i, res[i]
                queue.append(j)
    side = np.flatnonzero((role < 0) & seen)
    dropped_unreached = int(((role < 0) & ~seen).sum())

    # the new O atoms are paid for by side-branch leaves (random, repeatedly), so the count holds
    alive = np.zeros(n_at, dtype=bool)
    alive[bb.ravel()] = True
    alive[side] = True
    children = np.zeros(n_at, dtype=np.int64)
    np.add.at(children, parent[side], 1)
    trimmed = 0
    if keep_atoms:
        target = n_at - n_res
        while alive.sum() > target:
            leaves = np.flatnonzero(alive & (role < 0) & (children == 0))
            if len(leaves) == 0:
                break
            take = rng.choice(leaves, size=min(len(leaves), int(alive.sum() - target)), replace=False)
            alive[take] = False
            np.subtract.at(children, parent[take], 1)
            trimmed += len(take)

    # regroup each residue's side atoms into one branch on CB
    p = pos.astype(np.float64).copy()
    new_edges = [(bb[k, 0], bb[k, 1]) for k in range(n_res)] + [(bb[k, 1], bb[k, 2]) for k in range(n_res)]
    new_edges += [(bb[k, 2], bb[k + 1, 0]) for k in range(n_res - 1)]
    out_els = els.copy()
    out_els[bb[:, 0]] = E["N"]
    out_els[bb[:, 1:]] = E["C"]
    cb = np.full(n_res, -1)
    caps = {E["C"]: 3, E["N"]: 3, E["O"]: 1, E["S"]: 2}
    deg = np.zeros(n_at, dtype=np.int64)
    members_of = [[] for _ in range(n_res)]
    for i in side:
        if alive[i]:
            members_of[res[i]].append(i)
    detached = 0
    for k in range(n_res):
        mem = members_of[k]
        if not mem:
            continue
        mem_set = set(mem)
        roots = [i for i in mem if parent[i] not in mem_set]
        internal = [(i, parent[i]) for i in mem if parent[i] in mem_set]
        size = {r: 0 for r in roots}
        top = {}
        for i in mem:                                                  # subtree size per root
            j = i
            while j not in size:
                j = parent[j]
            top[i] = j
            size[j] += 1
        roots.sort(key=lambda r: -size[r])
        root = roots[0]
        cb[k] = root
        out_els[root] = E["C"]
        for i, j in internal:
            deg[i] += 1
            deg[j] += 1
        new_edges += internal
        new_edges.append((bb[k, 1], root))
        deg[root] += 1
        attached = [i for i in mem if top[i] == root]
        for r in roots[1:]:
            free = [i for i in attached if deg[i] < caps[int(out_els[i])]]
            if not free:
                detached += size[r]
                for i in mem:
                    if top[i] == r:
                        alive[i] = False
                new_edges = [(a, b) for a, b in new_edges if alive[a] and alive[b]]
                continue
            host = min(free, key=lambda i: np.linalg.norm(p[i] - p[r]))
            new_edges.append((host, r))
            deg[host] += 1
            deg[r] += 1
            attached += [i for i in mem if top[i] == r]
        # L: CB on the side where (N-CA).((C-CA)x(CB-CA)) > 0; reflect the whole branch if not
        grp = np.array([i for i in mem if alive[i]])
        if chiral_volume(p[bb[k, 0]][None], p[bb[k, 1]][None], p[bb[k, 2]][None], p[root][None])[0] < 0:
            p[grp] = _reflect(p[grp], p[bb[k, 0]], p[bb[k, 1]], p[bb[k, 2]])

    # carbonyl O: in the CA-C-N(next) plane on the outer bisector
    o_pos = np.empty((n_res, 3))
    for k in range(n_res):
        c = p[bb[k, 2]]
        u = p[bb[k, 1]] - c
        u /= np.linalg.norm(u)
        if k + 1 < n_res:
            w = p[bb[k + 1, 0]] - c
            d = -(u + w / np.linalg.norm(w))
        else:
            d = -u
        o_pos[k] = c + BOND[("C", "O")] * d / np.linalg.norm(d)

    # renumber: alive atoms (backbone and side), then the O atoms
    keep = np.flatnonzero(alive)
    index = np.full(n_at, -1)
    index[keep] = np.arange(len(keep))
    o_idx = len(keep) + np.arange(n_res)
    e = np.array([(index[a], index[b]) for a, b in new_edges if alive[a] and alive[b]], dtype=np.int64)
    e = np.concatenate([e.reshape(-1, 2), np.stack([index[bb[:, 2]], o_idx], 1)])
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    out_pos = np.concatenate([p[keep], o_pos])
    out_el = np.concatenate([out_els[keep], np.full(n_res, E["O"])])
    deg = np.bincount(e.ravel(), minlength=len(out_pos))
    side_mask = np.ones(len(out_pos), dtype=bool)
    side_mask[index[bb.ravel()]] = False
    side_mask[o_idx] = False
    over = side_mask & (deg > polymer.element_caps(out_el))
    out_el[over] = E["C"]

    # names: N CA C O, CB, then element + Greek letter by bonds from CA + count
    resseq = np.concatenate([res[keep] + 1, np.arange(1, n_res + 1)])
    names = [""] * len(out_pos)
    for k, nm in enumerate(("N", "CA", "C")):
        for i in index[bb[:, k]]:
            names[i] = nm
    for i in o_idx:
        names[i] = "O"
    nbr = [[] for _ in range(len(out_pos))]
    for a, b in e:
        nbr[a].append(b)
        nbr[b].append(a)
    cb_new = np.where(cb >= 0, index[np.maximum(cb, 0)], -1)
    for k in range(n_res):
        if cb_new[k] < 0:
            continue
        depth, count = {int(cb_new[k]): 0}, {}
        queue = deque([int(cb_new[k])])
        names[cb_new[k]] = "CB"
        while queue:
            i = queue.popleft()
            for j in nbr[i]:
                if side_mask[j] and j not in depth:
                    depth[j] = depth[i] + 1
                    g = GREEK[min(depth[j], len(GREEK) - 1)]
                    el = atoms.ELEMENTS[out_el[j]]
                    count[(el, g)] = count.get((el, g), 0) + 1
                    names[j] = f"{el}{g}{count[(el, g)]}"[:4]
                    queue.append(j)
    resnames = ["GLY" if cb_new[r - 1] < 0 else "UNK" for r in resseq]
    idx = {"n": index[bb[:, 0]], "ca": index[bb[:, 1]], "c": index[bb[:, 2]], "o": o_idx, "cb": cb_new}
    stats = {"residues": n_res, "glycine": int((cb_new < 0).sum()), "atoms_in": n_at, "atoms_out": len(out_pos),
             "carbonyl_O_added": n_res, "side_leaves_trimmed": trimmed,
             "side_atoms_detached": detached + dropped_unreached,
             "backbone_atoms_unused": int(len(backbone) - 3 * n_res),
             "side_atoms_relabelled_to_C": int(over.sum()),
             "side_atoms_per_residue": float(side_mask.sum() / n_res)}
    return {"pos": out_pos, "els": out_el, "edges": e, "resseq": resseq, "names": names, "resnames": resnames,
            **idx, "stats": stats}


def set_targets(r: polymer.Restraints, table) -> None:
    """Override bond (1-2) and 1-3 targets: table = [(pairs (k, 2) with i < j, target(s)), ...]."""
    n_at = r.n
    for keys, values in ((r.keys12, r.b0_np), (r.keys13, r.a0_np)):
        where = {int(k): i for i, k in enumerate(keys)}
        for pr, target in table:
            target = np.broadcast_to(np.asarray(target, dtype=np.float64), (len(pr),))
            for k, t in zip(pr[:, 0] * n_at + pr[:, 1], target):
                if int(k) in where:
                    values[where[int(k)]] = t
    r.b0 = torch.as_tensor(r.b0_np, dtype=torch.float32, device=r.b0.device)
    r.a0 = torch.as_tensor(r.a0_np, dtype=torch.float32, device=r.a0.device)


def restraints(pep: dict, ref: dict, device, seed: int = 0, link_weight=None, chiral_weight=None):
    """polymer.Restraints with the peptide's bond and 1-3 targets, plus the extra terms
    (trans planar peptide, L chirality) as a callable for polymer.relax.

    link_weight (n_res - 1) and chiral_weight (n_res) scale the extra terms per peptide link
    and per residue (to switch new ones on gradually); mutable: change them between relaxations.
    """
    r = polymer.Restraints(pep["els"], pep["edges"], ref, device, seed=seed)
    n, ca, c, o, cb = (pep[k] for k in ("n", "ca", "c", "o", "cb"))
    nxt = np.arange(len(n) - 1)
    has_cb = np.flatnonzero(cb >= 0)

    def pairs(a, b):
        return np.stack([np.minimum(a, b), np.maximum(a, b)], 1)

    bonds = [(pairs(n, ca), _bond("N", "CA")), (pairs(ca, c), _bond("CA", "C")),
             (pairs(c[nxt], n[nxt + 1]), _bond("C", "N+")), (pairs(c, o), _bond("C", "O")),
             (pairs(ca[has_cb], cb[has_cb]), _bond("CA", "CB"))]

    def third(x, y, z):
        return _third_side(_bond(x, y), _bond(y, z), ANGLE_DEG[(x, y, z)])

    angles = [(pairs(n, c), third("N", "CA", "C")),
              (pairs(ca[nxt], n[nxt + 1]), third("CA", "C", "N+")),
              (pairs(c[nxt], ca[nxt + 1]), third("C", "N+", "CA+")),
              (pairs(ca, o), third("CA", "C", "O")),
              (pairs(o[nxt], n[nxt + 1]), third("O", "C", "N+")),
              (pairs(n[has_cb], cb[has_cb]), third("N", "CA", "CB")),
              (pairs(c[has_cb], cb[has_cb]), third("C", "CA", "CB"))]

    set_targets(r, bonds + angles)

    t = {k: torch.as_tensor(v, device=device) for k, v in
         {"ca0": ca[nxt], "ca1": ca[nxt + 1], "o0": o[nxt],
          "n": n[has_cb], "ca": ca[has_cb], "c": c[has_cb], "cb": cb[has_cb]}.items()}

    t["c0"], t["n1"] = (torch.as_tensor(v, device=device) for v in (c[nxt], n[nxt + 1]))
    lw = np.ones(len(nxt)) if link_weight is None else link_weight
    cw = np.ones(len(n)) if chiral_weight is None else chiral_weight

    def extra(x):
        wl = torch.as_tensor(lw, dtype=x.dtype, device=x.device)
        wc = torch.as_tensor(cw[has_cb], dtype=x.dtype, device=x.device)
        cos_omega = cos_dihedral(x[t["ca0"]], x[t["c0"]], x[t["n1"]], x[t["ca1"]])
        return {
            "peptide": (wl * (((x[t["ca0"]] - x[t["ca1"]]).norm(dim=1) - CA_CA) ** 2
                              + ((x[t["o0"]] - x[t["ca1"]]).norm(dim=1) - O_CA_NEXT) ** 2
                              + OMEGA_WEIGHT * (1 + cos_omega))).sum(),
            "chirality": (wc * (chiral_volume(x[t["n"]], x[t["ca"]], x[t["c"]], x[t["cb"]]) - CHIRAL_VOLUME) ** 2
                          ).sum(),
        }

    return r, extra


def report(pep: dict, pos: np.ndarray) -> dict:
    n, ca, c, o, cb = (pep[k] for k in ("n", "ca", "c", "o", "cb"))
    has_cb = cb >= 0
    vol = chiral_volume(pos[n[has_cb]], pos[ca[has_cb]], pos[c[has_cb]], pos[cb[has_cb]])
    omega = dihedral(pos[ca[:-1]], pos[c[:-1]], pos[n[1:]], pos[ca[1:]])
    co = np.linalg.norm(pos[c] - pos[o], axis=1)
    return {"L_fraction": float((vol > 0).mean()) if len(vol) else None,
            "chiral_volume_median": float(np.median(vol)) if len(vol) else None,
            "omega_trans_fraction_gt150": float((np.abs(omega) > 150).mean()),
            "omega_abs_median_deg": float(np.median(np.abs(omega))),
            "CO_bond_median_A": float(np.median(co))}
