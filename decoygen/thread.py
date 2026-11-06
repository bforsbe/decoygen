"""Thread a bond forest into one linear chain (rung 2 -> rung 3 of the ladder).

1. split (optional): cut the forest into long paths by repeatedly removing the longest bond
   path of every piece that still has one of >= `split` atoms; shorter pieces stay bonded to a
   neighbouring path as side branches. Without it, a single big tree keeps only its longest path.
2. Each piece gets two ends, the atoms furthest apart along its bonds (double breadth-first
   search); its backbone is the bond path between them, and every other atom joins the backbone
   atom it hangs off ("residue" = one backbone atom plus its branches).
3. Pieces are ordered and oriented by a travelling-salesman heuristic on the end-to-end gaps
   (nearest neighbour, then 2-opt with orientation flips).
4. fill: each gap is bridged by new carbon atoms, as many as a zig-zag with protein bond and
   1-3 distances needs; prune: side-branch atoms more than `prune` bonds from the backbone go.
Atoms pushed past their element's bond cap by the joins become carbon.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from . import atoms, polymer

# a zig-zag with C-C bonds of 1.53 A and 1-3 distances of 2.49 A advances 2.49 / 2 A per bond
RISE_PER_BOND = 2.49 / 2


def bfs(adj: list[list[int]], start: int) -> dict[int, int]:
    """Parent of each atom reachable from start (start maps to -1), in visiting order."""
    parent, queue = {start: -1}, deque([start])
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in parent:
                parent[j] = i
                queue.append(j)
    return parent


def backbone(adj: list[list[int]], members: np.ndarray) -> list[int]:
    """Bond path between the two atoms of a fragment furthest apart along its bonds."""
    a = list(bfs(adj, int(members[0])))[-1]
    parent = bfs(adj, a)
    path = [list(parent)[-1]]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    return path


def hang(adj: list[list[int]], path: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    """Each atom of the fragment -> (index along path of the backbone atom it hangs off, bonds from it)."""
    owner, depth = {atom: k for k, atom in enumerate(path)}, dict.fromkeys(path, 0)
    queue = deque(path)
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in owner:
                owner[j], depth[j] = owner[i], depth[i] + 1
                queue.append(j)
    return owner, depth


def tour(starts: np.ndarray, ends: np.ndarray, passes: int = 50) -> tuple[list[int], list[bool]]:
    """Open-path order and orientation of fragments minimising the summed end-to-start gaps.

    starts/ends: (F, 3) end positions. Returns fragment order and, per position, whether that
    fragment is traversed reversed (end first).
    """
    f = len(starts)
    left = set(range(1, f))
    order, flip = [0], [False]
    while left:                                   # nearest neighbour, either orientation
        tail = starts[order[-1]] if flip[-1] else ends[order[-1]]
        cand = np.fromiter(left, dtype=np.int64)
        ds, de = (np.linalg.norm(starts[cand] - tail, axis=1), np.linalg.norm(ends[cand] - tail, axis=1))
        k = int(np.argmin(np.minimum(ds, de)))
        order.append(int(cand[k]))
        flip.append(bool(de[k] < ds[k]))
        left.remove(int(cand[k]))

    def entry(p):
        return ends[order[p]] if flip[p] else starts[order[p]]

    def exit_(p):
        return starts[order[p]] if flip[p] else ends[order[p]]

    for _ in range(passes):                       # 2-opt: reverse order[i..j] and flip each fragment in it
        improved = False
        for i in range(f):
            for j in range(i, f):
                before = (np.linalg.norm(entry(i) - exit_(i - 1)) if i > 0 else 0.0) + (
                    np.linalg.norm(entry(j + 1) - exit_(j)) if j < f - 1 else 0.0)
                after = (np.linalg.norm(exit_(j) - exit_(i - 1)) if i > 0 else 0.0) + (
                    np.linalg.norm(entry(j + 1) - entry(i)) if j < f - 1 else 0.0)
                if after < before - 1e-6:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    flip[i:j + 1] = [not x for x in flip[i:j + 1][::-1]]
                    improved = True
        if not improved:
            break
    return order, flip


def bridge(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of new atoms spanning a -> b: evenly spaced on the line, with a small sideways
    jitter so relaxation can fold them into a zig-zag."""
    k = max(1, int(np.ceil(np.linalg.norm(b - a) / RISE_PER_BOND)) - 1)
    t = np.arange(1, k + 1)[:, None] / (k + 1)
    return a + t * (b - a) + rng.normal(scale=0.3, size=(k, 3))


def decompose(n: int, edges: np.ndarray, min_len: int) -> np.ndarray:
    """Cut a bond forest into long paths: repeatedly remove the longest bond path of every piece
    that still has one of >= min_len atoms. What is left (pieces shorter than that) stays bonded
    to one neighbouring path as a side branch. Returns the bonds kept; the others are cut."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(int(j))
        adj[j].add(int(i))
    alive, on_path, kept = np.ones(n, dtype=bool), np.zeros(n, dtype=bool), []

    def sub(i):                                   # neighbours still in the working graph
        return [j for j in adj[i] if alive[j]]

    sub_adj = _View(adj, alive)        # bfs/backbone see only atoms not yet on a path
    changed = True
    while changed:
        changed = False
        seen = np.zeros(n, dtype=bool)
        for s in np.flatnonzero(alive):
            if seen[s]:
                continue
            members = np.fromiter(bfs(sub_adj, int(s)), dtype=np.int64)
            seen[members] = True
            path = backbone(sub_adj, members)
            if len(path) < min_len:
                continue
            kept += list(pairwise(path))
            alive[path], on_path[path] = False, True
            changed = True
    # leftovers: keep their internal bonds and one bond to a path atom per piece
    seen = np.zeros(n, dtype=bool)
    for s in np.flatnonzero(alive):
        if seen[s]:
            continue
        members = list(bfs(sub_adj, int(s)))
        seen[members] = True
        kept += [(i, j) for i in members for j in sub(i) if i < j]
        anchor = next(((i, j) for i in members for j in adj[i] if on_path[j]), None)
        if anchor is not None:
            kept.append(anchor)
    return np.array(kept, dtype=np.int64).reshape(-1, 2)


class _View:
    """adjacency restricted to atoms still alive (indexable like a list of neighbour lists)."""

    def __init__(self, adj, alive):
        self.adj, self.alive = adj, alive

    def __getitem__(self, i):
        return [j for j in self.adj[i] if self.alive[j]]


def thread(pos, els, edges, *, prune: int | None, fill: bool, seed: int = 0):
    """Returns (pos, els, edges, resseq, backbone, links, stats): atoms renumbered, the residue
    number of each atom along the chain, which atoms form the backbone, and the remaining gaps
    (atom pairs) when not filled."""
    n = len(pos)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(int(j))
        adj[j].append(int(i))
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_frag, label = csgraph.connected_components(graph, directed=False)
    paths = [backbone(adj, np.flatnonzero(label == c)) for c in range(n_frag)]
    order, flip = tour(np.array([pos[p[0]] for p in paths]), np.array([pos[p[-1]] for p in paths]))

    rng = np.random.default_rng(seed)
    resseq, keep = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    new_pos, new_edges, links, residue, prev = [], [], [], 0, None
    for c, rev in zip(order, flip):
        p = paths[c][::-1] if rev else paths[c]
        if prev is not None:
            if fill:
                chain = [prev]
                for x in bridge(pos[prev], pos[p[0]], rng):
                    residue += 1
                    new_pos.append((x, residue))
                    chain.append(n + len(new_pos) - 1)
                new_edges += list(zip(chain, chain[1:] + [p[0]]))
            else:
                links.append((prev, p[0]))
        owner, depth = hang(adj, p)
        for atom, k in owner.items():
            if prune is None or depth[atom] <= prune:
                keep[atom], resseq[atom] = True, residue + k + 1
        residue += len(p)
        prev = p[-1]

    on_backbone = np.zeros(n, dtype=bool)
    for p in paths:
        on_backbone[p] = True

    # renumber: kept old atoms, then the new ones
    index = np.full(n + len(new_pos), -1)
    index[np.flatnonzero(keep)] = np.arange(keep.sum())
    index[n:] = keep.sum() + np.arange(len(new_pos))
    all_edges = index[np.concatenate([edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])]
    all_edges = all_edges[(all_edges >= 0).all(1)]
    pos_out = np.concatenate([pos[keep], np.array([x for x, _ in new_pos]).reshape(-1, 3)])
    els_out = np.concatenate([els[keep], np.full(len(new_pos), atoms.ELEMENTS.index("C"))])
    res_out = np.concatenate([resseq[keep], np.array([r for _, r in new_pos], dtype=np.int64)])
    bb_out = np.concatenate([on_backbone[keep], np.ones(len(new_pos), dtype=bool)])
    links = [(index[a], index[b]) for a, b in links]

    # joins can push a terminal O or S past its bond cap: make those carbon
    deg = np.bincount(all_edges.ravel(), minlength=len(pos_out))
    over = deg > polymer.element_caps(els_out)
    els_out[over] = atoms.ELEMENTS.index("C")
    stats = {"atoms_in": n, "fragments": n_frag, "pruned": int(n - keep.sum()), "added": len(new_pos),
             "relabelled_to_C": int(over.sum()), "residues": residue}
    return pos_out, els_out, all_edges, res_out, bb_out, links, stats
