"""Thread a fragmented polymer into one chain, for rainbow colouring in ChimeraX.

For files without a decoygen header (e.g. the detector-optimised polymers); ladder models
use `decoygen thread`, which shares the algorithm (decoygen/thread.py).

    python experiments/hallucination/thread.py runs/hallucination/NAME_atoms.pdb [--fill] [--prune 4] [--relax 3000]

Each fragment (connected component of the bond graph) gets two ends: the atoms furthest apart
along its bonds (double breadth-first search; exact for trees, close enough with rings). The
backbone of a fragment is the bond path between its ends; every other atom joins the backbone
atom it hangs off, so a backbone atom plus its branches is one "residue". Fragments are then
ordered and oriented by a travelling-salesman heuristic on the end-to-end gaps (nearest
neighbour, then 2-opt with orientation flips), and residues are numbered along the whole chain.

Options:
  --prune N   drop side-branch atoms more than N bonds from the backbone
  --fill      bridge each gap with new carbon atoms (one residue each), as many as a zig-zag
              with protein bond and 1-3 distances needs to span it; the chain is then one molecule
  --relax S   S steps of polymer.relax (bonds, angles, sterics, envelope) on the result

Writes next to the input, NAME_TAG with TAG = threaded, or chain when --fill:
  NAME_TAG.pdb       one chain A; bonds as CONECT; B-factor = position along the chain (0-100)
  NAME_TAG_links.pb  without --fill: the gaps between fragments as ChimeraX pseudobonds
  NAME_TAG.cxc       opens them, rainbow by chain position
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph

from decoygen import atoms, io, polymer
from decoygen.thread import decompose, thread


def relax(pos, els, edges, steps: int, seed: int):
    """polymer.relax in the hallucination frame, with the default decoy's envelope."""
    import hallucinate

    from decoygen.params import DecoyParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = DecoyParams(seed=seed)
    _, _, mask = hallucinate.decoy_atoms(params, device)
    offset = (hallucinate.BOX * hallucinate.APIX - params.box * params.apix) / 2
    env = polymer.Envelope(mask, params.apix, device)
    x0 = torch.tensor(pos - offset, dtype=torch.float32, device=device)
    x, info = polymer.relax(x0, els, edges, polymer.load_reference(), envelope=env, steps=steps, seed=seed,
                            log=lambda r: print(f"  relax step {r['step']}: bond {r['bond']:.4f} "
                                                f"angle {r['angle']:.4f} steric {r['steric']:.4f}", flush=True))
    info.pop("history")
    return x.cpu().numpy() + offset, info


def write(stem: Path, pos, els, edges, resseq, links, title: str):
    n, n_res = len(pos), int(resseq.max())
    # atom names unique within a residue (needed for the pseudobond atom specs): element + index
    order_atoms = np.lexsort((np.arange(n), resseq))
    names, count = [""] * n, {}
    for i in order_atoms:
        count[resseq[i]] = count.get(resseq[i], 0) + 1
        names[i] = f"{atoms.ELEMENTS[els[i]]}{count[resseq[i]]}"
    serial = np.empty(n, dtype=np.int64)
    serial[order_atoms] = np.arange(1, n + 1)
    xyz = pos[:, ::-1]
    bfac = 100 * (resseq - 1) / max(n_res - 1, 1)
    lines = []
    for i in order_atoms:
        el = atoms.ELEMENTS[els[i]]
        lines.append(f"HETATM{serial[i]:5d} {names[i]:<4s} UNK A{resseq[i]:4d}    "
                     f"{xyz[i, 0]:8.3f}{xyz[i, 1]:8.3f}{xyz[i, 2]:8.3f}  1.00{bfac[i]:6.2f}          {el:>2s}")
    partners = [[] for _ in range(n)]
    for i, j in edges:
        partners[serial[i] - 1].append(serial[j])
        partners[serial[j] - 1].append(serial[i])
    for s, ps in enumerate(partners):
        for k in range(0, len(ps), 4):
            lines.append(f"CONECT{s + 1:5d}" + "".join(f"{q:5d}" for q in ps[k:k + 4]))
    lines.append("END")
    stem.with_name(stem.name + ".pdb").write_text("\n".join(lines) + "\n")

    gaps = ""
    if links:
        stem.with_name(stem.name + "_links.pb").write_text(
            "; gaps between fragments, in chain order\n"
            + "".join(f"/A:{resseq[a]}@{names[a]} /A:{resseq[b]}@{names[b]}\n" for a, b in links))
        gaps = f"open {stem.name}_links.pb\ncolor #2 #9a9893\nstyle #2 dashes 6\n"
    stem.with_name(stem.name + ".cxc").write_text(f"""# {title}
# Open in ChimeraX from this folder:  open {stem.name}.cxc
# Colour runs blue -> red along the chain{"; dashed lines are the gaps between fragments" if links else ""}.
close session
set bgColor white
graphics silhouettes true
lighting soft
open {stem.name}.pdb
{gaps}style stick
color bfactor #1 palette rainbow range 0,100
size #1 stickRadius 0.15
""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atoms", type=Path, help="NAME_atoms.pdb with CONECT bonds")
    ap.add_argument("--prune", type=int, default=None, help="max bonds from the backbone for side-branch atoms")
    ap.add_argument("--fill", action="store_true", help="bridge the gaps with new atoms")
    ap.add_argument("--relax", type=int, default=0, help="geometry relaxation steps after threading")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: next to the input)")
    ap.add_argument("--split", type=int, default=None,
                    help="first cut the bond tree into longest paths of >= SPLIT atoms (pieces shorter stay as branches)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pos, els, edges = io.read_pdb(args.atoms, atoms.ELEMENTS)
    if args.split:
        n_before = len(edges)
        edges = decompose(len(pos), edges, args.split)
        print(f"split into paths of >= {args.split} atoms: cut {n_before - len(edges)} of {n_before} bonds")
    pos, els, edges, resseq, _, links, stats = thread(pos, els, edges, prune=args.prune, fill=args.fill,
                                                   seed=args.seed)
    print(stats)
    if links:
        gaps = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in links])
        print(f"gaps between fragments: total {gaps.sum():.0f} A, median {np.median(gaps):.1f} A, "
              f"max {gaps.max():.1f} A")
    if args.relax:
        pos, info = relax(pos, els, edges, args.relax, args.seed)
        print(info)
    n_pieces = csgraph.connected_components(
        sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(pos),) * 2),
        directed=False)[0]
    tag = (("chain" if args.fill else "threaded") + (f"_split{args.split}" if args.split else "")
           + (f"_prune{args.prune}" if args.prune is not None else ""))
    out = args.out or args.atoms.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (args.atoms.name.removesuffix("_atoms.pdb") + f"_{tag}")
    title = (f"{stats['fragments']} fragments threaded into one chain of {stats['residues']} residues "
             f"({len(pos)} atoms, {n_pieces} connected piece(s))")
    write(stem, pos, els, edges, resseq, links, title)
    print(f"wrote {stem.name}.pdb/.cxc: {title}")


if __name__ == "__main__":
    main()
