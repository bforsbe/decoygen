"""Atom models on the protein-likeness ladder, and the guards between its rungs.

Every rung of the ladder adds one named piece of protein knowledge to the rung below it.
Each `decoygen` model step turns a model of one rung into a model of the next, and every
model file says which rung it is:

    REMARK 999 DECOYGEN MODEL
    REMARK 999 DECOYGEN RUNG 3 threaded chain
    REMARK 999 DECOYGEN MIRRORED no
    REMARK 999 DECOYGEN VERSION 0.2.0
    REMARK 999 DECOYGEN STEP 1 gas seed=0 mass_kda=100.0 frame_box=144.0
    REMARK 999 DECOYGEN STEP 2 polymer steps=3000
    ...

A step refuses a model that is not exactly the rung it builds on (`require`), so a rung is
never skipped or applied twice by accident; `--force` overrides and is recorded in the history.
Mirroring is the last thing done to a model: no step builds on a mirrored one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import __version__, atoms

RUNGS = {
    0: "hard-core liquid",
    1: "atom gas",
    2: "raw polymer",
    3: "threaded chain",
    4: "peptide backbone",
    5: "real side chains",
    6: "secondary structure",
    7: "fold",
    8: "protein",
}
# the step that makes each rung (rung 0 is made from rung 1: same atoms, overlaps removed)
MAKES = {0: "hardcore", 1: "gas", 2: "polymer", 3: "thread", 4: "peptide", 5: "sidechains"}
TAG = "REMARK 999 DECOYGEN "


class RungError(RuntimeError):
    pass


@dataclass
class Model:
    pos: np.ndarray                     # (N, 3) [z, y, x] in A, like the maps
    els: np.ndarray                     # (N,) indices into atoms.ELEMENTS
    edges: np.ndarray                   # (E, 2) bonds, i < j
    box: float                          # cubic frame edge in A (CRYST1); maps are rendered in it
    rung: int | None = None             # None: not a decoygen model
    mirrored: bool = False
    history: list[str] = field(default_factory=list)
    resseq: np.ndarray | None = None    # (N,) residue number; None: one per atom
    names: list[str] | None = None      # atom names; None: element symbol
    resnames: list[str] | None = None   # per atom; None: UNK
    bfactor: np.ndarray | None = None   # (N,); None: position along the chain, 0-100

    @property
    def n(self) -> int:
        return len(self.pos)

    def then(self, rung: int, step: str, **opts) -> Model:
        """Copy with rung set and this step appended to the history (positions etc. replaced by caller)."""
        line = " ".join([step] + [f"{k}={v}" for k, v in opts.items()])
        return dataclasses.replace(self, rung=rung, history=[*self.history, line])

    def describe(self) -> str:
        r = "not a decoygen model" if self.rung is None else f"rung {self.rung} ({RUNGS[self.rung]})"
        lines = [(f"{r}{', mirrored' if self.mirrored else ''}; {self.n} atoms, {len(self.edges)} bonds, "
                  f"{self.box:g} A frame")]
        lines += [f"  step {k + 1}: {h}" for k, h in enumerate(self.history)]
        return "\n".join(lines)


def require(model: Model, rung: int, step: str, force: bool = False) -> None:
    """Refuse to build rung + 1 from anything but an unmirrored rung `rung` model."""
    if force:
        return
    want = f"rung {rung} ({RUNGS[rung]})"
    if model.rung is None:
        raise RungError(f"{step} builds on a {want} model, but this file has no decoygen header. "
                        f"Make one with `decoygen {MAKES.get(rung, '...')}`, or pass --force.")
    if model.mirrored:
        raise RungError(f"{step}: the model is mirrored. Mirror only the finished model "
                        f"(`decoygen mirror`), or pass --force.")
    if model.rung == rung:
        return
    have = f"rung {model.rung} ({RUNGS[model.rung]})"
    if model.rung < rung:
        missing = ", ".join(f"`decoygen {MAKES[k]}`" for k in range(model.rung + 1, rung + 1) if k in MAKES)
        raise RungError(f"{step} builds on {want}, but this model is {have}: run {missing} first "
                        f"(or pass --force).")
    raise RungError(f"{step} makes rung {rung + 1} from {want}, but this model is already {have}. "
                    f"Pass --force to do it anyway.")


def default_path(src: Path, rung: int, mirrored: bool = False) -> Path:
    slug = RUNGS[rung].replace(" ", "_").replace("-", "_")
    return src.parent / f"rung{rung}_{slug}{'_mirror' if mirrored else ''}.pdb"


# ---------------------------------------------------------------- PDB

def write(path: str | Path, m: Model, overwrite: bool = False) -> Path:
    """PDB with the decoygen header, CRYST1, CONECT bonds; one chain A (more chains past 9999 residues)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass --overwrite to replace it)")
    n = m.n
    resseq = m.resseq if m.resseq is not None else np.arange(1, n + 1)
    names = m.names or [atoms.ELEMENTS[e] for e in m.els]
    resnames = m.resnames or ["UNK"] * n
    if m.bfactor is not None:
        bfac = m.bfactor
    else:
        bfac = 100 * (resseq - resseq.min()) / max(int(resseq.max() - resseq.min()), 1)
    rank = np.array([("N", "CA", "C", "O", "CB").index(x) if x in ("N", "CA", "C", "O", "CB") else 5 for x in names])
    order = np.lexsort((np.arange(n), rank, resseq))                  # N CA C O CB first in a residue
    serial = np.empty(n, dtype=np.int64)
    serial[order] = np.arange(1, n + 1)
    lines = [f"{TAG}MODEL"]
    if m.rung is not None:
        lines.append(f"{TAG}RUNG {m.rung} {RUNGS[m.rung]}")
    lines += [f"{TAG}MIRRORED {'yes' if m.mirrored else 'no'}", f"{TAG}VERSION {__version__}"]
    lines += [f"{TAG}STEP {k + 1} {h}" for k, h in enumerate(m.history)]
    lines.append(f"CRYST1{m.box:9.3f}{m.box:9.3f}{m.box:9.3f}  90.00  90.00  90.00 P 1           1")
    xyz = m.pos[:, ::-1]
    record = "ATOM  " if m.resnames else "HETATM"
    for i in order:
        r = int(resseq[i])
        chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[(r - 1) // 9999 % 26]
        el = atoms.ELEMENTS[m.els[i]]
        name = names[i] if len(names[i]) == 4 else f" {names[i]:<3s}"
        lines.append(f"{record}{serial[i]:5d} {name} {resnames[i]:>3s} {chain}{(r - 1) % 9999 + 1:4d}    "
                     f"{xyz[i, 0]:8.3f}{xyz[i, 1]:8.3f}{xyz[i, 2]:8.3f}  1.00{bfac[i]:6.2f}          {el:>2s}")
    partners = [[] for _ in range(n)]
    for i, j in m.edges:
        partners[serial[i] - 1].append(int(serial[j]))
        partners[serial[j] - 1].append(int(serial[i]))
    for s, ps in enumerate(partners):
        for k in range(0, len(ps), 4):
            lines.append(f"CONECT{s + 1:5d}" + "".join(f"{q:5d}" for q in ps[k:k + 4]))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def read(path: str | Path) -> Model:
    """Any PDB file; the decoygen header (if present) gives rung, mirroring and history."""
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}
    pos, els, serial, edges = [], [], {}, set()
    resseq, names, resnames, bfac, chains = [], [], [], [], {}
    rung, mirrored, history, box = None, False, [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith(TAG):
            key, _, rest = line[len(TAG):].partition(" ")
            if key == "RUNG":
                rung = int(rest.split()[0])
            elif key == "MIRRORED":
                mirrored = rest.strip() == "yes"
            elif key == "STEP":
                history.append(rest.split(" ", 1)[1] if " " in rest else "")
        elif line.startswith("CRYST1"):
            box = float(line[6:15])
        elif line.startswith(("ATOM", "HETATM")):
            el = line[76:78].strip() or line[12:16].strip()[0]
            if el not in lookup:                                  # hydrogens, ions
                continue
            serial[int(line[6:11])] = len(pos)
            pos.append([float(line[46:54]), float(line[38:46]), float(line[30:38])])
            els.append(lookup[el])
            ch = chains.setdefault(line[21], len(chains))
            resseq.append(ch * 9999 + int(line[22:26]))
            names.append(line[12:16].strip())
            resnames.append(line[17:20].strip())
            bfac.append(float(line[60:66]))
        elif line.startswith("CONECT"):
            fields = [int(line[k:k + 5]) for k in range(6, len(line.rstrip()), 5)]
            for j in fields[1:]:
                if fields[0] in serial and j in serial:
                    a, b = serial[fields[0]], serial[j]
                    edges.add((min(a, b), max(a, b)))
    pos = np.array(pos, dtype=np.float64).reshape(-1, 3)
    if box is None:
        box = float(np.ceil(pos.max() + 10)) if len(pos) else 0.0
    return Model(pos=pos, els=np.array(els, dtype=np.int64),
                 edges=np.array(sorted(edges), dtype=np.int64).reshape(-1, 2), box=box, rung=rung,
                 mirrored=mirrored, history=history, resseq=np.array(resseq, dtype=np.int64), names=names,
                 resnames=resnames if any(r != "UNK" for r in resnames) else None, bfactor=np.array(bfac))


# ---------------------------------------------------------------- confinement

def envelope_mask(pos: np.ndarray, box: float, apix: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    """The volume a model occupies: atom counts blurred by sigma (A), thresholded at half the
    blurred value found at the 10 % least-buried atoms, holes filled. Steps confine their
    relaxation to the input model's own envelope, so no file needs the original mask."""
    n = round(box / apix)
    idx = np.clip(np.round(pos / apix).astype(np.int64), 0, n - 1)
    occ = np.zeros((n, n, n))
    np.add.at(occ, tuple(idx.T), 1.0)
    blurred = ndimage.gaussian_filter(occ, sigma / apix)
    level = 0.5 * np.percentile(blurred[tuple(idx.T)], 10)
    return ndimage.binary_fill_holes(blurred > level)
