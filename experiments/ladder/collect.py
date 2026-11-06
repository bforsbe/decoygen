"""Lay out runs/ladder/<group>/<id>/ for every rung in rungs.py, copying results that already exist.

    python experiments/ladder/collect.py

Per rung:
  atoms.pdb / atoms.cif           the atom model (mirrored here for mirror rungs)
  maps/                           half1, half2, full (their mean), truth (noise-free), support;
                                  deh_wideTarget / deh_tightTarget / deh_highRes (DeepEMhancer)
  modelangelo/<v>/<v>.cif         ModelAngelo build_no_seq on maps/<v>.mrc (+ <v>.log)
  detector/rung_*                 C-alpha CNN in ModelAngelo's own frame (hallucinate.py atoms):
                                  full / clean / ca_prob maps, metrics.json
  detector/build/rung/rung.cif    ModelAngelo build_no_seq on detector/rung_full.mrc
  RUNG.md                         what the rung adds, where it came from, results (summarise.py)

Existing results are copied from runs/deepemhancer (NAME = Rung.deh) and runs/hallucination
(NAME = Rung.det), final models and logs only; nothing is overwritten. run.sbatch computes the rest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gemmi
from rungs import RUNGS, Rung, rung_dir

DEH = Path("runs/deepemhancer")
HALL = Path("runs/hallucination")
BOX_A = 144.0
MAPS = ["half1", "half2", "full", "truth", "support", "deh_wideTarget", "deh_tightTarget", "deh_highRes"]
SEE = Path("see_alpha_output/see_alpha_output_ca.cif")    # confident C-alpha points
VARIANTS = ["full", "deh_wideTarget", "deh_tightTarget", "deh_highRes"]


def copy(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def atoms_file(d: Path) -> Path | None:
    found = sorted(d.glob("atoms.*"))
    return found[0] if found else None


def place_atoms(r: Rung, d: Path) -> None:
    if r.source is None or not r.source.exists() or atoms_file(d):
        return
    if not r.mirror:
        copy(r.source, d / f"atoms{r.source.suffix}")
        return
    st = gemmi.read_structure(str(r.source))
    for model in st:
        for ch in model:
            for res in ch:
                for a in res:
                    a.pos = gemmi.Position(BOX_A - a.pos.x, a.pos.y, a.pos.z)
    st.write_pdb(str(d / "atoms.pdb"))


def header(r: Rung) -> str:
    return (f"# {r.id}: {r.title}\n\n"
            f"**Adds:** {r.adds}\n\n"
            f"**Atoms:** `{r.source}`{' (mirrored through x = 72 A)' if r.mirror else ''}\n\n"
            f"**Made by:** {r.recipe}\n")


def main():
    for r in RUNGS:
        d = rung_dir(r)
        d.mkdir(parents=True, exist_ok=True)
        place_atoms(r, d)
        if r.deh:
            for m in MAPS:
                copy(DEH / f"{r.deh}_{m}.mrc", d / "maps" / f"{m}.mrc")
            for v in VARIANTS:
                copy(DEH / "build" / f"{r.deh}_{v}" / f"{r.deh}_{v}.cif", d / "modelangelo" / v / f"{v}.cif")
                copy(DEH / "build" / f"{r.deh}_{v}.log", d / "modelangelo" / f"{v}.log")
                copy(DEH / "build" / f"{r.deh}_{v}" / SEE, d / "modelangelo" / v / SEE)
        if r.det:
            for m in ("full", "clean", "ca_prob"):
                copy(HALL / f"{r.det}_{m}.mrc", d / "detector" / f"rung_{m}.mrc")
            copy(HALL / f"{r.det}_metrics.json", d / "detector" / "rung_metrics.json")
            copy(HALL / "build" / r.det / f"{r.det}.cif", d / "detector" / "build" / "rung" / "rung.cif")
            copy(HALL / "build" / f"{r.det}.log", d / "detector" / "build" / "rung.log")
            copy(HALL / "build" / r.det / SEE, d / "detector" / "build" / "rung" / SEE)
        if not (d / "RUNG.md").exists():
            (d / "RUNG.md").write_text(header(r))
        state = "atoms" if atoms_file(d) else "no atoms yet"
        print(f"{d}: {state}, {len(list(d.rglob('*.cif')))} cif, {len(list(d.rglob('*.mrc')))} maps")


if __name__ == "__main__":
    main()
