"""The protein-likeness ladder: each rung adds one named piece of protein knowledge.

Main ladder: what a rung adds over the one below. Branch: objects biased toward one network
(ModelAngelo's C-alpha CNN) rather than toward protein; kept apart so the ladder stays a
clean dose-response axis.

Every rung lives in runs/ladder/<group>/<id>/ (see collect.py) and is evaluated the same way
(run.sbatch, summarise.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path("runs/ladder")
BOOK = Path("runs/bookmarks/raw-polymer")
HALL = Path("runs/hallucination")
CLI = ROOT / "build/cli"          # decoygen ladder --to 4 --out runs/ladder/build/cli --frame-box 144
LADDER_CMD = "decoygen ladder --to 4 --out runs/ladder/build/cli --frame-box 144 --seed 0"
SYM_FLAGS = {                                                       # tag -> decoygen ladder flags
    "C3": "--symmetry C3 --contact 0.6",
    "C7": "--symmetry C7 --contact 0.5",                           # C7 at 0.5: a ring with a pore
    # stress test: small, as rough as the calibration allows, little smoothing, rough interfaces
    "C5r": "--symmetry C5 --contact 0.5 --mass-kda 40 --roughness-D 2.6 --compactness 2 --owner-fine 1.5",
}


def SYM(sym: str) -> Path:                                         # expanded assemblies (decoygen expand)
    return ROOT / f"build/sym_{sym.lower()}/expanded"


def SYM_CMD(sym: str) -> str:
    return (f"decoygen ladder --to 5 {SYM_FLAGS[sym]} --frame-box 144 --seed 0 "
            f"--out runs/ladder/build/sym_{sym.lower()}; decoygen expand")


@dataclass
class Rung:
    id: str                       # directory name; sorts in ladder order
    title: str
    adds: str                     # the protein knowledge this rung adds
    source: Path | None           # atom model (pdb with CONECT, or cif); None = not built yet
    recipe: str                   # how the atoms were made
    group: str = "main"           # "main", "sym" (symmetric assemblies) or "branch"
    mirror: bool = False          # mirror the source through x = 72 A (the box centre)
    deh: str | None = None        # existing results: NAME in runs/deepemhancer
    det: str | None = None        # existing results: NAME in runs/hallucination


RUNGS = [
    Rung("0_hardcore", "Hard-core liquid",
         "protein atom density inside the envelope; atoms cannot overlap; no bonds",
         CLI / "rung0_hard_core_liquid.pdb", LADDER_CMD),
    Rung("1_gas", "Atom gas (v0 decoy)",
         "protein element mix and density, clustered (atoms may overlap like bonded neighbours); no bonds",
         CLI / "rung1_atom_gas.pdb", LADDER_CMD),
    Rung("2_raw_polymer", "Raw polymer",
         "covalent bond and 1-3 distances, valence per element, protein-like branching; self-avoiding",
         CLI / "rung2_raw_polymer.pdb", LADDER_CMD),
    Rung("3_threaded_chain", "Threaded chain",
         "one linear molecule: the bond tree cut into its longest paths (>= 5 atoms), joined end to end by new atoms",
         CLI / "rung3_threaded_chain.pdb", LADDER_CMD),
    Rung("4_peptide_backbone", "Peptide backbone",
         "N-CA-C(=O) repeat, planar trans peptides, L-amino acids; side atoms regrouped on CB",
         CLI / "rung4_peptide_backbone.pdb", LADDER_CMD),
    Rung("4_peptide_backbone_mirror", "Peptide backbone, mirrored",
         "rung 4 with D-amino acids (handedness control)",
         CLI / "rung4_peptide_backbone_mirror.pdb", "decoygen mirror rung4_peptide_backbone.pdb"),
    Rung("5_real_side_chains", "Real side chains",
         "backbone thinned to a protein's residues per mass (in passes of 1 in 5), random natural-frequency "
         "sequence, real side-chain conformations",
         CLI / "rung5_real_side_chains.pdb", LADDER_CMD.replace("--to 4", "--to 5")),
    Rung("5_real_side_chains_mirror", "Real side chains, mirrored",
         "rung 5 with D-amino acids (handedness control)",
         CLI / "rung5_real_side_chains_mirror.pdb", "decoygen mirror rung5_real_side_chains.pdb"),
    # 6 secondary structure; 7 fold
    Rung("8_protein", "Real protein",
         "everything: 6CVM chain A (beta-galactosidase)",
         BOOK / "protein_atoms.pdb", "6CVM chain A, placed in the 144 A frame (tag raw-polymer)",
         deh="protein", det="protein"),
    Rung("8_protein_mirror", "Mirrored protein",
         "everything except handedness (D-amino acids, left-handed helices)",
         BOOK / "protein_atoms.pdb", "6CVM chain A mirrored through x = 72 A", mirror=True,
         deh="protein_mirror", det="protein_mirror"),

    *[r for sym in SYM_FLAGS for r in (
        Rung(f"{sym}.1_gas", f"{sym} atom gas", f"rung 1 as a {sym} assembly (one unit + operators)",
             SYM(sym) / "rung1_atom_gas.pdb", SYM_CMD(sym), group="sym"),
        Rung(f"{sym}.2_raw_polymer", f"{sym} raw polymer", "rung 2, relaxed against its symmetry images",
             SYM(sym) / "rung2_raw_polymer.pdb", SYM_CMD(sym), group="sym"),
        Rung(f"{sym}.3_threaded_chain", f"{sym} threaded chain", "rung 3, one chain per copy",
             SYM(sym) / "rung3_threaded_chain.pdb", SYM_CMD(sym), group="sym"),
        Rung(f"{sym}.4_peptide_backbone", f"{sym} peptide backbone", "rung 4 per copy",
             SYM(sym) / "rung4_peptide_backbone.pdb", SYM_CMD(sym), group="sym"),
        Rung(f"{sym}.5_real_side_chains", f"{sym} real side chains", "rung 5 per copy",
             SYM(sym) / "rung5_real_side_chains.pdb", SYM_CMD(sym), group="sym"),
        Rung(f"{sym}.5m_real_side_chains_mirror", f"{sym} real side chains, mirrored",
             "rung 5 with D-amino acids (handedness control)",
             SYM(sym) / "rung5_real_side_chains_mirror.pdb", SYM_CMD(sym) + "; decoygen mirror", group="sym"),
    )],

    Rung("B1_gas_detector", "Gas optimised against the C-alpha CNN",
         "rung 1, atoms moved to maximise sum p^4 of ModelAngelo's C-alpha CNN (min distance 2.2 A)",
         HALL / "decoy_hallucinated_dmin2.2_p4_atoms.cif", "hallucinate.py optimise --dmin 2.2 --power 4",
         group="branch", det="decoy_hallucinated_dmin2.2_p4"),
    Rung("B2_polymer_fixed", "Raw polymer optimised, fixed bonds",
         "rung 2 optimised against the C-alpha CNN with hard covalent geometry",
         HALL / "decoy_polymer_fixed_p4_atoms.pdb", "hallucinate.py polymer --steps 300 --power 4",
         group="branch", det="decoy_polymer_fixed_p4"),
    Rung("B3_polymer_rewire", "Raw polymer optimised, re-wired every 10 steps",
         "as B2, bond tree rebuilt every 10 steps (changed no bonds in practice)",
         HALL / "decoy_polymer_rewire10_p4_atoms.pdb", "hallucinate.py polymer --rewire-every 10",
         group="branch", det="decoy_polymer_rewire10_p4"),
    Rung("B4_polymer_soft_lr0.05", "Raw polymer optimised, soft bonds, step 0.05",
         "as B2 with soft bonds (valence weight 0.2), hardened at step 240; re-wired nothing",
         HALL / "decoy_polymer_soft_cw0.2_lr0.05_p4_atoms.pdb",
         "hallucinate.py soft --count-weight 0.2 --lr 0.05", group="branch",
         det="decoy_polymer_soft_cw0.2_lr0.05_p4"),
    Rung("B5_polymer_soft_lr0.15", "Raw polymer optimised, soft bonds, step 0.15",
         "as B4 with 3x the step: 567 bonds broken, 139 formed, 482 fragments; final (hardened) state",
         HALL / "decoy_polymer_soft_cw0.2_lr0.15_p4_atoms.pdb",
         "hallucinate.py soft --count-weight 0.2 --lr 0.15 --trajectory-every 1", group="branch",
         det="decoy_polymer_soft_cw0.2_lr0.15_p4"),
    Rung("B6_soft_lr0.15_threaded", "B5 threaded into one chain",
         "B5's 482 fragments joined by new atoms, side branches <= 4 bonds from the backbone, relaxed",
         HALL / "decoy_polymer_soft_cw0.2_lr0.15_p4_chain_prune4.pdb",
         "experiments/hallucination/thread.py <B5 atoms> --fill --prune 4 --relax 3000", group="branch"),
]


def rung_dir(r: Rung) -> Path:
    return ROOT / r.group / r.id
