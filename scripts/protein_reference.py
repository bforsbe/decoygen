"""Rebuild decoygen/data/protein_reference.json from a set of PDB/mmCIF files.

    python scripts/protein_reference.py experiments/polymer/reference/*.cif
"""

import sys
from pathlib import Path

from decoygen.io import write_json
from decoygen.polymer import REFERENCE_FILE, protein_reference

paths = [Path(p) for p in sys.argv[1:]]
ref = protein_reference(paths)
write_json(REFERENCE_FILE, ref)
print(f"wrote {REFERENCE_FILE} from {len(paths)} structures, {ref['n_atoms']} atoms")
for cat in ("bond", "angle", "torsion", "nonbonded"):
    q = ref["quantiles"][cat]["*"]
    print(f"{cat:10s} 1%={q[1]:.2f}  median={q[50]:.2f}  99%={q[99]:.2f}  pairs={sum(ref['counts'][cat].values())}")
