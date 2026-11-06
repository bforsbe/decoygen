# decoygen

Generator for synthetic, non-protein **decoy cryo-EM density maps**: maps that match
proteins in low-order statistics (histogram, radial power spectrum, solvent fraction,
volume per mass, envelope roughness) but carry no protein structure (no chain,
helices, sheets or chirality). Intended as hard negatives for map-interpretation
networks and for probing hallucination.

The generator is never trained or fitted on real maps or atomic models; see
[`decoy_density_brief.md`](decoy_density_brief.md) for the design and data tiers.

**Status:** v0 (milestone 1): rough envelope with calibrated roughness (D),
Turing mesoscale field, Poisson pseudo-atom gas, Gerchberg–Saxton amplitude imposition,
half-maps with resolution-calibrated noise, MRC/JSON output and diagnostics; plus the
protein-likeness ladder of atom models (rungs 0–5, below).

## Install

```bash
pip install -e ".[test]"
```

## Usage

```bash
decoygen generate --out runs/ --seed 1 --plots          # one decoy, 128³ at 1 Å/px, 3.5 Å, 100 kDa
decoygen batch --out runs/set1 --n 100 --start-seed 0   # many decoys + manifest.csv
decoygen calibrate                                       # rebuild the roughness calibration table
```

Each decoy directory holds the full map, two half-maps, the noise-free map, the
envelope mask (all MRC, float32, origin 0) and a JSON file with all parameters,
seeds, versions and diagnostics.

## Symmetric envelopes

```bash
decoygen envelope --out run/sym --symmetry C3 --seed 0                 # C3 assembly, 100 kDa in total
decoygen envelope --out run/sym --symmetry C7 --contact 0.5            # a ring with a pore
decoygen envelope --out run/sym --symmetry C3 --swap 1.0               # domain swap
decoygen envelope --out run/sym --symmetry C3 --axis random --shift 12 -8 5
```

Copies of one subunit field compete for space: the assembly is their union, each voxel belongs to the
strongest copy, and interfaces form where copies meet. The output is the envelope, a subunit label map, one
map per subunit, a JSON file with the symmetry operators and checks, and a ChimeraX script. Only Cn is
parsed so far; the code takes any list of rotations. See [docs/SYMMETRY.md](docs/SYMMETRY.md).

## The protein-likeness ladder (atom models)

Beyond the v0 maps, decoygen builds atom models that add protein knowledge one named
piece at a time, as a dose-response axis for testing how much protein-ness a map
interpretation method needs before it builds (or hallucinates) a protein. Each command
makes the next rung from the one below:

| rung | model | made by | adds |
|---|---|---|---|
| 0 | hard-core liquid | `decoygen hardcore` (from rung 1) | atoms cannot overlap; no bonds |
| 1 | atom gas | `decoygen gas` | the v0 decoy's atoms: protein element mix and density |
| 2 | raw polymer | `decoygen polymer` | covalent bond and angle distances, valence, protein-like branching |
| 3 | threaded chain | `decoygen thread` | one linear molecule |
| 4 | peptide backbone | `decoygen peptide` | N–CA–C(=O) repeat, planar trans peptides, L-amino acids |
| 5 | real side chains | `decoygen sidechains` | backbone thinned to a protein's residues per mass, random sequence, real side-chain conformations |
| 6–7 | (planned) | | secondary structure, fold |
| 8 | real protein | | reference |

```bash
decoygen gas -o run/rung1_atom_gas.pdb --seed 0 --frame-box 144
decoygen polymer run/rung1_atom_gas.pdb          # -> run/rung2_raw_polymer.pdb
decoygen thread  run/rung2_raw_polymer.pdb       # -> run/rung3_threaded_chain.pdb
decoygen peptide run/rung3_threaded_chain.pdb    # -> run/rung4_peptide_backbone.pdb
decoygen sidechains run/rung4_peptide_backbone.pdb  # -> run/rung5_real_side_chains.pdb
decoygen mirror  run/rung5_real_side_chains.pdb  # D-amino acid control (always the last step)
decoygen render  run/rung5_real_side_chains.pdb -o run/maps5   # truth, half-maps (FSC 0.143 at 3.5 A)
decoygen info    run/rung5_real_side_chains.pdb  # rung and full history

decoygen ladder --to 5 --out run --frame-box 144 # all of the above (rungs 0-5) in one go
decoygen ladder --to 5 --out run_c3 --frame-box 144 --symmetry C3   # a C3 assembly: one unit + operators
decoygen expand run_c3/rung5_real_side_chains.pdb -o c3_full.pdb    # the whole assembly, one chain per copy
```

A symmetric model holds one asymmetric unit and its operators (also as `REMARK 350 BIOMT`, so ChimeraX
shows the assembly with `sym #1 assembly 1`). Every step relaxes the unit against its symmetry images, so
interfaces are built from both sides at every rung.

The atom count, and so the map's mass, is the same on every rung from 1 to 5. Threading pays for
each bridging atom by removing the nearest side-branch leaf (`--no-conserve` keeps them all, which adds
~17 % mass). The peptide step pays for each carbonyl O with a side atom; when the chain is too long for
that, its most bent backbone atoms first leave the chain in place as side atoms. The side-chain step then thins the over-long backbone to a
protein's ~7.8 atoms per residue in passes (at most 1 residue in 5 per pass, most bent first).
Each excised residue's atoms stay in place as its neighbour's side group, and the new junction
is switched on gradually during relaxation, so the volume never collapses. A random
natural-frequency sequence is then matched to the residues by side-group size, and each group
is replaced by the best-fitting real side-chain conformation. The conformations come from
`decoygen/data/sidechain_library.json`, built from the same reference proteins as the
geometry statistics.

**Rung guards.** Every model file carries a header with its rung, whether it is mirrored,
the decoygen version and every step (with options) that made it:

```
REMARK 999 DECOYGEN RUNG 4 peptide backbone
REMARK 999 DECOYGEN MIRRORED no
REMARK 999 DECOYGEN STEP 1 gas seed=0 frame_box=144.0
REMARK 999 DECOYGEN STEP 2 polymer steps=3000 seed=0
...
```

A step accepts only the rung directly below it, so a rung cannot be skipped or applied
twice by accident (`decoygen peptide rung2_raw_polymer.pdb` stops with *run `decoygen thread`
first*). Steps refuse mirrored models (mirroring is the last step) and files without the header.
`--force` overrides the check and is recorded in the history. Outputs default to
`rung<N>_<name>.pdb` next to the input, and existing files are not overwritten without
`--overwrite`. Maps from `decoygen render` carry the rung and history in their MRC labels
and in `maps.json`.

## Viewing decoys in ChimeraX

`scripts/chimerax_decoys.py` opens every decoy folder under a directory, one tile per decoy.
It shows the noise-free map contoured to enclose the envelope volume, and opens the full map,
envelope mask and Turing field hidden at the same position. Toggle these in the Models panel.

```bash
chimerax --cmd "runscript scripts/chimerax_decoys.py runs/examples"
# options: --halves (also open half-maps), --fraction 0.7 (tighter contour), --columns 4
```

or, inside ChimeraX: `runscript /path/to/chimerax_decoys.py /path/to/decoys`.

## Methods

See [`docs/METHODS.md`](docs/METHODS.md) for the full description of methods, math and
design decisions, with references.

## Tests

```bash
pytest -m "not slow"
```

## License

MIT, see [LICENSE](LICENSE).
