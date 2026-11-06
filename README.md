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
protein-likeness ladder of atom models (rungs 0–4, below).

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
| 5–7 | (planned) | | real side chains, secondary structure, fold |
| 8 | real protein | | reference |

```bash
decoygen gas -o run/rung1_atom_gas.pdb --seed 0 --frame-box 144
decoygen polymer run/rung1_atom_gas.pdb          # -> run/rung2_raw_polymer.pdb
decoygen thread  run/rung2_raw_polymer.pdb       # -> run/rung3_threaded_chain.pdb
decoygen peptide run/rung3_threaded_chain.pdb    # -> run/rung4_peptide_backbone.pdb
decoygen mirror  run/rung4_peptide_backbone.pdb  # D-amino acid control (always the last step)
decoygen render  run/rung4_peptide_backbone.pdb -o run/maps4   # truth, half-maps (FSC 0.143 at 3.5 A)
decoygen info    run/rung4_peptide_backbone.pdb  # rung and full history

decoygen ladder --to 4 --out run --frame-box 144 # all of the above (rungs 0-4) in one go
```

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
