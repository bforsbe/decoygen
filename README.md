# decoygen

Generator for synthetic, non-protein **decoy cryo-EM density maps**: maps that match
proteins in low-order statistics (histogram, radial power spectrum, solvent fraction,
volume per mass, envelope roughness) but carry no protein structure (no chain,
helices, sheets or chirality). Intended as hard negatives for map-interpretation
networks and for probing hallucination.

The generator is never trained or fitted on real maps or atomic models; see
[`decoy_density_brief.md`](decoy_density_brief.md) for the design and data tiers.

**Status:** v0 (milestone 1) under development: rough envelope with α→D calibration,
Turing mesoscale field, Poisson pseudo-atom gas, Gerchberg–Saxton amplitude imposition,
half-maps with resolution-calibrated noise, MRC/JSON output and diagnostics.

## Install

```bash
pip install -e ".[test]"
```

## Usage

```bash
decoygen generate --out runs/ --seed 1 --plots          # one decoy, 128³ at 1 Å/px, 3.5 Å
decoygen batch --out runs/set1 --n 100 --start-seed 0   # many decoys + manifest.csv
decoygen calibrate                                       # rebuild the α → D lookup table
```

Each decoy directory holds the full map, two half-maps, the noise-free map, the
envelope mask (all MRC, float32, origin 0) and a JSON file with all parameters,
seeds, versions and diagnostics.

## Tests

```bash
pytest -m "not slow"
```

## License

MIT, see [LICENSE](LICENSE).
