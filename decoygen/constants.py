"""Tier-0/1 constants. Nothing in this package is fitted to real maps or models."""

# Protein partial specific volume: 1.35 g/cm^3  ->  ~1.21-1.23 A^3/Da.
PROTEIN_VOLUME_PER_DA = 1.21

# Average residue mass ~110 Da with ~7.8 heavy atoms -> ~14.1 Da per heavy atom
# (hydrogen mass included). Together with the volume above this gives
# 1 heavy atom per ~17 A^3.
MASS_PER_HEAVY_ATOM = 14.1

# Heavy-atom composition by count, from average amino-acid formulae.
HEAVY_ATOM_COMPOSITION = {"C": 0.63, "N": 0.17, "O": 0.19, "S": 0.01}

# Tier-1 literature scalar: SASA ~ 6.3 * M^0.73 A^2 (Miller et al. 1987, M in Da).
SASA_PREFACTOR = 6.3
SASA_EXPONENT = 0.73

# Half-map FSC of 0.143 for the full map corresponds to half-map SNR 0.143/(1-0.143).
FSC_THRESHOLD = 0.143
SNR_AT_THRESHOLD = FSC_THRESHOLD / (1.0 - FSC_THRESHOLD)
