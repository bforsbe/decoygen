#!/bin/bash
# Download the PDB entries the experiments use (kept out of git).
#   bash experiments/fetch_pdb.sh
set -euo pipefail
cd "$(dirname "$0")"
get() { mkdir -p "$1"; for id in "${@:2}"; do [ -f "$1/$id.cif" ] || curl -sfo "$1/$id.cif" "https://files.rcsb.org/download/$id.cif"; done; }
get hallucination/data 6CVM                                         # test protein, beta-galactosidase chain A
get polymer/reference 1AKE 2LZM 1UBQ 1TIM 3PGK 1HHO 2CI2 1RX2       # protein_reference.json sources
