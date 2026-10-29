#!/bin/bash
# Example gallery: one decoy per setting, with diagnostics plots and the Turing field.
#   CAL=runs/calib-<jobid>/roughness_calibration.json sbatch scripts/examples.sbatch
# Without CAL the packaged calibration table is used.
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=decoygen-examples
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load Miniforge3/25.3.1-0 >/dev/null 2>&1
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
if [ -n "${CAL:-}" ]; then export DECOYGEN_CALIBRATION=$CAL; fi
OUT=${OUT:-runs/examples}
mkdir -p "$OUT"
gen() { name=$1; shift; "$ENV/bin/decoygen" generate --device cuda --out "$OUT" --name "$name" \
          --plots --extra-maps --seed 0 "$@" | head -1; }

gen default
gen D2.10           --roughness-D 2.10
gen D2.40           --roughness-D 2.40
gen meso0.0         --meso-strength 0.0
gen meso0.7         --meso-strength 0.7
gen fill0.3_tubes   --meso-strength 0.7 --fill-fraction 0.3
gen peaks           --target-spectrum physics+peaks
gen res2.5          --resolution 2.5
gen res6.0          --resolution 6.0
gen two_domains     --mass-kda 60 --n-domains 2 --elongation 1.4
"$ENV/bin/decoygen" batch --device cuda --out "$OUT/batch" --n 4 --start-seed 1 --plots
echo "done: $OUT"
