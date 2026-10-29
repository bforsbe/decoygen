#!/bin/bash
# GPU smoke run on a Berzelius compute node: timing of default 128^3 decoys,
# the alpha -> D calibration table, and a few demo decoys with diagnostics plots.
#
#   sbatch scripts/calibrate_and_demo.sbatch            (from the repo root)
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoygen-calib
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load Miniforge3/25.3.1-0 >/dev/null 2>&1
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
RUN=runs/calib-${SLURM_JOB_ID}
mkdir -p "$RUN"
nvidia-smi -L

echo "== timing: default 128^3 decoy on GPU (theoretical alpha, before calibration)"
"$ENV/bin/decoygen" generate --device cuda --out "$RUN/pre" --seed 1 --plots --extra-maps

echo "== calibration"
"$ENV/bin/decoygen" calibrate --device cuda --seeds 4 --out "$RUN/alpha_to_D.json"

echo "== demo decoys with calibrated alpha"
cp "$RUN/alpha_to_D.json" decoygen/data/alpha_to_D.json
"$ENV/bin/decoygen" batch --device cuda --out "$RUN/demo" --n 4 --start-seed 0 --plots
for D in 2.1 2.4; do
  "$ENV/bin/decoygen" generate --device cuda --out "$RUN/demo_D$D" --seed 0 --roughness-D "$D" --plots
done
echo "done: $RUN"
