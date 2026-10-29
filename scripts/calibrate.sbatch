#!/bin/bash
# Build the (alpha, roughness amplitude) -> D calibration table on a GPU node.
#   sbatch scripts/calibrate.sbatch            (from the repo root)
# Writes runs/calib-<jobid>/roughness_calibration.json; review it, then copy it to
# decoygen/data/roughness_calibration.json and commit.
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
"$ENV/bin/decoygen" calibrate --device cuda --seeds 4 --out "$RUN/roughness_calibration.json"
echo "done: $RUN"
