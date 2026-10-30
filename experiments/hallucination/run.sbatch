#!/bin/bash
# Hallucination pilot, one mode per job:
#   MODE=controls sbatch experiments/hallucination/run.sbatch
#   MODE=relax ARGS="--dmin 2.2" sbatch experiments/hallucination/run.sbatch
#   MODE=optimise ARGS="--dmin 2.2" sbatch experiments/hallucination/run.sbatch
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --job-name=hallucinate
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
OUT=${OUT:-runs/hallucination}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" experiments/hallucination/hallucinate.py "$MODE" --out "$OUT" ${ARGS:-}
