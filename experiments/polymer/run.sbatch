#!/bin/bash
# Atom-model comparison (protein / gas / hard-core / polymer); no ModelAngelo.
#   sbatch experiments/polymer/run.sbatch
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=polymer-compare
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
"$ENV/bin/python" experiments/polymer/compare.py --out "${OUT:-runs/polymer}" ${ARGS:-}
