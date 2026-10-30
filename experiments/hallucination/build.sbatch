#!/bin/bash
# Full ModelAngelo build_no_seq on every NAME_full.mrc in $OUT, then summarise.
#   OUT=runs/hallucination sbatch experiments/hallucination/build.sbatch
# Uses the relion-5.1 env (ModelAngelo 1.0.18) read-only; weights from TORCH_HOME.
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=ma-build
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
OUT=${OUT:-runs/hallucination}
export TORCH_HOME=/proj/liu-folab-cryoem-dev/users/x_bjfor/torch_home
MA=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/relion-5.1/bin/model_angelo
mkdir -p "$OUT/build"
for map in "$OUT"/*_full.mrc; do
    name=$(basename "$map" _full.mrc)
    dst="$OUT/build/$name"
    [ -f "$dst/$name.cif" ] && continue
    echo "== $name"
    start=$(date +%s)
    "$MA" build_no_seq -v "$map" -o "$dst" -d 0 --keep-intermediate-results > "$dst.log" 2>&1 \
        || echo "build failed for $name (see $dst.log)"
    echo "   $(( $(date +%s) - start )) s"
done
/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy/bin/python \
    experiments/hallucination/summarise.py "$OUT"
