#!/bin/bash
# Decoy ladder through DeepEMhancer (all three models), then ModelAngelo on inputs and outputs.
#   sbatch experiments/deepemhancer/run.sbatch
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=deepemhancer
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
OUT=${OUT:-runs/deepemhancer}
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
"$ENV/bin/python" experiments/deepemhancer/prepare.py --atoms runs/bookmarks/raw-polymer --out "$OUT" || exit 1

MODELS=/software/presto/e/14.1/software/deepEMhancer/0.16-foss-2023a-CUDA-12.1.1/deepEMhancerModels
for name in protein protein_mirror polymer gas hardcore; do
    for mode in wideTarget tightTarget highRes; do
        dst="$OUT/${name}_deh_${mode}.mrc"
        [ -f "$dst" ] && continue
        echo "== DeepEMhancer $name $mode"
        bash -lc "module load deepEMhancer/0.16-CUDA-12.1.1-PReSTO-14.1 >/dev/null 2>&1 && \
            deepemhancer -i $OUT/${name}_half1.mrc -i2 $OUT/${name}_half2.mrc -o $dst -p $mode \
            --deepLearningModelPath $MODELS -g 0" > "$OUT/${name}_deh_${mode}.log" 2>&1 \
            || echo "   failed (see ${dst%.mrc}.log)"
    done
done

export TORCH_HOME=/proj/liu-folab-cryoem-dev/users/x_bjfor/torch_home
MA=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/relion-5.1/bin/model_angelo
mkdir -p "$OUT/build"
for map in "$OUT"/*_full.mrc "$OUT"/*_deh_*.mrc; do
    name=$(basename "$map" .mrc)
    [ -f "$OUT/build/$name/$name.cif" ] && continue
    echo "== ModelAngelo $name"
    "$MA" build_no_seq -v "$map" -o "$OUT/build/$name" -d 0 --keep-intermediate-results \
        > "$OUT/build/$name.log" 2>&1 || echo "   no model for $name"
done

"$ENV/bin/python" experiments/deepemhancer/analyse.py "$OUT"
