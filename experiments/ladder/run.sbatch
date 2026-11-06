#!/bin/bash
# Every test on every rung of runs/ladder that doesn't have it yet, then the summary.
#   sbatch experiments/ladder/run.sbatch [rung dirs...]
# 1. builds the atoms of rungs made from others (threaded chains), 2. collect.py,
# 3. per rung: evaluation maps, DeepEMhancer x3, C-alpha CNN in ModelAngelo's frame,
#    ModelAngelo build_no_seq on each map, 4. summarise.py.
#
#SBATCH --account=berzelius-2026-172
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --job-name=ladder
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
ENV=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/decoy
PY="$ENV/bin/python"
ROOT=runs/ladder

# 1. derived atoms
[ -f $ROOT/build/polymer_chain_split5_prune4.pdb ] || "$PY" experiments/hallucination/thread.py --split 5 \
    runs/bookmarks/raw-polymer/polymer_atoms.pdb --fill --prune 4 --relax 3000 --out $ROOT/build
B5=runs/hallucination/decoy_polymer_soft_cw0.2_lr0.15_p4
[ -f ${B5}_chain_prune4.pdb ] || "$PY" experiments/hallucination/thread.py ${B5}_atoms.pdb \
    --fill --prune 4 --relax 3000

# 2. layout
"$PY" experiments/ladder/collect.py || exit 1

# 3. tests
MODELS=/software/presto/e/14.1/software/deepEMhancer/0.16-foss-2023a-CUDA-12.1.1/deepEMhancerModels
export TORCH_HOME=/proj/liu-folab-cryoem-dev/users/x_bjfor/torch_home
MA=/proj/liu-folab-cryoem-dev/users/x_bjfor/.conda/envs/relion-5.1/bin/model_angelo
build() {   # build MAP DIR: ModelAngelo build_no_seq into DIR (DIR/<name>/<name>.cif)
    local map=$1 dir=$2 name
    name=$(basename "$dir")
    [ -f "$dir/$name.cif" ] && return
    echo "   ModelAngelo $dir"
    mkdir -p "$(dirname "$dir")"
    "$MA" build_no_seq -v "$map" -o "$dir" -d 0 --keep-intermediate-results > "$dir.log" 2>&1 \
        || echo "   no model ($dir.log)"
}
for d in ${@:-$ROOT/main/* $ROOT/branch/*}; do   # rung dirs given as arguments, or all
    atoms=$(ls "$d"/atoms.* 2>/dev/null | head -1)
    [ -n "$atoms" ] || { echo "== $d: no atoms yet"; continue; }
    echo "== $d"
    [ -f "$d/maps/truth.mrc" ] || "$PY" experiments/ladder/maps.py "$d"
    for mode in wideTarget tightTarget highRes; do
        dst="$d/maps/deh_${mode}.mrc"
        [ -f "$dst" ] && continue
        echo "   DeepEMhancer $mode"
        bash -lc "module load deepEMhancer/0.16-CUDA-12.1.1-PReSTO-14.1 >/dev/null 2>&1 && \
            deepemhancer -i $d/maps/half1.mrc -i2 $d/maps/half2.mrc -o $dst -p $mode \
            --deepLearningModelPath $MODELS -g 0" > "$d/maps/deh_${mode}.log" 2>&1 \
            || echo "   DeepEMhancer $mode failed (${dst%.mrc}.log)"
    done
    [ -f "$d/detector/rung_metrics.json" ] || "$PY" experiments/hallucination/hallucinate.py atoms \
        --pdb "rung=$atoms" --out "$d/detector"
    for v in full deh_wideTarget deh_tightTarget deh_highRes; do
        [ -f "$d/maps/$v.mrc" ] && build "$d/maps/$v.mrc" "$d/modelangelo/$v"
    done
    build "$d/detector/rung_full.mrc" "$d/detector/build/rung"
done

# 4. summary
"$PY" experiments/ladder/summarise.py
