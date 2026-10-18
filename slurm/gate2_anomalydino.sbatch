#!/bin/bash
# Gate 2: our `anomalydino` recipe vs the published AnomalyDINO numbers (spec sec 13).
# 15 MVTec AD categories x {1, 16} shots x seeds 0-2 (the official contiguous-block references),
# one GPU, sequential. Completed runs are skipped, so a resubmitted job continues where it stopped.
# Submit from the front-end with:  sbatch slurm/gate2_anomalydino.sbatch
#SBATCH --job-name=gate2_ad
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# #SBATCH directives do not expand $HOME, so this is the user's actual cluster home
#SBATCH --output=/ceph/home/student.aau.dk/rx38fx/cmad-data/results/logs/%x-%j.out

SIF=/ceph/container/pytorch/pytorch_26.08.sif

singularity exec --nv --bind "$HOME" "$SIF" bash -c '
cd $HOME/cmad && source .venv/bin/activate
export CMAD_DATA_ROOT=$HOME/cmad-data TORCH_HOME=$HOME/torch_home OMP_NUM_THREADS=8
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
for shots in 1 16; do
  for seed in 0 1 2; do
    for c in $CATS; do
      python -m cmad.run category=$c seed=$seed scorer=anomalydino layers=last_raw \
        interpolation=bicubic roi=pca_official_mvtec reference_rotations=official8 \
        regime.name=few_norm_k$shots regime.shot_selection=contiguous_blocks 2>&1 \
        | grep -v -i warn | tail -1 | cut -c1-140
    done
  done
done
echo "== summary"
python scripts/gate2_summary.py --runs $CMAD_DATA_ROOT/results/runs \
  --official $HOME/anomalydino-ref/repo/results_MVTec/dinov2_vits14_448
'
