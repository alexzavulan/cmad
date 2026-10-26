#!/bin/bash
# Fill the feature cache for the core tier: one backbone, a list of layers, all 15 categories, one GPU.
# Layers already in the cache are skipped, so a resubmitted job continues where it stopped.
# Submit from the front-end with, for example:
#   sbatch --job-name=extract_b --time=03:00:00 slurm/extract_core.sbatch dinov2_vitb14 "last_raw 8_raw"
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
# #SBATCH directives do not expand $HOME, so this is the user's actual cluster home
#SBATCH --output=/ceph/home/student.aau.dk/rx38fx/cmad-data/results/logs/%x-%j.out

BACKBONE=$1
LAYERS=$2
SIF=/ceph/container/pytorch/pytorch_26.08.sif

singularity exec --nv --bind "$HOME" "$SIF" bash -c "
cd \$HOME/cmad && source .venv/bin/activate
export CMAD_DATA_ROOT=\$HOME/cmad-data TORCH_HOME=\$HOME/torch_home OMP_NUM_THREADS=15
python -m cmad.features extract --backbone $BACKBONE --layers $LAYERS --tier core 2>&1 | grep -v -i warn
"
