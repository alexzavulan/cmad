#!/bin/bash
# Job-array template, filled in by `python -m cmad.sweep experiment=<name> launcher=slurm`.
# One array task = one line of the task list (a category); its seeds, regimes and sources run
# inside the task. Finished runs are skipped, so a killed task is resubmitted and resumes.
#SBATCH --job-name={{experiment}}
#SBATCH --partition=l4
#SBATCH --array=0-{{n_tasks_minus_1}}%8
#SBATCH --gres=gpu:1
# sized for ViT-S and WRN-50 on one L4; revisit --mem for the larger backbones
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --time={{walltime}}
# #SBATCH lines do not expand $HOME, so sweep renders the absolute path here
#SBATCH --output={{results_dir}}/logs/%x-%a.out

export CMAD_DATA_ROOT={{data_root}}
export TORCH_HOME="$HOME/torch_home"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {{task_list}})   # e.g. "category=screw"
SIF=/ceph/container/pytorch/pytorch_26.08.sif

singularity exec --nv --bind "$HOME" "$SIF" bash -c '
cd {{repo}} && source .venv/bin/activate
python -m cmad.sweep experiment={{experiment}} launcher=local sweep_dir={{sweep_dir}} $TASK
'
