#!/bin/bash
#SBATCH --job-name=lunar_lc_cvar_pg
#SBATCH --output=lunar_%j.out
#SBATCH --error=lunar_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=general-compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# SBATCH --gres=gpu:1    # uncomment if you want GPU

module load anaconda  # or the right module on NJIT

source activate lc_cvar_pg   # or `conda activate lc_cvar_pg`

cd $SLURM_SUBMIT_DIR

python lunar_experiments.py