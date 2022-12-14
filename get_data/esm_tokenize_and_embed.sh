#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 0-06:00
#SBATCH -o slurm_jobs/%j.out
#SBATCH -e slurm_jobs/%j.err
#SBATCH --mail-user=ralphestanboulieh@hms.harvard.edu
#SBATCH --mail-type=ALL

module load conda3 gcc/6.2.0 cuda/11.2
python -u esm_tokenize_and_embed.py