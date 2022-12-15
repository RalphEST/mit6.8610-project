#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -p priority
#SBATCH -t 0-06:00
#SBATCH -o slurm_jobs/%j.out
#SBATCH -e slurm_jobs/%j.err
#SBATCH --mail-user=ralphestanboulieh@hms.harvard.edu
#SBATCH --mail-type=ALL

module load conda3 gcc/6.2.0 cuda/11.2
python -u save_as_npy.py