#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -p priority
#SBATCH -t 0-05:00
#SBATCH -o slurm_jobs/%j.out
#SBATCH -e slurm_jobs/%j.err
#SBATCH --mail-user=ralphestanboulieh@hms.harvard.edu
#SBATCH --mail-type=ALL

module load gcc/9.2.0 bcftools/1.14 conda3 plink2/2.0
ukbbdir=/n/groups/marks/databases/ukbiobank/ukbb_450k

bash make_data.sh ../data/temp \
                    ../data/data \
                    ../gene_list.txt \
                    $ukbbdir/pop_vcf \
                    $ukbbdir/vep \
                    ../figures/init_data_plots
