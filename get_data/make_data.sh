#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=25G
#SBATCH -p priority
#SBATCH -t 0-04:00
#SBATCH -o slurm_jobs/makedata_%j.out
#SBATCH -e slurm_jobs/makedata_%j.err
#SBATCH --mail-user=ralphestanboulieh@hms.harvard.edu
#SBATCH --mail-type=ALL

# Inputs:
# 1) output folder for VCFs
# 2) output folder for ML-ready data and plots
# 3) gene list file (one gene per line)
# 4) ukbb data folder (organized by chromosome)
# 5) vep annotations folder (organized by chromosome)
# 6) output folder for plots

module load gcc/9.2.0 bcftools/1.14 conda3 plink2/2.0
source activate myjupyter

mkdir $1
mkdir $1/variant_ids \
        $1/refseq \
        $1/unannot_vcf \
        $1/annot_table \
        $1/annot_vcf \
        $1/annot_vcl \
        $1/sample_names
mkdir $2


python process_gene_list.py \
            --gene_list $3 \
            --output_file $1/genetable.csv

# make header file for annotations
touch $1/header.hdr
echo "##INFO=<ID=AA_POS,Number=1,Type=Integer,Description=\"Position within protein\">" > $1/header.hdr
echo "##INFO=<ID=AA_REF,Number=1,Type=String,Description=\"Reference amino acid\">" >> $1/header.hdr
echo "##INFO=<ID=AA_ALT,Number=1,Type=String,Description=\"Alternate amino acid\">" >> $1/header.hdr
echo "##INFO=<ID=AA_STRAND,Number=1,Type=String,Description=\"CDS strand\">" >> $1/header.hdr

while IFS=, read -r protein id chr; do
        echo "Making $protein annotated VCFs ..."
        bash make_annotated_vcf.sh $protein \
                                $id \
                                $chr \
                                $1/refseq \
                                $1/variant_ids \
                                $1/unannot_vcf \
                                $1/annot_table \
                                $1/header.hdr \
                                $1/annot_vcf \
                                $1/annot_vcl \
                                $1/sample_names \
                                $4 \
                                $5
        
        echo "Making $protein data and plots ..."
        mkdir $2/$protein
        python process_vcf.py --vclist $1/annot_vcl/$protein.vclist.tsv \
                        --variants_table $1/annot_vcl/$protein.var.table.tsv \
                        --output_folder $2/$protein \
                        --sample_names $1/sample_names/$protein.samples.txt \
                        --refseq_fasta $1/refseq/$protein.fasta \
                        --gene_symbol $protein \
                        --plots_folder $6
        echo "Done."
done < $1/genetable.csv
