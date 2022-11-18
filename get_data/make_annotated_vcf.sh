#!/bin/bash

# Complete pipeline from a population-level VCF and a VEP annotation table to
# a VEP-annotated VCF file.

# Notes: 
# ------------------------
# 1) Run this script in a SLURM environment with enough RAM (on the order of 50G)
# 2) Make sure to load the modules:
#           module load conda3 plink2/2.0 gcc/9.2.0 bcftools/1.14
# 3) Make sure to activate a conda environment (python scripts are used):
#           source activate mypythonenv
# 4) Make sure efetch is installed and added to the PATH variable

# Arguments:
# ------------------------
# 1) protein name
# 2) protein RefSeq ID
# 3) protein chromosome
# 4) folder for reference FASTA sequences
# 5) folder for variant IDs text file
# 6) folder for region and ID-filtered VCF files
# 7) folder for annotation tables
# 8) header file for VCF annotation
# 9) folder for final annotated VCF files
# 10) folder for VCL and variants table files
# 11) folder for VCF header (to extract sample names)
# 12) VEP annotation folder
# 13) UKBB data folder

protein=$1
id=$2
chr=$3
refseq_folder=$4
variant_ids_folder=$5
unannot_vcf_folder=$6
annot_table_folder=$7
header=$8
annot_vcf_folder=$9
annot_vclist_folder=${10}
vcf_header_folder=${11}
pop_vcf_folder=${12}
vep_folder=${13}

# vep_folder=../../ukbb_450k/vep
# pop_vcf_folder=../../ukbb_450k/pop_vcf

variant_ids_file=$variant_ids_folder/$protein.variant.ids.txt
region_filtered_vcf_file=$unannot_vcf_folder/$protein.region.filtered.vcf.gz
id_filtered_vcf_file=$unannot_vcf_folder/$protein.id.filtered
unannot_vcf_file=$unannot_vcf_folder/$protein.unannot.vcf.gz
unproc_annot_table=$annot_table_folder/$protein.unproc.annot.tab
annot_table=$annot_table_folder/$protein.annot.tab
annot_vcf_file=$annot_vcf_folder/$protein.annot.poplevel.vcf.gz
vclist_file=$annot_vclist_folder/$protein.vclist.tsv
var_table_file=$annot_vclist_folder/$protein.var.table.tsv
vcf_header_file=$vcf_header_folder/$protein.samples.txt

echo "Fetching $protein RefSeq FASTA file ..."
efetch -db protein -id $id -format fasta > $refseq_folder/${protein}.fasta

# Preparing annotation tables and generating variant ids
echo "Preparing annotation table ..."
grep -P "#Uploaded_variation|${id}" \
        $vep_folder/ukb23149_c${chr}_b0_v1.coding.vep.tsv > $unproc_annot_table
variants_range=$(python clean_annot.py $id $unproc_annot_table $annot_table $variant_ids_file)
bgzip --keep $annot_table
tabix -s 1 -b 2 -e 2 $annot_table.gz

# clean-up
rm $unproc_annot_table

# Filtering population-level VCF file using genomic region and variant IDs
echo "Filtering and tagging pop-level VCF ..."
bcftools view -Oz -r $chr:$variants_range \
                $pop_vcf_folder/ukb23149_c${chr}_b0_v1.filtered.vcf.gz > $region_filtered_vcf_file

plink2 --vcf $region_filtered_vcf_file \
		--extract $variant_ids_file \
		--export vcf bgz \
		--out $id_filtered_vcf_file \
		--id-delim '_'

## computing AN, AC, and AF tags
bcftools +fill-tags $id_filtered_vcf_file.vcf.gz -Oz -o $unannot_vcf_file -- -t AN,AC,AF

echo "Top 5 rows:"
bcftools query -f '%CHROM %POS %REF %ALT %AN %AC %AF\n' $unannot_vcf_file | head -5
echo "*****"

## clean-up
rm $id_filtered_vcf_file.log $id_filtered_vcf_file.vcf.gz $region_filtered_vcf_file

# Annotating population-level VCF file
echo "Annotating pop-level VCF file ..."
bcftools annotate -a $annot_table.gz \
                -h $header \
                -c CHROM,POS,ID,REF,ALT,INFO/AA_POS,INFO/AA_STRAND,INFO/AA_REF,INFO/AA_ALT \
                -Oz -o $annot_vcf_file \
                $unannot_vcf_file

echo "Top 5 variants:"
bcftools query -f "%ID %AC %AA_POS %AA_STRAND %AA_REF %AA_ALT\n" $annot_vcf_file | head -5
echo "*****"

echo "Creating variant table file ..."
echo -e "ID\tAA_POS\tAA_STRAND\tAA_REF\tAA_ALT\tAC\tAF" > $var_table_file
bcftools query -f "%ID\t%AA_POS\t%AA_STRAND\t%AA_REF\t%AA_ALT\t%AC\t%AF\n" $annot_vcf_file >> $var_table_file

echo "Creating variant call list ..."
echo -e "ID\tSAMPLE\tGT" > $vclist_file
bcftools query -f'[%ID\t%SAMPLE\t%GT\n]' -i'GT="alt"' $annot_vcf_file >> $vclist_file

echo "Creating VCF header file (to extract sample names) ..."
bcftools view -h $annot_vcf_file > $vcf_header_file
