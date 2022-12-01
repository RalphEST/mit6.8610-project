# Pseudocode:
# ------------
# > Import modules.
# > Read in inputs.
# > Read sample names files, variants table, and variant call list (VCL).
# > Create and clean variants table.
# > Pivot the VCL and create the sequence-variant matrix.
# > Create sequences table. The vars_list column will have a '_'-joined list of IDs of the variants in a particular sequence. Haplotype 0 is considered the main one.
# > Create patients table.
# > Count sequences.
# > If data is unphased, check if 'automatic phasing' is needed.
#   By default, all variants are assumed to be on the same haplotype. 
#   If two variants affect the same position (or codon), then automatic 
#   phasing is needed; the less common variant is assigned to haplotype 0, 
#   while the other is assigned to haplotype 1. (If both variants have 
#   the same AC, then they are randomly phased.) This is so that no variant
#   left behind.
# > If data is phased (or requires automatic phasing), split the vars_list column into two haplotypes. 
# > Mutate the reference sequence to create the matrix and FASTA data.
# > Save all the data. 

print("Importing modules ... ", end="")

# basics
import os, sys, argparse
import pandas as pd
import numpy as np
import re

# fasta sequence processing
from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord

# custom module(s)
UTILS_PATH = '/'.join(os.getcwd().split('/')[:-1] + ['utils'])
sys.path.insert(1, UTILS_PATH)
import mutate
import plotting_utils

# plotting
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans serif"

print("Done.")

# getting arguments and inputs
parser = argparse.ArgumentParser(description='Process tab-delimited simplified VCF')
parser.add_argument('--vclist', type=str, help='Path to the VCList file')
# parser.add_argument('--phased', type=bool, help='Whether the variant data is phased or not')
parser.add_argument('--variants_table', type=str, help='Path to the variants table')
parser.add_argument('--sample_names', type=str, help='Path to the samples names text file')
parser.add_argument('--output_folder', type=str, help='Path to the output data folder')
parser.add_argument('--refseq_fasta', type=str, help='Path to the RefSeq FASTA file')
parser.add_argument('--gene_symbol', type=str, help='Name of the sequence')
parser.add_argument('--plots_folder', type=str, help='Path to summary plot file')
args = parser.parse_args()

gene_symbol = args.gene_symbol
sample_names = args.sample_names
vclist = args.vclist
# phased = args.phased
variants_table = args.variants_table
output_folder = args.output_folder
plots_folder = args.plots_folder
refseq_fasta = args.refseq_fasta

print("Reading and processing inputs ...", end=" ") 

# read and counting sample names
with open(sample_names, "r") as f:
    for line in f.readlines():
        if line[0:2]!="##":
            header = line

sample_names = header.split('\t')

## find the first sample ID (to skip the other VCF columns)
for idx,s in enumerate(sample_names):
    if re.search("\d+(_\d+)+", s): break
    
sample_names = header.split('\t')[idx:]
sample_names = [x.split('_')[0] for x in sample_names]
n_samples = len(sample_names)

# read variants table
variants = pd.read_table(variants_table, 
                         sep = '\t')
variants = variants.set_index('ID')
variants.index.names = ['var_id']

## filter out variants with count 0
variants = variants[variants['AC']>0.0]
n_vars = len(variants)
var_ids = variants.index

# read variant call list (VCL)
var_samples = pd.read_table(vclist, 
                            sep = '\t')

## simplify sample names
var_samples['SAMPLE'] = var_samples['SAMPLE'].str.split('_', expand = True)[0]
var_samples['GT'] = var_samples['GT'].replace({'1/0':1, '0/1':1, '1/1': 2})
var_samples = var_samples.astype({'GT':np.byte})

print("Done.")

print("Creating sequence table ...", end=" ")

## pivot var_samples from long to wide data format
var_samples_wide = var_samples.pivot(index = 'SAMPLE', 
                                     columns = 'ID', 
                                     values = 'GT')\
                              .fillna(0).astype(np.byte)

## concatenate variant ids to generate new sequence-specific ids 
def concatenate_var_ids(row):
    return('_'.join([f"{v}#{k}" for k,v in row[row>0].items()]))
var_samples_wide["vars_list"] = var_samples_wide.apply(concatenate_var_ids, axis = 1)
var_samples_wide["n_vars"] = var_samples_wide.loc[:,var_ids]\
                                             .sum(1).astype(np.uint8)

## create table of unique sequences
seq_table = var_samples_wide.copy(deep=True)\
                            .drop_duplicates()

## create sequence-variant matrix 
seq_var_matrix = seq_table.loc[:, var_ids].to_numpy()

## reorganize sequence table
seq_table = seq_table[["vars_list", "n_vars"]]
seq_table.index = [f'seq_{i}' for i in range(1, len(seq_table) + 1)]
seq_table = pd.concat((pd.DataFrame({'vars_list': '0', 'n_vars':0}, 
                                    index=['seq_0']), 
                       seq_table))
seq_table.index.name = 'seq_id'
n_unique_seqs = len(seq_table)

print("Done.")

print("Phasing ...", end=" ")

def hacky_phase(vars_list):
    if vars_list == '0':
        return '0|0'
    vars_list = [[v.split('#')[1]]*int(v.split('#')[0]) for v in vars_list.split("_")]
    vars_list = [x for v in vars_list for x in v]
    vars_df = variants.loc[vars_list].reset_index().sort_values(by=["AA_POS", "AC"])
    vars_df['CHROM'] = vars_df['var_id'].str.split(":", expand=True)[1]
    hap0, hap1 = [], []
    for pos, df in vars_df.groupby('AA_POS'):
        if len(df)==1:
            hap0 += [df['var_id'].item()]
        else:
            hap0 += df.drop_duplicates(subset = ["AA_POS"] if len(df)==2 else ["CHROM", "AA_POS"], 
                                       keep='first')["var_id"].to_list()
            hap1 += df[df.duplicated(subset = ["AA_POS"] if len(df)==2 else ["CHROM", "AA_POS"], 
                                     keep='first')]["var_id"].to_list()
    # hap0 = vars_df.drop_duplicates(subset = ["CHROM", "AA_POS"], keep='first')["var_id"].to_list()
    # hap1 = vars_df[vars_df.duplicated(subset = ["CHROM", "AA_POS"], keep='first')]["var_id"].to_list()
    return '|'.join(['_'.join(hap0), '_'.join(hap1) if len(hap1)>0 else '0'])
    
seq_table['phased_vars_list'] = seq_table['vars_list'].apply(hacky_phase)
seq_table[["vars_list_0", "vars_list_1"]] = seq_table['phased_vars_list'].str.split('|', expand = True)

print("Done.")

print("Creating patients table ...", end=" ")
# initializing patients table
patients_table = pd.DataFrame({'patient_id': sample_names, 
                               'n_vars': 0, 
                               'vars_list': '0'}) 

# instead of computing the variant index-based sequence ids, substitute in the ones already computed
# the rest is set to '0' by default
patients_table = patients_table.set_index('patient_id')
patients_table.loc[var_samples_wide.index, ['vars_list', 'n_vars']] = var_samples_wide[['vars_list', 'n_vars']]

# assigning sequence ids to patients
patients_table = pd.merge(left = patients_table.reset_index(), 
                         right = seq_table.reset_index()[['seq_id', 'vars_list', 'phased_vars_list']], 
                         left_on = 'vars_list', 
                         right_on = 'vars_list',
                         how = 'left')
# reorganizing
patients_table = patients_table.astype({'n_vars':np.uint8, 
                                        'phased_vars_list':'category',
                                        'seq_id':'category'})

print("Done.")

print("Counting sequences ... ", end="")

# counting unique sequences in population
seq_counts = patients_table['seq_id'].value_counts().rename('seq_count')

# adding counts to sequence table
seq_table = pd.merge(left = seq_table, 
                     right = seq_counts, 
                     left_index = True, 
                     right_index = True, 
                     how = 'left')

# adding counts to patients table
patients_table = pd.merge(left = patients_table, 
                         right = seq_counts, 
                         left_on = 'seq_id', 
                         right_index = True, 
                         how = 'left')
patients_table = patients_table.set_index('patient_id')

print("Done.")

print("Create haplotypes table ...", end=" ")

haplos = list(set(seq_table['vars_list_0'].to_list() + seq_table['vars_list_1'].to_list()))
hap_table = pd.DataFrame({'hap_id': [f"hap_{i}" for i in range(len(haplos))],
              'hap_vars': sorted(haplos)})

seq_table = pd.merge(left = seq_table.reset_index(),
         right = hap_table.rename(columns={'hap_id':'hap0'}),
         left_on = 'vars_list_0',
         right_on = 'hap_vars',
         how = 'left')

seq_table = pd.merge(left = seq_table,
         right = hap_table.rename(columns={'hap_id':'hap1'}),
         left_on = 'vars_list_1',
         right_on = 'hap_vars',
         how = 'left')

hap_table = hap_table.set_index('hap_id')
seq_table = seq_table.set_index('seq_id')
seq_table = seq_table[['phased_vars_list', 'vars_list', 'n_vars', 'seq_count', 'hap0', 'hap1']]

print("Done.")

print("Creating variant sequence data ...", end=" ")

refseq = SeqIO.read(refseq_fasta, "fasta")
hap_data, hap_list, hap_indicator = mutate.mutate(refseq, hap_table, variants)

print("Done.")

print("Saving data ...", end=" ")

# saving tables
patients_table.to_parquet(os.path.join(output_folder, 'patients_table.parquet'))
seq_table.to_parquet(os.path.join(output_folder, 'seq_table.parquet'))
hap_table.to_parquet(os.path.join(output_folder, 'hap_table.parquet'))
variants.to_parquet(os.path.join(output_folder, 'variants_table.parquet'))

# saving matrices
np.save(os.path.join(output_folder, 'hap_data.npy'), hap_data)
np.save(os.path.join(output_folder, 'seq_var_matrix.npy'), seq_var_matrix)
np.save(os.path.join(output_folder, 'hap_indicator.npy'), hap_indicator)

# saving fasta file
SeqIO.write(hap_list, os.path.join(output_folder, 'hap_fasta.fasta'), "fasta")

print("Done.")

print("Plotting ...", end=" ")

fig = plt.figure(figsize = (6*1.35, 2.1*1.3), constrained_layout=True)
subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1,2])

plotting_utils.plot_variation_summary(gene_symbol,      
                       variants,
                       patients_table,
                       refseq,
                       subfigs)
fig.savefig(os.path.join(plots_folder, gene_symbol + ".summary.png"))

fig, ax = plt.subplots(figsize = (3.5,3.5), constrained_layout=True)

legend_kwargs = {'fontsize': 9, 
                 'labelspacing': 0.2, 
                 'ncol':1}
plotting_utils.plot_sequence_variation_content(gene_symbol, 
                                seq_table,
                                ax, legend_kwargs)

fig.savefig(os.path.join(plots_folder, gene_symbol + ".step.png"))

print("Done.")
