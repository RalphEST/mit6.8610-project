print("Importing modules ...", end=" ")

# basics
import os, sys, argparse, requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# custom
import plotting_utils
import mutate

# fasta sequence processing
from Bio import SeqIO, Seq, Data

# nice plotting
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans serif"

print("Done.")

# getting arguments and inputs
parser = argparse.ArgumentParser(description='Process tab-delimited simplified VCF')
parser.add_argument('--vclist', type=str, help='Path to the VCList file')
parser.add_argument('--variants_table', type=str, help='Path to the variants table')
parser.add_argument('--sample_names', type=str, help='Path to the samples names text file')
parser.add_argument('--output_folder', type=str, help='Path to the output data folder')
parser.add_argument('--refseq_fasta', type=str, help='Path to the RefSeq FASTA file')
parser.add_argument('--gene_symbol', type=str, help='Name of the sequence')
parser.add_argument('--plots_folder', type=str, help='Path to summary plot file')
args = parser.parse_args()

print("Reading sample names ...", end=" ") 

# getting sample names
with open(args.sample_names, "r") as f:
    for line in f.readlines():
        if line[0:2]!="##":
            header = line

sample_names = header.split('\t')

# find the first sample ID (to skip the other VCF columns)
for idx,s in enumerate(sample_names):
    if re.search("\d+(_\d+)+", s): break
    
sample_names = header.split('\t')[idx:]
sample_names = [x.split('_')[0] for x in sample_names]
n_samples = len(sample_names)

print("Done.")

print("Reading and processing variants table and VCList ...", end=" ")

variants = pd.read_table(args.variants_table, 
                     sep = '\t')

# reindexing starting with 1 (leaving 0 for WT)
variants = variants.set_index('ID')
variants.index.names = ['var_id']
variants = variants[variants['AC']>0.0]
n_vars = len(variants)

# read VCList of variant samples
var_samples = pd.read_table(args.vclist, 
                            sep = '\t')

## simplify sample names
var_samples['SAMPLE'] = var_samples['SAMPLE'].str.split('_', expand = True)[0]

## 0/1, 1/0, 1/1 --> 1
var_samples['GT'] = 1
var_samples = var_samples.astype({'GT':np.byte})

# turn VCList into matrix
var_samples = var_samples.pivot(index = 'SAMPLE', columns = 'ID', values = 'GT')\
                            .fillna(0).astype(np.byte)

# number of variants per sample
nvars_per_sample = var_samples.sum(1).astype(np.uint8)

print(f"Done. Found {n_vars} variants and {n_samples} samples.")

print("Selecting and ID-ing variant sequences ...", end=" ")

# create sample-wise IDs to speed-up sample-sequence matching
# ids are "[variant 1 index]_[variant 2 index]_ ..."
var_id_num = np.arange(1, n_vars+1)
var_ids = variants.index.to_list()
var_samples_ids = (var_samples * var_id_num).astype('int') \
                                        .replace({0:None}) \
                                        .agg(lambda x: '_'.join([var_ids[y-1] for y in filter(None, x)]), axis = 1) \
                                        .to_frame(name = 'vars_list') \
                                        .astype({'vars_list': 'category'})

var_samples_ids['n_vars'] = nvars_per_sample.loc[var_samples_ids.index]

print("Done.")

print("Creating sequences table ...", end=" ")

# creating sequences table
seq_table = var_samples_ids.copy(deep=True).drop_duplicates()
n_unique_var_seqs = len(seq_table)

# ID-ing unique sequence
seq_table.index = [f'seq_{i}' for i in range(1, n_unique_var_seqs + 1)]

# adding back reference sequence as 'seq_0'
# the reference sequence has vars_list '0'
seq_table = pd.concat((pd.DataFrame({'vars_list': '0', 'n_vars':0}, 
                                    index=['seq_0']), 
                       seq_table))

# reorganizing and saving space
seq_table['n_vars'] = seq_table['n_vars'].astype(np.uint8)
seq_table.index.name = 'seq_id'
seq_table = seq_table.reset_index()

print(f"Done. Found {n_unique_var_seqs} unique variant sequences.")

print("Creating patients table ...", end=" ")

# initializing patients table
patients_table = pd.DataFrame({'patient_id': sample_names, 
                               'n_vars': 0, 
                               'vars_list': '0'}) 


# instead of computing the variant index-based sequence ids, substitute in the ones already computed
# the rest is set to '0' by default
patients_table = patients_table.set_index('patient_id')
patients_table.loc[var_samples_ids.index, ['vars_list', 'n_vars']] = var_samples_ids[['vars_list', 'n_vars']]

# assigning sequence ids to patients
patients_table = pd.merge(left = patients_table.reset_index(), 
                         right = seq_table[['seq_id', 'vars_list']], 
                         left_on = 'vars_list', 
                         right_on = 'vars_list', 
                         how = 'left')

# reorganizing
patients_table = patients_table.astype({'n_vars':np.uint8, 
                                    'vars_list':'category', 
                                    'seq_id':'category'})

print("Done.")

print("Creating sequence-variant matrix ...", end=" ")

seq_var_matrix = np.vstack((np.zeros((1, n_vars)), 
                            var_samples.drop_duplicates().to_numpy()))

print("Done.")

print("Counting sequences ...", end=" ")

# counting unique sequences in population
seq_counts = patients_table['seq_id'].value_counts().rename('seq_count')
seq_counts = seq_counts.astype('int32')
seq_table = pd.merge(left = seq_table, 
                     right = seq_counts, 
                     left_on = 'seq_id', 
                     right_index = True, 
                     how = 'left')

patients_table = pd.merge(left = patients_table, 
                         right = seq_counts, 
                         left_on = 'seq_id', 
                         right_index = True, 
                         how = 'left')

# reorganizing
patients_table = patients_table.set_index('patient_id')
seq_table = seq_table.set_index('seq_id')

print("Done.")

# creating sequence data and fasta files
# read in the RefSeq sequence
refseq = SeqIO.read(args.refseq_fasta, "fasta")
seq_data, seq_fasta = mutate.mutate(args.gene_symbol, refseq, seq_table, variants)


print("Saving data ...", end=" ")

# saving data
patients_table.to_parquet(os.path.join(args.output_folder, 'patients_table.parquet'))
seq_table.to_parquet(os.path.join(args.output_folder, 'seq_table.parquet'))
variants.to_parquet(os.path.join(args.output_folder, 'variants_table.parquet'))
np.save(os.path.join(args.output_folder, 'seq_data.npy'), seq_data)
np.save(os.path.join(args.output_folder, 'seq_var_matrix.npy'), seq_var_matrix)

with open(os.path.join(args.output_folder, 'seqs.fasta'), 'w') as file:
    for idx, seq in enumerate(seq_fasta):
        file.write(f">seq_{idx} | {seq_table.iloc[idx,0]}\n")
        file.write(seq+"\n")

print("Done.")

print("Plotting ...", end=" ")

fig = plt.figure(figsize = (6*1.35, 2.1*1.3), constrained_layout=True)
subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1,2])

plotting_utils.plot_variation_summary(args.gene_symbol,      
                       variants,
                       patients_table,
                       refseq,
                       subfigs)
fig.savefig(os.path.join(args.plots_folder, args.gene_symbol + ".summary.png"))

fig, ax = plt.subplots(figsize = (3.5,3.5), constrained_layout=True)

plotting_utils.plot_sequence_variation_content(args.gene_symbol, 
                                seq_table,
                                ax)

fig.savefig(os.path.join(args.plots_folder, args.gene_symbol + ".step.png"))

print("Done.")
print("#####")