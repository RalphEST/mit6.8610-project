print("Importing modules ... ", end="")

# basics
import os, sys, argparse, requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# fasta sequence processing
from Bio import SeqIO, Seq, Data

# efficient data saving and processing
import polars as pl
import h5py
import pyarrow.parquet as pq

# nice plotting
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans serif"

print("Done.")

if __name__=="__main__":
    '''
    Inputs:
    1) the name of the protein (for plotting)
    2) path to the header text file (to get sample names)
    3) path to the 'simplified VCF,' a tab-delimited file with the columns:
        ID, POS, REF, ALT, AF, AC, [SAMPLES]
       such a file can be prodcued using bcftools query
    4) path to output data folder
    5) path to reference sequence file 
    6) path for the summary plot 
    '''

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
    
    print("Reading sample names ... ", end="") 
    
    # getting sample names
    with open(args.sample_names, "r") as f:
        for line in f.readlines():
            if line[0:2]!="##":
                header = line
    sample_names = header.split('\t')[9:]
    sample_names = [x.split('_')[0] for x in sample_names]
    n_samples = len(sample_names)
    
    print("Done.")

    print("Reading and processing variants table and VCList ... ", end="")

    variants = pd.read_table(args.variants_table, 
                         sep = '\t')
    n_vars = len(variants)
    
    # reindexing starting with 1 (leaving 0 for WT)
    var_ids = np.arange(1, n_vars + 1, 1)
    variants['idx'] = var_ids
    
    # read VCList of variant samples
    var_samples = pd.read_table(args.vclist, 
                                sep = '\t')

    ## simplify sample names
    var_samples['SAMPLE'] = var_samples['SAMPLE'].str.split('_', expand = True)[0]

    ## 0/1, 1/0, 1/1 --> 1
    var_samples['GT'] = 1
    var_samples = var_samples.astype({'GT':np.byte})
    
    # simplify variant IDs by replacing them with their indexes in the variants table
    var_samples = pd.merge(left = var_samples, 
                           left_on='ID', 
                           right = variants[['idx', 'ID']], 
                           right_on = 'ID', 
                           how = 'left')
    
    # turn VCList into matrix
    var_samples = var_samples.pivot(index = 'SAMPLE', columns = 'idx', values = 'GT')\
                                .fillna(0).astype(np.byte)

    # number of variants per sample
    nvars_per_sample = var_samples.sum(1).astype(np.uint8)
    
    print(f"Done. Found {n_vars} variants and {n_samples} samples.")

    print("Selecting and ID-ing variant sequences ... ", end="")

    # create sample-wise IDs to speed-up sample-sequence matching
    # ids are "v:[variant 1 index].[variant 2 index] ...", indexes are sorted
    # the reference sequence has id 'v:0'
    var_samples_ids = (var_samples * var_ids).replace({0:''}) \
                                            .astype('str') \
                                            .agg(lambda x: 'v:' + '.'.join(filter(None, x)), axis = 1) \
                                            .to_frame(name = 'v:idx') \
                                            .astype({'v:idx': 'category'})

    var_samples_ids['n_vars'] = nvars_per_sample.loc[var_samples_ids.index]
    
    print("Done.")

    print("Creating sequences table ... ", end="")

    # creating sequences table
    seq_table = var_samples_ids.copy(deep=True).drop_duplicates()
    n_unique_var_seqs,_ = seq_table.shape

    # ID-ing unique sequence
    seq_table.index = [f'seq_{i}' for i in range(1, n_unique_var_seqs + 1)]

    # adding back reference sequence as 'seq_0'
    seq_table = pd.concat((pd.DataFrame({'v:idx': '0', 'n_vars':0}, index=['seq_0']), seq_table))

    # reorganizing
    seq_table['n_vars'] = seq_table['n_vars'].astype(np.uint8)
    seq_table.index.name = 'seq_id'
    seq_table = seq_table.reset_index()

    print(f"Done. Found {n_unique_var_seqs} unique variant sequences.")

    print("Creating patients table ... ", end="")

    # initializing patients table
    patients_table = pd.DataFrame({'patient_id': sample_names, 
                                   'n_vars': 0, 
                                   'v:idx': '0'}) 
    

    # instead of computing the variant index-based sequence ids, substitute in the ones already computed
    # the rest is set to '0' by default
    patients_table = patients_table.set_index('patient_id')
    patients_table.loc[var_samples_ids.index, ['v:idx', 'n_vars']] = var_samples_ids[['v:idx', 'n_vars']]

    # assigning sequence ids to patients
    patients_table = pd.merge(left = patients_table.reset_index(), 
                             right = seq_table[['seq_id', 'v:idx']], 
                             left_on = 'v:idx', 
                             right_on = 'v:idx', 
                             how = 'left')
    
    # reorganizing
    patients_table = patients_table.astype({'n_vars':np.uint8, 
                                        'v:idx':'category', 
                                        'seq_id':'category'})

    print("Done.")

    print("Counting sequences ... ", end="")

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
    variants = variants.set_index('idx')

    print("Done.")

    print("Creating sequence data ... ", end="")

    # creating sequence data
    # read in the RefSeq sequence
    refseq = SeqIO.read(args.refseq_fasta, "fasta")

    # one-hot encoding refseq sequence
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    alpha_dict = {k:v for v,k in enumerate(alphabet)}

    refseq_onehot = np.zeros((len(refseq.seq), len(alphabet))).astype(np.byte)
    for idx, letter in enumerate(refseq.seq):
        if letter in alphabet: 
            refseq_onehot[idx, alpha_dict[letter]] = 1

    seq_data = np.repeat(refseq_onehot[np.newaxis, :, :], n_unique_var_seqs + 1, axis=0)

    # turn REF and ALT into indexes to avoid recomputations
    variants['REF_idx'] = variants['AA_REF'].replace(alpha_dict)
    variants['ALT_idx'] = variants['AA_ALT'].replace(alpha_dict)

    # helper functions to deal with cases where two variants affect the same position
    # will require using the VEP API
    def vep_request(chrom, start, end, alt):
        server = "https://rest.ensembl.org"
        ext = f"/vep/human/region/{chrom}:{start}-{end}:1/{alt}?"
        r = requests.get(server+ext, headers={ "Content-Type" : "application/json", 
                                              "refseq":'True'})
        if not r.ok:
            r.raise_for_status()
            sys.exit()

        return r.json()

    # in case the two variants assign the first and third nucleotides of a codon, but not
    # the second one, we will need to determine the original codon to fill in the gaps
    def complement(s):
        s = Seq.Seq(s)
        return str(s.reverse_complement())

    codon_table = Data.CodonTable.standard_dna_table.forward_table
    codon_table_rev = {1: {aa:[v for v,k in codon_table.items() if k==aa] for aa in set(codon_table.values())}, 
                   -1: {aa:[complement(v) for v,k in codon_table.items() if k==aa] for aa in set(codon_table.values())}}
    
    requests_cache = {}

    def find_combined_effect(vars_per_pos):
        '''
        * Determine the combined effect of two SNPs affecting the same protein position
        * takes as input the result of variants.iloc[[variant_ids]]
        '''
        cache_key = tuple(vars_per_pos.index.to_list())
        if cache_key in requests_cache.keys():
            return requests_cache[cache_key]
        else:
            # use ID to get chrom, pos, ref
            df = vars_per_pos['ID'].str.split(':', expand = True).sort_values(by=1)
            chrom = df.iloc[0,0]
            start, end = df[1].min(), df[1].max()
            ref = df[2].to_list()

            # construct the alt allele
            if (int(end)-int(start)==2) and (len(df)==2):
                # only first and third codon nucleotides are known
                # will need to reconstruct full codon
                ref_aa = vars_per_pos['AA_REF'].unique()[0]
                ref_codons = codon_table_rev[strand][ref_aa]
                ori_codon = [codon for codon in ref_codons if codon[0]==ref[0] and codon[2]==ref[1]][0]
                mid_nuc = ori_codon[1]
                alt = f'{mid_nuc}'.join(df[3].to_list())
            else:
                # if two consecutive positions are affected
                # concatenating them should be enough
                alt = ''.join(df[3].to_list())

            # make VEP API request
            decoded = vep_request(chrom, start, end, alt)

            for conseq in decoded[0]['transcript_consequences']:
                if ('missense_variant' in conseq['consequence_terms']) and (conseq['gene_symbol']==args.gene_symbol):
                    vep_ref, vep_alt = conseq['amino_acids'].split('/')
                    break

            requests_cache[cache_key] = (alpha_dict[vep_ref], alpha_dict[vep_alt])

            return alpha_dict[vep_ref], alpha_dict[vep_alt]   

    # counter for wrong REF alleles
    wrong_refs = 0
    # counter for SNPs affecting the same position
    same_aa_pos_vars = 0
    # counter for samples with different alleles on each copy at the same position
    same_nuc_pos_vars = 0


    # mutate the sequences
    for seq_id, row in seq_table.iloc[1:, :].iterrows():
        
        # use v:id to retrieve the variants to add
        vars_to_add = variants.loc[[int(x) for x in row['v:idx'][2:].split('.')]]
        # use id in seq_id as index in the sequence data
        seq_idx = int(seq_id.split('_')[1])
        # find positions affected
        var_pos = set(vars_to_add['AA_POS'])

        for pos in var_pos:
            vars_per_pos = vars_to_add[vars_to_add["AA_POS"] == pos]

            if vars_per_pos.shape[0] == 1:
                # if only one variant per position
                ref, alt = vars_per_pos[["REF_idx", "ALT_idx"]].to_numpy()[0]
            else:
                # if more than one variant per position
                if len(vars_per_pos['ID'].str.split(':', expand=True)[1].unique()) == 1:
                    # if sample has variants in both copies at the same position
                    # pick the variant with lowest AC
                    same_nuc_pos_vars += 1
                    ref, alt = vars_per_pos.sort_values(by='AC').iloc[0][['REF_idx', 'ALT_idx']]
                else:
                    same_aa_pos_vars += 1
                    ref, alt = find_combined_effect(vars_per_pos)

            pos -= 1
            if not seq_data[seq_idx, pos, ref] == 1:
                wrong_refs += 1
            seq_data[seq_idx, pos, :] = 0
            seq_data[seq_idx, pos, alt] = 1

    print(f"Done. Found {wrong_refs} wrong REF allele(s) and {same_aa_pos_vars} SNP set(s) that affect the same protein position.")
    
    if same_nuc_pos_vars > 0:
        print(f"Cool! Found {same_nuc_pos_vars} sequence(s) with alleles on the same genomic position in both copies.")

    print("Saving data ... ", end="")

    # saving data
    patients_table.to_parquet(os.path.join(args.output_folder, 'patients_table.parquet'))
    seq_table.to_parquet(os.path.join(args.output_folder, 'seq_table.parquet'))
    variants.to_parquet(os.path.join(args.output_folder, 'variants.parquet'))
    with h5py.File(os.path.join(args.output_folder, 'seq_data.h5'), "w") as seq_data_file:
        seq_data_file.create_dataset("data", data=seq_data, compression = 'gzip')

    print("Done.")

    ###################
    # Plotting
    ###################

    print("Plotting ... ", end="")

    # plotting
    fig = plt.figure(figsize = (6*1.3, 2*1.3), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1, 2])

    # histogram: number of variants per sample
    axL = subfigs[0].subplots()
    r = subfigs[0].canvas.get_renderer()

    n_var_samples = len(nvars_per_sample)                   # total number of variant samples
    heights, xs = np.histogram(a = patients_table['n_vars'], 
                               bins = np.arange(0, patients_table['n_vars'].max() + 3,1))

    axL.bar(xs[:-1], heights, width = 1, color = 'tab:grey')
    axL.set_yscale('log')
    axL.set_ylabel("# samples")
    axL.set_xlabel("# variants in sample")
    axL._children[0]._facecolor = (0,0,0,1)
    txt = axL.text(axL.dataLim._points[1, 0],
                   axL.dataLim._points[1, 1],
                   f"{n_var_samples:,}", 
                   ha='right', va='top', 
                   fontsize = 11, 
                   weight = 'bold', 
                   color = 'tab:grey')

    txtbox = axL.transData.inverted().transform(txt.get_window_extent(renderer = r).get_points()).data

    axL.text(txtbox[1, 0],
             txtbox[0,1],
             "variant samples", 
             ha='right', va='top', 
             fontsize = 8, 
             color = 'tab:grey')

    # sequence polymorphism plot
    axR = subfigs[1].subplots(1, 2, 
                              gridspec_kw={"width_ratios":[3,1], 
                                                 "wspace":0.005})
    r = subfigs[1].canvas.get_renderer()

    nvars_per_pos = variants.groupby("AA_POS").aggregate({'AC': np.sum})
    seq_variation = np.zeros(len(refseq.seq))                    
    seq_variation[nvars_per_pos.index.to_numpy()-1] = nvars_per_pos['AC'].to_numpy()

    ax = axR[0]
    ax.plot(seq_variation, lw = 0.5)
    ax.set_xticks(np.linspace(0, len(refseq.seq), 3).astype(int))
    ax.set_yscale('log')
    ax.set_ylabel('# variants')
    ax.set_xlabel('res. position')

    txt = ax.text(ax.dataLim._points[0, 0],
                  ax.dataLim._points[1, 1],
                  f"{len(variants['AA_POS'].unique()):,}", 
                  ha='left', va='top', 
                  fontsize = 11, 
                  weight = 'bold', 
                  color = 'tab:blue')

    txtbox = ax.transData.inverted().transform(txt.get_window_extent(renderer = r).get_points()).data

    ax.text(txtbox[0, 0],
            txtbox[0,1],
            "variant positions", 
            ha='left', va='top', fontsize = 8, color = 'tab:blue')

    # histogram of variant counts per position 
    ax = axR[1]           
    hist = ax.hist(np.log10(nvars_per_pos['AC'].to_numpy()), 
                   bins = 20, 
                   orientation='horizontal', 
                   align = 'left', 
                   alpha = 0.6);

    # add bar for 0s
    hist_w = hist[2].patches[0]._height
    ax.barh(y =-hist_w, 
            width = (seq_variation == 0).sum(), 
            height = hist_w, 
            color = "black")
    # match the two plots' y scales
    axR[1].set_ylim(np.log10(axR[0].get_ylim()))

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('# positions')
    ax.set_xscale('log')

    fig.suptitle(args.gene_symbol, fontsize = 15)
    fig.savefig(os.path.join(args.plots_folder, args.gene_symbol + ".summary.png"))

    # step plot

    fig, ax = plt.subplots(figsize = (3.5,3.5), constrained_layout=True)

    unique_n_vars = sorted(seq_table.loc[seq_table['n_vars'] > 0, 'n_vars'].unique())

    counts_per_nvars = [seq_table[seq_table['n_vars']==n]['seq_count'].value_counts() for n in unique_n_vars]
    counts_per_nvars = [c.sort_index() for c in counts_per_nvars]

    for counts in counts_per_nvars:
        ax.step(counts.index, np.cumsum(counts.values[::-1])[::-1])

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid('on', alpha = 0.3)
    ax.set_xlabel('allele count (AC)')
    ax.set_ylabel(r'# sequences $\geq$ AC')
    ax.set_title(f"{args.gene_symbol} ({n_unique_var_seqs})")
    ax.legend(unique_n_vars, title = "# SNPs")

    fig.savefig(os.path.join(args.plots_folder, args.gene_symbol + ".step.png"))

    print("Done.")
