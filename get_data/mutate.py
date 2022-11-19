from Bio import SeqIO, Seq, Data
import os, sys, argparse, requests
import pandas as pd
import numpy as np

### helper variables and functions

aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
n_aa = len(aa_alphabet)
aa_to_index = {k:v for v,k in enumerate(aa_alphabet)}
index_to_aa = {v:k for v,k in enumerate(aa_alphabet)}

# deal with cases where two variants affect the same position
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

def find_combined_effect(vars_per_pos, gene_symbol):
    '''
    * Determine the combined effect of two SNPs affecting the same protein position
    * takes as input the result of variants.iloc[[variant_ids]]
    '''
    cache_key = tuple(vars_per_pos.index.to_list())
    if cache_key in requests_cache.keys():
        return requests_cache[cache_key]
    else:
        # use ID to get chrom, pos, ref
        df = vars_per_pos['var_id'].str.split(':', expand=True).sort_values(1)
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
            if ('missense_variant' in conseq['consequence_terms']) and (conseq['gene_symbol']==gene_symbol):
                vep_ref, vep_alt = conseq['amino_acids'].split('/')
                break

        requests_cache[cache_key] = (vep_ref, vep_alt)

        return vep_ref, vep_alt   
    
def mut(s,pos,alt):
    a = list(s)
    a[pos] = alt
    return ''.join(a)
    
def mutate(gene_symbol, refseq, seq_table, variants):
    
    print("Creating sequence data and fasta files ... ", end="")
    variants = variants.copy()
    variants['var_id'] = variants.index

    refseq = str(refseq.seq)
    seq_list = [refseq] * len(seq_table)
    
    # counter for wrong REF alleles
    wrong_refs = 0
    # counter for SNPs affecting the same position
    same_aa_pos_vars = 0
    # counter for samples with different alleles on each copy at the same position
    same_nuc_pos_vars = 0

    # mutate the sequences
    for seq_id, row in seq_table.iloc[1:, :].iterrows():
        # use vars_list to retrieve the variants to add
        vars_to_add = variants.loc[row['vars_list'].split('_')]
        # use id in seq_id as index in the sequence list
        seq_idx = int(seq_id.split('_')[1])
        # find positions affected
        var_positions = set(vars_to_add['AA_POS'])

        for pos in var_positions:
            vars_per_pos = vars_to_add[vars_to_add["AA_POS"] == pos]

            if vars_per_pos.shape[0] == 1:
                # if only one variant per position
                ref, alt = vars_per_pos[["AA_REF", "AA_ALT"]].to_numpy()[0]
            else:
                # if more than one variant per position
                if len(set(vars_per_pos['var_id'].str.split(':', expand=True)[1])) == 1:
                    # if sample has variants in both copies at the same position
                    # pick the variant with lowest AC
                    same_nuc_pos_vars += 1
                    ref, alt = vars_per_pos.sort_values(by='AC').iloc[0][['AA_REF', 'AA_ALT']]
                else:
                    same_aa_pos_vars += 1
                    ref, alt = find_combined_effect(vars_per_pos, gene_symbol)

            pos -= 1
            if not seq_list[seq_idx][pos] == ref:
                wrong_refs += 1
            seq_list[seq_idx] = mut(seq_list[seq_idx], pos, alt)

    print(f"Done. Found {wrong_refs} wrong REF allele(s) and {same_aa_pos_vars} SNP set(s) that affect the same protein position.")

    if same_nuc_pos_vars > 0:
        print(f"* Found {same_nuc_pos_vars} sequence(s) with alleles on the same genomic position in both copies.")
    
    # one-hot encode
    seq_matrices = [np.identity(n_aa)[[aa_to_index[a] for a in s], :] for s in seq_list]
    seq_matrices = np.concatenate([mat[np.newaxis,:,:] for mat in seq_matrices], axis=0)
    
    return seq_matrices, seq_list