import pandas as pd
import sys, requests
import numpy as np
import copy

# fasta sequence processing
from Bio import SeqIO, Data
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord

aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
n_aa = len(aa_alphabet)
aa_to_index = {k:v for v,k in enumerate(aa_alphabet)}
index_to_aa = {v:k for v,k in enumerate(aa_alphabet)}
codon_to_aa = Data.CodonTable.standard_dna_table.forward_table
def complement(s):
    return str(Seq(s).reverse_complement())
aa_to_codon = {1: {aa:[v for v,k in codon_to_aa.items() if k==aa] for aa in set(codon_to_aa.values())}, 
              -1: {aa:[complement(v) for v,k in codon_to_aa.items() if k==aa] for aa in set(codon_to_aa.values())}}
codon_to_aa = {1:codon_to_aa, -1:{complement(k):v for k, v in codon_to_aa.items()}}
vep_api_calls_cache = {}

def vep_api_call(chrom, start, end, strand, alt):
    server = "https://rest.ensembl.org"
    ext = f"/vep/human/region/{chrom}:{start}-{end}:{strand}/{alt}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json", 
                                          "refseq":'True'})
    return r.json()

def find_onp_effect(onp_df):
    """
    * Find the effect of an oligonucleotide polymorphism (multiple SNPs)
      affecting the same codon
    * Input: 
        - Pandas dataframe, with one row per variant and at least the columns:
            `var_id` as index
            `AA_STRAND`
            `AA_REF`
    """
    cache_key = '_'.join(onp_df.index.to_list())
    if cache_key in vep_api_calls_cache.keys():
        return vep_api_calls_cache[cache_key]
    onp_df = onp_df.reset_index()
    onp_df[["chr", "pos", "ref", "alt"]] = onp_df['var_id'].str.split(':', expand=True)
    onp_df = onp_df.astype({'pos':int}).sort_values("pos")
    chrom, strand, ref = onp_df['chr'][0], int(onp_df['AA_STRAND'][0]), onp_df['AA_REF'][0]
    start, end = int(onp_df['pos'].min()), int(onp_df['pos'].max())
    
    if len(onp_df)==3:
        # all 3 nucleotides in a codon are altered
        alt_nuc = ''.join(onp_df['alt'].to_list())
        alt = codon_to_aa[strand][alt_nuc]
    elif len(onp_df)==2:
        # 2 of 3 nucleotides in a codon are altered
        if end-start == 1:
            # the two nucleotides are consecutive
            alt_nuc = ''.join(onp_df['alt'].to_list())
        elif end-start == 2:
            # the two nucleotides are separated by a third
            # first need to find the middle nucleotide
            possible_codons = aa_to_codon[strand][ref]
            ref_nuc = [nuc for nuc in 'ACGT' if f"{nuc}".join(onp_df['ref'].to_list()) in possible_codons][0]
            alt_nuc = f'{ref_nuc}'.join(onp_df['alt'].to_list())
            
        vep_res = vep_api_call(chrom, start, end, 1, alt_nuc)
        
        for vep in vep_res[0]['transcript_consequences']:
            if ("missense_variant" in vep['consequence_terms']) and\
               (vep["amino_acids"][0] == ref):
                alt = vep['amino_acids'][-1]
                break
    vep_api_calls_cache[cache_key] = alt
    return alt

def mutate(refseq, hap_table, variants):
    refseq = MutableSeq(refseq.seq)
    hap_list = [SeqRecord(copy.deepcopy(refseq), 
                          id=row.name, 
                          description=row['hap_vars']) for _,row in hap_table.iterrows()]
    hap_indicator = np.zeros((len(hap_table), len(refseq))) 
    
    for idx, seq in enumerate(hap_list):
        if seq.description == '0':
            continue
            
        vars_list = seq.description.split('_')
        vars_df = variants.loc[vars_list]
        
        for pos,df in vars_df.groupby("AA_POS"):
            if seq.seq[pos-1] != df['AA_REF'][0]:
                print(f"\tAA_REF does not match reference sequence at AA_POS")
                print(f"\t\tID: {seq.id}, DESCRIPTION:{seq.description}")
            if len(df)==1:
                seq.seq[pos-1] = df['AA_ALT'].item()
            else:
                print(f"\tFound {len(df)} variants affecting the same codon at position {pos} ({seq.id})")
                seq.seq[pos-1] = find_onp_effect(df)
                print(f"\t\t{' + '.join(df.index.to_list())} = {df['AA_REF'][0]} -> {seq.seq[pos-1]}")
            assert hap_indicator[idx, pos-1] == 0
            hap_indicator[idx, pos-1] = 1

    hap_data = [np.identity(20)[np.newaxis, [aa_to_index[a] for a in hap.seq],:] for hap in hap_list]
    hap_data = np.concatenate(hap_data, axis = 0)
    
    return hap_data, hap_list, hap_indicator