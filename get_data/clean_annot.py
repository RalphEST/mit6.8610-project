import pandas as pd
import sys

if __name__=="__main__":
    
    '''
    * clean up VEP annotations file
    * required inputs:
        1 - protein RefSeq id
        2 - input TSV file path
        3 - output TSV file path
    '''

    refseq_id, tsv_in, tsv_out, ids_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    df = pd.read_table(tsv_in, sep = '\t')
    print(f"#### LENGTH OF TABLE: {len(df)} ####")
    res_df = df[['chrom', 'position', 'variant_id', 'GIVEN_REF', 'Allele']].copy()

    # find our protein of interest's position in the list of proteins affected by each variant
    df = df.astype({'EVEmap_REFSEQ_PROTEIN':str, 
                    'EVEmap_AMINO_ACID_POS':str, 
                    'EVEmap_WT_AA':str, 
                    'EVEmap_SUBS_AA':str})
    df['protein_idx'] = df['EVEmap_REFSEQ_PROTEIN'].str.split(',').apply(lambda x: x.index(refseq_id))

    # add protein information
    res_df["AA_POS"] = df.apply(lambda r: r['EVEmap_AMINO_ACID_POS'].split(',')[r['protein_idx']], axis = 1)
    res_df['AA_STRAND'] = df['STRAND']
    res_df["AA_REF"] = df.apply(lambda r: r['EVEmap_WT_AA'].split(',')[r['protein_idx']], axis = 1)
    res_df["AA_ALT"] = df.apply(lambda r: r['EVEmap_SUBS_AA'].split(',')[r['protein_idx']], axis = 1)
    
    # rename for annotation later
    res_df = res_df.rename(columns = {'chrom': '#CHROM', 
                                      'position':'POS', 
                                      'variant_id':'ID', 
                                      'GIVEN_REF':'REF', 
                                      'Allele':'ALT'})
    
    # only include non-synonymous variants
    res_df = res_df[res_df["AA_REF"] != res_df['AA_ALT']]
    
    # write out annotation table
    res_df.to_csv(tsv_out, sep="\t", index=False)
    
    # write out variant IDs table
    res_df["ID"].to_csv(ids_out, index = False, header = False)
    
    # return the genomic range for VCF filtration
    print(f"{res_df['POS'].min()}-{res_df['POS'].max()}")