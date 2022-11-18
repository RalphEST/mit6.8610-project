import pandas as pd
import subprocess
import sys
import re
import argparse

# getting arguments and inputs
parser = argparse.ArgumentParser(description='Get annotated UKBB variation data for a list of genes')
parser.add_argument('--gene_list', type=str, help='Path to the list of genes of interest')
parser.add_argument('--output_file', type=str, help='Path to the output data folder')
args = parser.parse_args()

# generate biomart table to get gene coordinates and protein isoforms
gene_list = []
with open(args.gene_list, 'r') as file:
    for line in file.readlines():
        gene_list.append(line.strip())
gene_list = ','.join(gene_list)
biomart_url = ('http://www.ensembl.org/biomart/martservice?query='
               '<?xml version="1.0" encoding="UTF-8"?>'
               '<!DOCTYPE Query>'
               '<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >'
               '<Dataset name = "hsapiens_gene_ensembl" interface = "default" >'
               '<Filter name = "transcript_biotype" value = "protein_coding"/>'
               '<Filter name = "biotype" value = "protein_coding"/>'
               f'<Filter name = "hgnc_symbol" value = "{gene_list}"/>'
               '<Attribute name = "ensembl_gene_id" />'
               '<Attribute name = "ensembl_transcript_id" />'
               '<Attribute name = "chromosome_name" />'
               '<Attribute name = "refseq_peptide" />'
               '<Attribute name = "ensembl_peptide_id" />'
               '<Attribute name = "description" />'
               '<Attribute name = "strand" />'
               '<Attribute name = "start_position" />'
               '<Attribute name = "end_position" />'
               '<Attribute name = "external_gene_name" />'
               '<Attribute name = "gene_biotype" />'
               '<Attribute name = "transcript_biotype" />'
               '<Attribute name = "uniprot_isoform" />'
               '<Attribute name = "transcript_length" />'
               '</Dataset></Query>')
batch_command = ["wget", "-O", "biomart_table.tsv", f"{biomart_url}"]
# subprocess.run(batch_command)

# read biomart table
biomart_df = pd.read_csv('biomart_table.tsv', sep='\t')
print(f"BioMart output contains {len(biomart_df)} rows over {len(set(biomart_df['Gene name']))} unique genes.")
biomart_df = biomart_df[~biomart_df["RefSeq peptide ID"].isna()]
print(f"{len(set(biomart_df['Gene name']))} unique genes left after removing rows without RefSeq peptide ID.")

# fix chromosome/contig annotation issues
def get_chrom(s):
    if s.isdigit(): return int(s)
    elif s in 'XY': return s
    else: return re.search("(?:CHR)(\d+)", s).group(1)
biomart_df['Chromosome/scaffold name'] = biomart_df['Chromosome/scaffold name'].apply(get_chrom)

# select isoform refseq IDs
# This is done by picking the isoform smallest ID number, as it is in most cases
# considered to be the canonical isoform
refseq_ids = []
for name, b in biomart_df.groupby('Gene name'):
    b['ID->int'] = b['RefSeq peptide ID'].apply(lambda s: int(re.search("(?<=NP_)(?:\d+)", s).group(0)))
    refseq_ids.append(b.sort_values('ID->int')['RefSeq peptide ID'].iloc[0])


# drop gene duplicates and sort alphabetically
biomart_df = biomart_df[biomart_df['RefSeq peptide ID'].isin(refseq_ids)]\
                                                        .drop_duplicates('Gene name')\
                                                        .sort_values('Gene name')

print(f"{len(biomart_df)} RefSeq peptide IDs found!")

biomart_df[['Gene name', 'RefSeq peptide ID', 'Chromosome/scaffold name']]\
                            .to_csv(args.output_file, header=False, index=False)

