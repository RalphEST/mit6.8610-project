### MIT 6.8610 Course Project
# Team: NLProteins
Ralph Estanboulieh, Yepeng Huang, Shuvom Sadhuka, Shashata Sawmya

## Description
Using Transformers, transfer learning, and biologically-informed attention to predict clinical phenotypes from genomic variation in the UK Biobank.

## Documentation
### Data organization

Exome sequencing data is first filtered, annotated, and processed to generate the following structure. Each gene has its own directory in which are stored the following data:
  - The variants table: a Pandas `DataFrame` stored as `variants_table.parquet`. Each row is a variant, and the columns are:
    - `var_id` (index): the variant ID in the `chr:pos:ref:alt` format 
    - `AA_POS`: the position of the variant in the protein sequence coordinate
    - `AA_STRAND`: the genomic strand of the variant
    - `AA_REF` and `AA_POS`: the reference and alternate alleles, respectively
    - `AC` and `AF`: allele count and frequencies, respectively
  - The sequences table: a Pandas `DataFrame` stored as `seq_table.parquet`. Each row is a *unique* sequence, and the columns are:
    - `seq_id` (index): the ID of the sequence in format `seq_[int]`
    - `n_vars`: the number of variants in that sequence
    - `seq_count`: the number of times a sequence is seen in the population
    - `vars_list`: the underscore-delimited string of variant IDs in the sequence. Each ID is preceded by `n#`, where `n` is the zygosity of the variant (either 1 or 2)
    - `hap0` and `hap1`: the haplotype IDs post-phasing, each of the format `hap_[int]`
    - `phased_vars_list`: two underscore-delimited string of variant IDs after phasing, separated by a `|` (i.e. `1:115768346:T:C_1:115701283:A:C|1:115768346:T:C	`)
  - the haplotypes table: a Pandas `DataFrame` stored as `hap_table.parquet`. Each row is a *unique* haplotype, and the columns are:
    - `hap_id` (index): the haplotype ID, of the format `hap_[int]`
    - `hap_vars`: an underscore-delimited string of variant IDs in the sequence, this time without the zygosity (i.e. `1:115768529:G:C_1:115768346:T:C`)
  - the patients table: a Pandas `DataFrame` stored as `patients_table.parquet`. Each row is an individual participant, and the columns are:
    - `patient_id`: the ID of the participant
    - `n_vars`: the number of variants seen in the gene of interest in that patient
    - `phased_vars_list` and `vars_list`: same as `phased_vars_list` and `vars_list` in sequences table (redundancy for easier downstream processing)
    - `seq_id`: the unique sequence ID of that participant (which can be retrieved in the sequences table)
    - `seq_count`: same as `seq_count` in the sequences table
  - the haplotypes FASTA file: saved as `hap_fasta.fasta`, contains all the unique haplotype sequences saved as a fasta file. The ID of each sequence is the `hap_id` and the description is `hap_vars`, both as seen in the haplotypes table.
  - the haplotypes one-hot encodings file: saved as `hap_data.npy`, and is a numpy matrix of dimensions `(N, L, 20)`, where `N` is the number of unique haplotypes seen in a gene, `L` is the length of the protein, and `20` is the number of common amino acids. 
  - the haplotypes variant position indicator file: saved as `hap_indicator.npy`, and is a numpy matric of dimensions `(N, L)`, where `N`is the number of unique haplotypes seen in a gene, and `L` is the length of the protein. All elements are `0`, except the positions at which there is a variant, which have value `1`. This is to make it easier to retrieve the positions (which are redundantly stored in the variants table), which can be helpful in computing more sophisticated embedding aggregation methods.
  - the sequence-variant matrix: stored as `seq_var_matrix`, and is a numpy matrix of dimensions `(S, V)`, where `S` is the number of unique sequences and `V` is the number of SNPs found in a gene. Each row is an indicator vector that is `1` when the corresponding sequence contains the variant corresponding to the column, and `0` otherwise. This matrix is useful for simple regressions and PRS-like analysis. 

