### MIT 6.8610 Course Project
# Team: NLProteins
Ralph Estanboulieh, Yepeng Huang, Shuvom Sadhuka, Shashata Sawmya

## Description
Using Transformers, transfer learning, and biologically-informed attention to predict clinical phenotypes from genomic variation in the UK Biobank.

## Documentation
### Data organization
<img width="600" alt="image" src="https://user-images.githubusercontent.com/34459243/205469380-ad9e68ee-178b-4d7c-8328-dfa889b7c432.png">

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
  - the haplotypes one-hot encodings matrix: saved as `hap_data.npy`, and is a numpy matrix of dimensions `(N, L, 20)`, where `N` is the number of unique haplotypes seen in a gene, `L` is the length of the protein, and `20` is the number of common amino acids. 
  - the haplotypes variant position indicator matrix: saved as `hap_indicator.npy`, and is a numpy matric of dimensions `(N, L)`, where `N`is the number of unique haplotypes seen in a gene, and `L` is the length of the protein. All elements are `0`, except the positions at which there is a variant, which have value `1`. This is to make it easier to retrieve the positions (which are redundantly stored in the variants table), which can be helpful in computing more sophisticated embedding aggregation methods.
  - the sequence-variant matrix: stored as `seq_var_matrix`, and is a numpy matrix of dimensions `(S, V)`, where `S` is the number of unique sequences and `V` is the number of SNPs found in a gene. Each row is an indicator vector that is `1` when the corresponding sequence contains the variant corresponding to the column, and `0` otherwise. This matrix is useful for simple regressions and PRS-like analysis. 

This data organization scheme only stores each unique sequence (technically, haploptype) once and stores the maps from patient to sequence, from sequence to haplotype pair, and from haplotype to variants. A guiding principle in organizing the data here was to maximize pre-computation.

**Note:** the order of haplotypes across the first dimension of the haplotypes one-hot encodings, fasta file, and variant indicator matrix follows the order of the haplotypes in the haplotypes table. The same goes for the sequences and variants in the sequence-variant matrix: the rows correnspond to the ordered sequences in the sequences table, and the columns correspond to the ordered variants in the variants table. This makes it easy to retrieve data from the mapped IDs in the tables. The one-hot encoding of haplotype `hap_42` can simply be found at `hap_data[42]`!

**Example: getting the sequences across different genes in an individual** 

To get the sequence data across all genes for an individual, one simply finds the pair of haplotype IDs for each gene in said individual and retrieves (using the simple slicing routine mentioned in the note above) the one-hot encodings (or embeddings, or variant position indicators, or fasta sequence) of each of the two haplotypes. Then, the user has complete control on how they want to combine those haplotypes and return them. One option would be to add them up (in the case of one-hot encodings) and either return them as a tuple of `(L_p, 20)` matrices, where `L_p` is the length of protein `p`, these matrices can also be concatenated along the first dimension. (Returning a tupe might be better in cases wherein each gene needs to be processed independently first). Another (more biologically plausible) way to deal with haplotypes would be to consider them as separate entities. 

