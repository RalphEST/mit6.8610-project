import pandas as pd
import numpy as np
import h5py

class VariationDataset:
    """
    Class to hold and process sequence variation and phenotypic data. For each gene, the
    following data should be provided (via input `seq_data_paths` in at instantiation):
        * A Pandas DataFrame of variants with columns
            - `ID` (as the index): generally of the form chr:pos:ref:alt, but can be 
              something else
            - `AA_POS`, `AA_REF`, `AA_ALT`, `AA_STRAND`: the position and strand of the
              variant, and the reference and alternate alleles
            - Other optional information, such as allele counts `AC` and frequencies `AF` 
              precalculated from VCF file, which are mandatory for variant filtering
        * A Pandas DataFrame of patients and their sequences with columns:
            - `patient_id` (as the index): the ID of each patient
            - `seq_id`: the ID of a patient's sequence
            - Other optional information (e.g. the number of variants per sequence, the 
              IDs of each variant per sequence, sequence counts)
        * A Pandas DataFrame of unique sequences with columns:
            - `seq_id` (as the index): the ID of a unique sequence
            - `v_ids`: the variant IDs in the sequence, joined with '_'
            - `n_vars`: the number of variants in the sequence
            - `seq_count`: the count of  a unique sequence in the data 
        * The sequence data (either one-hot encoding or some embedding), saved as an
          binary `.npy` array
    
    NOTE: the Pandas DataFrames are expected to be stored as `.parquet` files.
          
    The phenotypic data should be a Pandas DataFrame stored as a `.parquet` file, and it
    must contain a `patient_id` column that matches the identically-named column in each
    sequence table
    
    NOTE: The class can be given the variation data for one or more genes. Gene-specific  
    suffixes will be added to the appropriate column names to reduce confusion.
    
    """
    
    def __init__(self, seq_data_paths, phen_data_path):
        """        
        -----------------------------------------------------------------------------------
        Inputs          | Description
        ----------------+------------------------------------------------------------------
        seq_data_paths  | Two-level dictionary. First level of keys should be the sequence 
                        | (gene) names, and the second level dictionaries should be like:
                        | {'variants': VARIANTS_TABLE_PATH, 
                        |  'sequences': SEQUENCE_TABLE_PATH, 
                        |  'patients': PATIENT_DATA_PATH, 
                        |  'seqdata': SEQUENCE_DATA_PATH}
        ----------------+------------------------------------------------------------------
        phen_data_path  | The path to the `.parquet` dataframe of patient phenotypes
        -----------------------------------------------------------------------------------
        
        """
        self.seq_data_paths = seq_data_paths
        self.phen_data_path = phen_data_path
        self.gene_names = list(seq_data_paths.keys())
        self.n_genes = len(self.gene_names)
        self.many_genes = (self.n_genes > 1)
        
        self.variants = {s:None for s in self.gene_names}
        self.sequences = {s:None for s in self.gene_names}
        self.patients = {s:None for s in self.gene_names}
        self.seqdata = {s:None for s in self.gene_names}
        
        # read tables and data
        self.summary = {}
        total_seq_length, total_n_vars = 0, 0
        
        for gene in self.gene_names:
            self.variants[gene] = pd.read_parquet(seq_data_paths[gene]['variants'])
            self.sequences[gene] = pd.read_parquet(seq_data_paths[gene]['sequences'])
            self.patients[gene] = pd.read_parquet(seq_data_paths[gene]['patients'], 
                                                 columns = ['n_vars', 'seq_id'])\
                                    .add_prefix(f"{gene}:" if self.many_genes else "")
            self.seqdata[gene] = np.load(seq_data_paths[gene]['data'])
            
            total_length += self.seqdata[gene].shape[1]
            total_n_vars += len(self.variants[gene])
            self.summary[gene] = {"length": self.data[gene].shape[1], 
                                  "n_vars": len(self.variants[gene]), 
                                  "n_seqs": len(self.sequences[gene])}
        
        if not self.many_genes:
            self.variants = self.variants[gene]
            self.sequences = self.sequences[gene]
            self.patients = self.patients[gene]
            self.data = self.data[gene]
        else:
            # combine patient sequences
            self.patients = pd.concat(self.patients.values(), axis = 1)
            # sum the number of variants per patient
            self.patients['n_vars'] = self.patients.loc[:,self.patients.columns.str.contains('n_vars')].sum(1)
            # combine the sequence IDs across all genes to generate new network-level sequence IDs
            self.patients['jointed_seq_ids'] = self.patients\
                                                .loc[:, self.patients.columns.str.contains('seq_id')]\
                                                .agg('_'.join, axis = 1)
            # create new sequences table
            sequences = self.patients['jointed_seq_ids'].copy().drop_duplicates()
            sequences.index.name = 'new_seq_id'
            
            # assign combined sequence IDs to patients
            self.patients = pd.merge(left = self.patients.reset_index(), 
                                     right = sequences.reset_index(), 
                                     left_on = 'jointed_seq_ids', 
                                     right_on = 'new_seq_id', 
                                     how = 'left')
            self.patients = self.patients[["patient_id", "n_vars", "new_seq_id"]]\
                                                    .astype({'new_seq_id': 'category'})\
                                                    .set_index("patient_id")
            # count combined sequences
            seq_counts = self.patients['new_seq_id'].value_counts().rename('seq_count')
            seq_counts = seq_counts.astype('int32')
            sequences = pd.merge(left = sequences, 
                                 right = seq_counts,
                                 left_on = "new_seq_id", 
                                 right_index = True,
                                 how = "left")
            
            # resplit the combined sequence IDs in the sequences table
            sequences[self.gene_names] = sequences['jointed_seq_ids'].str.split('.', expand = True)
            sequences = sequences.drop(columns = ['jointed_seq_ids'])
            
            self.sequences['all'] = sequences
            
            # add to summary
            self.summary['all_genes'] = {'length' : total_length, 
                                         'n_vars': total_n_vars, 
                                         'n_seqs': len(sequences)}
            
        self.summary = pd.DataFrame(self.summary).T
            
        self.seq_lengths = self.summary['length'].tolist() 
        self.n_sequences = self.summary['n_seqs'].tolist() 
        if not self.many_genes: 
            self.seq_lengths, self.n_sequences = self.seq_lengths[0], self.n_sequences[-1]
        self.seq_idxs = np.arange(0, self.n_sequences)
                
    def weights(self, t):
        """
        Calculate the training weights of the sequences using their counts in the population. 
        Parameter t <= 1 is to flatten the distribution (smaller t leads to flatter distributions):
        
        w'_i = c_i / (sum_j c_j)
        w_i = w'_i^t / (sum_j w'_j^t)
        
        """
        counts = self.sequences['seq_count'].to_numpy()
        weights = counts/counts.sum()
        weights = weights ** t
        return weights/weights.sum()