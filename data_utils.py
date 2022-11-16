import pandas as pd
import numpy as np
import h5py

class VariationDataset:
    """
    Class to hold and process sequence variation and phenotypic data. For each gene, the
    following data should be provided (via input `seq_data_paths` in at instantiation):
        * A Pandas DataFrame of variants with columns
            - `ID`: generally of the form chr:pos:ref:alt, but can be something else
            - `AA_POS`, `AA_REF`, `AA_ALT`, `AA_STRAND`: the position and strand of the
              variant, and the reference and alternate alleles
            - Other optional information (e.g. allele counts/frequencies from VCF file)
        * A Pandas DataFrame of patients and their sequences with columns:
            - `patient_id`: the ID of each patient
            - `seq_id`: the ID of a patient's sequence
            - Other optional information (e.g. the number of variants per sequence, the 
              IDs of each variant per sequence, sequence counts)
        * A Pandas DataFrame of unique sequences with columns:
            - `seq_id`: the ID of a unique sequence
            - `v_ids`: the variant IDs in the sequence, joined with '_'
            - `n_vars`: the number of variants in the sequence
            - `seq_count`: the count of a unique sequence in the data 
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
        Inputs          | Description
        ---------------------------------------------------------------------------------
        seq_data_paths  | Two-level dictionary. First level of keys should be the sequence 
                        | (gene) names, and the second level dictionaries should be like:
                        | {'variants': VARIANTS_TABLE_PATH, 
                        |  'sequences': SEQUENCE_TABLE_PATH, 
                        |  'patients': PATIENT_DATA_PATH, 
                        |  'data': SEQUENCE_DATA_PATH}
        phen_data_path  | The path to the `.parquet` dataframe of patient phenotypes
        
        """
        self.seq_data_paths = seq_data_paths
        self.phen_data_path = phen_data_path
        self.gene_names = list(seq_data_paths.keys())
        self.n_genes = len(self.gene_names)
        self.many_genes = (self.n_genes > 1)
        
        self.variants = {s:None for s in self.gene_names}
        self.sequences = {s:None for s in self.gene_names}
        self.patients = {s:None for s in self.gene_names}
        self.data = {s:None for s in self.gene_names}
        
        # read tables and data
        for seq in self.gene_names:
            self.variants[seq] = pd.read_parquet(path_dict[seq]['variants'])
            self.sequences[seq] = pd.read_parquet(path_dict[seq]['sequences'])
            self.patients[seq] = pd.read_parquet(path_dict[seq]['patients'], 
                                                 columns = ['n_vars', 'seq_id'])
            
            with h5py.File(path_dict[seq]['data'], 'r') as file:
                self.data[seq] = file['data'][()]
                self.data[seq] = np.expand_dims(self.data[seq], axis = 1)
        
        if not self.many_genes:
            self.variants = self.variants[seq]
            self.sequences = self.sequences[seq]
            self.patients = self.patients[seq]
            self.data = self.data[seq]
            
        self.seq_length = self.data.shape[2]
        self.n_sequences = len(self.sequences)
        self.n_variants = len(self.variants)
            
        self.seq_idxs = np.arange(0, self.n_sequences)
                
    def weights(self, t):
        counts = self.sequences['seq_count'].to_numpy()
        weights = counts/counts.sum()
        weights = weights ** t
        return weights/weights.sum()
    
    def seqvar_matrix(self):
        vidxs_list = self.sequences['v:idx'].to_list()
        sv_matrix = np.zeros((self.n_sequences, self.n_variants))
        for vidxs, sv_row in zip(vidxs_list, sv_matrix):
            if vidxs == '0':
                continue
            else:
                vidxs = [int(x)-1 for x in vidxs[2:].split('.')]
                sv_row[vidxs] = 1
        return sv_matrix