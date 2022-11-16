import pandas as pd
import numpy as np
import h5py
import pyarrow.parquet as pq

import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class SeqVarData:
    """
    Class to hold and process sequence variation data. Each data folder should contain:
        * a table of variants
        * a table of patients and their sequences (and n_var, seq_count)
        * a table of unique sequences (and n_var, seq_count)
        * the sequence sequence data
    The tables must be stored as parquet files. The sequence data must be a NumPy array
    compressed and stored as an HDF5 file.
    
    The class can be passed the variant data for many genes. In this case, the patient
    IDs must match across patients tables. Gene-specific suffixes will be added to the
    appropriate column names to reduce confusion.
    
    """
    
    def __init__(self, path_dict):
        """
        Note: in this implementation, weights() only  works with one protein
        ToDo: implement routines for sequence network
        
        Inputs      | Description
        ---------------------------------------------------------------------------------
        path_dict   | Two-level dictionary. First level of keys should be the sequence 
                    | names, and the second level dictionaries should be of the form:
                    | {'variants': PATH, 
                    |  'sequences': PATH, 
                    |  'patients': PATH, 
                    |  'data': PATH}
        
        """
        self.path_dict = path_dict
        self.gene_names = list(path_dict.keys())
        self.n_gene_names = len(self.gene_names)
        self.many_genes = (self.n_gene_names > 1)
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