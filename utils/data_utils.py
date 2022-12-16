# basics
import pandas as pd
import numpy as np
import os, sys

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data

# sklearn imports
from sklearn.model_selection import train_test_split

default_hap_collapse_funcs = {'one-hot-encodings': lambda x: np.sum(x, axis=0), 
                              'embeddings': lambda x: x[0], 
                              'indicators': lambda x: np.sum(x, axis=0),
                              'esm-tokens': lambda x: x[0]}

class VariationDataset(Dataset):
    """
    Class to hold and process sequence variation and phenotypic data. 
    """
    
    def __init__(self, 
                 table_paths,
                 data_paths,
                 phenotypes_path,
                 predict,
                 ppi_graph_path=None,
                 hap_collapse_funcs=None,
                 keep_genes_separate=True,
                 low_memory=True):
        """    
        Inputs:
        * `table_paths`: 
            Two-level dictionary. First level of keys should be the sequence (gene) names,
            and the second level dictionaries should be of the form:
            {'variants': path/to/variants/table.parquet,
             'haplotypes': path/to/haplotypes/table.parquet,
             'sequences': path/to/sequences/table.parquet, 
             'patients': path/to/patients/table.parquet}
        * `data_paths`: 
            Two-level dictionary, with first level keys being the sequence (gene) names and
            the second level dictionaries of the form:
            {'one-hot-encodings': path/to/one/hot/encodings.npy
             'embeddings': path/to/sequence/embeddings.npy,
             'indicators': path/to/variant/position/indicators.npy,
             'seq-var-matrix': path/to/sequence/variant/matrix.npy,
             'esm-tokens': path/to/unique/sequence/esm/tokens.npy}
             Only the data needed for the analysis should be added to the `data_paths` 
             dictionaries.
        * `phenotypes_path`:
            Path to Pandas DataFrame with index `patient_id` and at least one column of 
            phenotypes of interest. The patient IDs should be of dtype `str` and should 
            match the IDs in the patients table.
        * `ppi_graph`:
            Path to protein-protein interactions table.
        * `hap_collapse_funcs`:
            Dictionary of functions to deal with haplotypes. Keys can be one or more of the
            following: `one-hot-encodings`, `embeddings`, and `indicators`. (The sequence-
            variant matrix, because it is genotype-level, does not require haplotype merging.)
            The default functions sum up the one-hot-encodings and indicators and pick the
            first embedding and tokens list of the two haplotypes.
        * `keep_genes_separate`:
            If `True`, `__getitem__` returns, per each data type selected, a dictionary of
            genes and their respective data. If `False`, data is concatenated across all genes
            along the first dimension.
        
        """
        self.table_paths = table_paths
        self.data_paths = data_paths
        self.predict = predict
        self.low_memory = low_memory
        self.phenotypes_path = phenotypes_path
        self.ppi_graph_path = ppi_graph_path
        self.hap_collapse_funcs = hap_collapse_funcs if hap_collapse_funcs else default_hap_collapse_funcs
        self.keep_genes_separate = keep_genes_separate
        
        self.gene_names = list(table_paths.keys())
        self.seq_id_columns = [g+":seq_id" for g in self.gene_names]
        self.n_genes = len(self.gene_names)
        self.gene_to_index = {g:i for i,g in enumerate(self.gene_names)}
        self.many_genes = (self.n_genes > 1)
        self.data_types = list(data_paths.values())[0]
        
        self.variants = {s:None for s in self.gene_names}
        self.sequences = {s:None for s in self.gene_names}
        self.patients = {s:None for s in self.gene_names}
        self.haplotypes = {s:None for s in self.gene_names}
        
        self.data_types = list(list(data_paths.values())[0].keys())
        self.data = {t:{s:None for s in self.gene_names} for t in self.data_types}
        
        # read tables and data
        self.summary = {}
        total_seq_length, total_n_vars = 0, 0
        
        for gene in self.gene_names:
            print(f'Fetching {gene} data ...', end=" ")
            # reading .parquet tables
            self.variants[gene] = pd.read_parquet(table_paths[gene]['variants'])
            self.sequences[gene] = pd.read_parquet(table_paths[gene]['sequences'])
            self.patients[gene] = pd.read_parquet(table_paths[gene]['patients'], 
                                                  columns = ['n_vars', 'seq_id'])\
                                    .add_prefix(f"{gene}:" if self.many_genes else "")
            self.haplotypes[gene] = pd.read_parquet(table_paths[gene]['haplotypes'])
            
            # reading data
            seq_length = None
            
            for data_t in self.data_types:
                self.data[data_t][gene] = np.load(data_paths[gene][data_t], 
                                                  mmap_mode='r+' if self.low_memory else None)
                
                seq_length_dim = -1 if data_t in ['indicators', 'seq-var-matrix'] else 1
                seq_length = self.data[data_t][gene].shape[seq_length_dim]
                
            # summarizing
            self.summary[gene] = {"length": seq_length, 
                                  "n_vars": len(self.variants[gene]), 
                                  "n_seqs": len(self.sequences[gene])}
            print("Done")
        
        print("Combining tables ...", end=" ")
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
            self.patients['concat_seq_ids'] = self.patients\
                                                .loc[:, self.seq_id_columns]\
                                                .agg(':'.join, axis = 1)
            # create new sequences table
            sequences = self.patients.copy(deep=True).drop_duplicates(subset = ['concat_seq_ids'])
            sequences.index = [f"xseq_{i}" for i in range(len(sequences))]
            sequences.index.name = 'xseq_id'
            # assign combined sequence IDs to patients
            self.patients = pd.merge(left = self.patients.reset_index(), 
                                     right = sequences.reset_index()[['concat_seq_ids', 'xseq_id']], 
                                     left_on = 'concat_seq_ids', 
                                     right_on = 'concat_seq_ids', 
                                     how = 'left')
            
            self.patients = self.patients[["patient_id", "n_vars", "xseq_id"]]\
                                .astype({'xseq_id': 'category'})\
                                .set_index("patient_id")
            
            # count combined sequences
            seq_counts = self.patients['xseq_id'].value_counts().rename('seq_count')
            seq_counts = seq_counts.astype('int32')
            sequences = pd.merge(left = sequences, 
                                 right = seq_counts,
                                 left_on = "xseq_id", 
                                 right_index = True,
                                 how = "left")
            
            # clean up
            sequences = sequences.drop(columns = ['concat_seq_ids'])
            sequences = sequences.loc[:, ~sequences.columns.str.contains(':n_vars')]            
            self.sequences['all'] = sequences
            
        self.summary = pd.DataFrame(self.summary).T
        self.summary.loc['all'] = self.summary.sum(0)
        self.summary.loc['all', 'n_seqs'] = len(self.sequences['all'])
        
        # optimizing
        # this step speeds up datum retrieval at the expense of space
        self.xseq_id_to_seq_idxs = self.sequences['all'][self.seq_id_columns].apply(lambda r: [int(s[4:]) for s in r])
        
        for g in self.gene_names:
            self.sequences[g]['hap_list'] = self.sequences[g][['hap0', 'hap1']]\
                                                .apply(lambda r: [int(r[0][4:]), int(r[1][4:])], 
                                                       axis=1)
            self.sequences['all'] = pd.merge(left=self.sequences['all'],
                                             right=self.sequences[g]['hap_list'].to_frame(g+':hap_list'),
                                             left_on=g+":seq_id",
                                             right_index=True)
        self.hap_list_columns = [g+":hap_list" for g in self.gene_names]
        self.xseq_id_to_hap_idxs = self.sequences['all'][self.hap_list_columns]
        self.sequences['all'] = self.sequences['all'].drop(columns=self.hap_list_columns)
        
        self.ac_filter = self.filter_variants_by_ac()
        print("Done")
        
        print("Integrating with phenotypes data ...", end="")
        self.phenotypes = pd.read_parquet(phenotypes_path)
        self.phenotype_cols = self.phenotypes.columns.to_list()
        self.patients = pd.merge(left=self.patients,
                                 right=self.phenotypes,
                                 left_index=True,
                                 right_index=True,
                                 how='inner')
        self.patients = self.patients.dropna(axis='index', 
                                             how='any', 
                                             subset=self.phenotype_cols)
        self.sample_names = self.patients.index.to_list()
        print("Done")
        
        if self.ppi_graph_path:
            print("Processing PPI graph ...", end="")
            self.ppi_table = pd.read_table(self.ppi_graph_path)
            self_loops = torch.tensor([[i,i] for i in range(self.n_genes)], dtype=torch.long)
            interactions = torch.tensor(self.ppi_table[['#node1', 'node2']].replace(self.gene_to_index).to_numpy(), 
                                        dtype=torch.long)
            self.edge_index = torch.cat([self_loops, 
                                         interactions, 
                                         interactions.flip(1)], 
                                        axis=0).t().contiguous()
            print("Done")
            
    def get_feature_dimensions(self, features):
        return {f:list(self.data[f].values())[0].shape for f in features}
    
    def filter_variants_by_ac(self, ac_thresh_lo=1, ac_thresh_hi=np.inf):
        self.ac_filter = [self.variants[g]['AC'].between(ac_thresh_lo, ac_thresh_hi, inclusive='both').values 
                          for g in self.gene_names]
        
    def train_test_split(self, balance_on=None, train_fraction=0.8):
        if isinstance(balance_on, str):
            self.samples_train, self.samples_test = train_test_split(self.sample_names,
                                                                     train_size=train_fraction,
                                                                     stratify=self.patients[balance_on],
                                                                     random_state=42)
        elif isinstance(balance_on, list):
            self.patients['balance_on'] = self.patients[balance_on].astype(str)\
                                                .apply(lambda r: ''.join(r), axis = 1)

            # isolate samples in categories that only occur once (sklearn's train_test_split cannot deal with these)
            isolated_samples = self.patients[~self.patients['balance_on'].duplicated(keep=False)].index.to_list()
            
            self.samples_train, self.samples_test = train_test_split(self.patients.drop(isolated_samples).index.to_list(),
                                                                     train_size=train_fraction,
                                                                     stratify=self.patients.drop(isolated_samples)[balance_on],
                                                                     random_state=42)
            self.samples_train += isolated_samples
        else:
            self.samples_train, self.samples_test = train_test_split(self.sample_names,
                                                                     train_size=train_fraction,
                                                                     random_state=42)
            
        return (VariationDatasetSplit(self, train=True), 
                VariationDatasetSplit(self, train=False))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, pid):
        # the label
        label = self.patients.loc[pid, self.predict].values.astype(float)
        item_dict = {"labels": label}
        
        # the sequences and haplotypes
        xseq_id = self.patients.loc[pid, 'xseq_id']
        seq_idxs = self.xseq_id_to_seq_idxs.loc[xseq_id]
        hap_idxs = self.xseq_id_to_hap_idxs.loc[xseq_id]

        for dt in self.data_types:
            if dt == "seq-var-matrix":
                dt_data = [self.data[dt][g][i][f] for g,i,f in zip(self.gene_names, 
                                                                   seq_idxs, 
                                                                   self.ac_filter)]
            else:
                dt_data = [self.hap_collapse_funcs[dt]([self.data[dt][g][i[0]], 
                                                        self.data[dt][g][i[1]]]) for g,i in zip(self.gene_names, 
                                                                                                hap_idxs)]
            item_dict[dt] = np.concatenate(dt_data, axis=0) if not self.keep_genes_separate\
                            else {g:d for g,d in zip(self.gene_names, dt_data)}
        return item_dict
        
class VariationDatasetSplit(Dataset):
    """
    Class to hold the train-test splits without having to deep-copy VariationDataset instances.
    """
    def __init__(self, dataset, train):
        """
        Inputs:
        * `dataset`: 
            A VariationDataset instance after the method `train_test_split` has been run at least
            once.
        * `train`:
            `True` or `False` to select the training and testing splits, respectively.
            
        """
        self.dataset = dataset
        self.train = train
        self.samples = self.dataset.samples_train if train else self.dataset.samples_test
        self.samples_df = self.dataset.patients.loc[self.samples]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.dataset[self.samples[idx]]
    
    def weights(self, on, method='ins', flatten_factor=1):
        if method=='ins':
            class_weights = self.samples_df[on].value_counts()
            weights = np.array([1e3/class_weights[s] for s in self.samples_df[on].to_list()])
        
        return weights**flatten_factor
    
def load_variation_dataset(data_dir, 
                           gene_list, 
                           data_types, 
                           phenotypes_path,
                           predict,
                           ppi_graph_path=None,
                           embeddings_file=None,
                           hap_collapse_funcs=None,
                           keep_genes_separate=True,
                           low_memory=True):
    
    if isinstance(gene_list, str):
        gene_list = read_gene_list(gene_list)
        
    table_paths = {g:{'variants': os.path.join(data_dir, g, 'variants_table.parquet'), 
                  'sequences': os.path.join(data_dir, g, 'seq_table.parquet'), 
                  'patients': os.path.join(data_dir, g, 'patients_table.parquet'),
                  'haplotypes': os.path.join(data_dir, g, 'hap_table.parquet')} for g in gene_list}
    data_type_2_file_name = {
        "one-hot-encodings": 'hap_data.npy',
        "indicators": 'hap_indicator.npy',
        "seq-var-matrix": 'seq_var_matrix.npy',
        "embeddings": embeddings_file,
        "esm-tokens": "hap_esm_tokens.npy"
    }
    data_paths = {g:{dt:os.path.join(data_dir, g, data_type_2_file_name[dt]) 
                     for dt in data_types} 
                  for g in gene_list}
        
    data = VariationDataset(table_paths=table_paths, 
                            data_paths=data_paths, 
                            phenotypes_path=phenotypes_path,
                            predict=predict,
                            ppi_graph_path=ppi_graph_path,
                            hap_collapse_funcs=hap_collapse_funcs,
                            keep_genes_separate=keep_genes_separate,
                            low_memory=low_memory)

    return data

def batch_dict_to_device(batch_dict, device):
    for k,v in batch_dict.items():
        if torch.is_tensor(v):
            batch_dict[k] = v.to(device).float()
        elif isinstance(v, dict):
            batch_dict_to_device(v, device)
            
def batch_dict_to_float(batch_dict):
    for k,v in batch_dict.items():
        if torch.is_tensor(v):
            batch_dict[k] = v.float()
        elif isinstance(v, dict):
            batch_dict_to_float(v)
            
def read_gene_list(file):
    with open(file, 'r') as file:
        gene_list = list([x.strip() for x in file.readlines()])
    return gene_list
        