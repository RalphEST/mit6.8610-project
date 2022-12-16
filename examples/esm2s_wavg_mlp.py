# basics + plotting
import os, sys
import json
import pandas as pd
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# custom
PROJECT_PATH = '/n/groups/marks/databases/ukbiobank/users/ralphest/mit6.8610-project'
sys.path.insert(1, PROJECT_PATH)

from utils import (
    data_utils, 
    eval_utils, 
    plotting_utils, 
    train_test_utils
)

from models import (
    esm_transfer_learning,
    gnn
)

data = data_utils.load_variation_dataset("../data/data/", 
                                         "../gene_list.txt", 
                                         ["embeddings", "indicators"], 
                                         "../data/phenotypes_hcm_only.parquet",
                                         predict=["hcm"], 
                                         low_memory=True,
                                         embeddings_file='esm2s_embeddings.npy',
                                         ppi_graph_path='../ppi_networks/string_interactions_short.tsv')

train_dataset, test_dataset = data.train_test_split(balance_on=['hcm','ethnicity'])

batch_size = 128

train_loader = DataLoader(
    dataset = train_dataset, 
    batch_size = batch_size,
    sampler = WeightedRandomSampler(train_dataset.weights('hcm',flatten_factor=1), num_samples = len(train_dataset)),
    num_workers=12
)
    
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=12
)

mlp_params = {
    'in_dim': 0,
    'hidden_dims': [512],
    'out_dim':0,
    'actn': 'gelu'
}

model = esm_transfer_learning.ESMTransferLearner(
    esm_model_name="esm2_t6_8M_UR50D",
    agg_emb_method="weighted_average", 
    predict_method="mlp",
    n_genes=29,
    predictor_params=mlp_params,
    add_residue_features=data.get_feature_dimensions(['indicators']),
    edge_index=data.edge_index.to('cuda')
)

train_test_utils.train(model, 
      train_loader,
      test_loader,
      save_model_to = '../data/results/esm2s_wavg_mlp',
      save_metrics_to = 'esm2s_wavg_mlp',
      log_every = 10,
      lr=1e-3, 
      n_epochs=6)