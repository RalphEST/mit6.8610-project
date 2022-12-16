# basics + plotting
import os, sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans serif"

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# custom
PROJECT_PATH = '/'.join(os.getcwd().split('/')[:-1])
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

import importlib
data_utils = importlib.reload(data_utils)
eval_utils = importlib.reload(eval_utils)
train_test_utils = importlib.reload(train_test_utils)
esm_transfer_learning = importlib.reload(esm_transfer_learning)
gnn = importlib.reload(gnn)

data = data_utils.load_variation_dataset("../data/data/", 
                                         "../gene_list.txt", 
                                         ["embeddings", "indicators"], 
                                         "../data/phenotypes_hcm_only.parquet",
                                         predict=["hcm"], 
                                         low_memory=True,
                                         embeddings_file='esm2m_embeddings.npy',
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

gat_params = {
    'in_dim': 0,
    'embed_dim': 256,
    'n_heads': 4,
    'n_nodes': 0,
    'mlp_hidden_dims': [128],
    'mlp_actn': 'gelu'
}

model = esm_transfer_learning.ESMTransferLearner(
    esm_model_name="esm2_t12_35M_UR50D",
    agg_emb_method="weighted_average", 
    predict_method="gat",
    n_genes=29,
    predictor_params=gat_params,
    add_residue_features=data.get_feature_dimensions(['indicators']),
    edge_index=data.edge_index.to('cuda')
)
    

train_test_utils.train(model, 
      train_loader,
      test_loader,
      save_model_to = '../data/results/esm2m_wavg_gat',
      save_metrics_to = 'esm2m_wavg_gat',
      log_every = 10,
      lr=1e-3, 
      n_epochs=6)