import os, sys
import torch
from torch import cuda
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data as graphData
from torch_geometric.data import Batch as graphBatch
from torch import nn

# custom
PROJECT_PATH = '/n/groups/marks/databases/ukbiobank/users/ralphest/mit6.8610-project'
sys.path.insert(1, PROJECT_PATH)

from models import (
    mlp
)

class GATv2(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dim:int,
                 n_heads:int,
                 n_nodes:int,
                 mlp_hidden_dims:list,
                 mlp_actn:str):
        
        super(GATv2, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_actn = mlp_actn
        
        self.input_layers = nn.ModuleList([nn.Linear(in_features=self.in_dim,
                                      out_features=self.embed_dim)
                            for _ in range(n_nodes)])
        self.leakyrelu = nn.LeakyReLU()
        self.gatv2conv_layer = GATv2Conv(in_channels=self.embed_dim,
                                         out_channels=self.embed_dim,
                                         heads=n_heads,
                                         concat=False,
                                         share_weights=False)
        self.output_mlp = mlp.MLP(in_dim=self.embed_dim * n_nodes,
                                  hidden_dims=mlp_hidden_dims,
                                  out_dim=1,
                                  actn=mlp_actn)
        
    def build_gatv2conv_batch(self, x, edge_index):
        return graphBatch.from_data_list([graphData(x=xi, edge_index=edge_index) for xi in x])
        
    def forward(self, protein_embeddings, edge_index):
        x = torch.stack([self.input_layers[i](emb) 
                            for i, emb in enumerate(protein_embeddings.values())], 
                        dim=0)
        batch_size = x.shape[1]
        x = torch.transpose(self.leakyrelu(x), dim0=0, dim1=1)
        g = self.build_gatv2conv_batch(x, edge_index)
        x = self.gatv2conv_layer(g.x, g.edge_index)
        x = x.reshape(batch_size, self.n_nodes, self.embed_dim)
        preds = self.output_mlp(torch.flatten(x, 
                                              start_dim=1, 
                                              end_dim=2))
        
        return preds
    
class GCN(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dim:int,
                 n_nodes:int,
                 mlp_hidden_dims:list,
                 mlp_actn:str):
        
        super(GCN, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_nodes = n_nodes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_actn = mlp_actn
        
        self.input_layers = nn.ModuleList([nn.Linear(in_features=self.in_dim,
                                      out_features=self.embed_dim)
                            for _ in range(n_nodes)])
        self.leakyrelu = nn.LeakyReLU()
        self.gcnconv_layer = GCNConv(in_channels=self.embed_dim,
                                     out_channels=self.embed_dim,
                                     node_dim=1)
        self.output_mlp = mlp.MLP(in_dim=self.embed_dim * n_nodes,
                                  hidden_dims=mlp_hidden_dims,
                                  out_dim=1,
                                  actn=mlp_actn)
        
    def forward(self, protein_embeddings, edge_index):
        x = torch.stack([self.input_layers[i](emb) 
                        for i, emb in enumerate(protein_embeddings.values())],
                        dim=0)
        x = self.gcnconv_layer(torch.transpose(self.leakyrelu(x), 
                                               dim0=0, 
                                               dim1=1), 
                               edge_index)
        preds = self.output_mlp(torch.flatten(x, 
                                              start_dim=1, 
                                              end_dim=2))
        
        return preds
    