import torch
from torch import cuda
from torch_geometric.nn import GCNConv, GATv2Conv
from torch import nn
import mlp

class GATv2(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dim:int,
                 n_heads:int,
                 n_genes:int,
                 mlp_hidden_dims:list,
                 mlp_actn:str):
        
        super(GATv2, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.leakyrelu_slope = leakyrelu_slope
        self.n_genes = n_genes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_actn = mlp_actn
        
        self.input_layer = [nn.Linear(in_features=self.in_dim,
                                      out_features=self.embed_dim)
                            for _ in range(n_genes)]
        self.gatv2conv_layer = GATv2Conv(in_channels=self.embed_dim,
                                         out_channels=self.embed_dim,
                                         heads=n_heads,
                                         concat=False,
                                         share_weights=False)
        
        self.output_mlp = mlp.MLP(in_dim=self.embed_dim * n_genes,
                                  hidden_dims=mlp_hidden_dims,
                                  out_dim=1,
                                  actn=mlp_actn)
        
    def forward(self, protein_embeddings, edge_index):
        
        
        
        x = torch.relu(self.input_layer(x))
        x = self.leakyrelu(self.GATLayer1(x, edge_index))
        x = torch.flatten(x)
        x = torch.relu(self.output_layer1(x))
        x = self.output_layer(x)
        return x
    
    # def avg_attention_heads(self, x):
        # return torch.mean(x, dim = 1)
        
    