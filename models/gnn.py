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
        self.n_genes = n_genes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_actn = mlp_actn
        
        self.input_layers = [nn.Linear(in_features=self.in_dim,
                                      out_features=self.embed_dim)
                            for _ in range(n_genes)]
        self.leakyrelu = nn.LeakyReLU()
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
        x = torch.stack([self.input_layers[i](emb) 
                        for i, emb in enumerate(protein_embeddings.values())],
                        dim=0)
        x = self.gatv2conv_layer(torch.transpose(self.leakyrelu(x), 
                                                 dim0=0, 
                                                 dim1=1), 
                                 edge_index)
        preds = self.output_mlp(torch.flatten(x, 
                                              start_dim=1, 
                                              end_dim=2))
        
        return preds
    
class GCN(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dim:int,
                 n_genes:int,
                 mlp_hidden_dims:list,
                 mlp_actn:str):
        
        super(GCN, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_genes = n_genes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_actn = mlp_actn
        
        self.input_layers = [nn.Linear(in_features=self.in_dim,
                                      out_features=self.embed_dim)
                            for _ in range(n_genes)]
        self.leakyrelu = nn.LeakyReLU()
        self.gcnconv_layer = GCNConv(in_channels=self.embed_dim,
                                     out_channels=self.embed_dim)
        self.output_mlp = mlp.MLP(in_dim=self.embed_dim * n_genes,
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
    