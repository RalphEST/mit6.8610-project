import torch
from torch import cuda
from torch_geometric.nn import GCNConv, GATv2Conv
from torch import nn

class GATv2(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dims:int,
                 leakyrelu_slope:float,
                 n_heads:int,
                 n_genes:int):
        
        super(GATv2, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.leakyrelu_slope = leakyrelu_slope
        self.n_genes = n_genes
        
        self.input_layer = [nn.Linear(in_features=self.in_dim,
                                     out_features=self.embed_dims)
                            for _ in range(n_genes)]
        self.gatv2_layer = GATv2Conv(in_channels=self.embed_dims,
                                     out_channels=self.embed_dims)
        
        self.output_layer1 = nn.Linear(self.hsz*ngenes,self.hsz)
        self.output_layer = nn.Linear(self.hsz,1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, edge_index):
        
        x = torch.relu(self.input_layer(x))
        x = self.leakyrelu(self.GATLayer1(x, edge_index))
        x = torch.flatten(x)
        x = torch.relu(self.output_layer1(x))
        x = self.output_layer(x)
        return x
    
    # def avg_attention_heads(self, x):
        # return torch.mean(x, dim = 1)
        
    