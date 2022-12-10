import torch
from torch import cuda
from torch_geometric.nn import GCNConv, GATv2Conv
from torch import nn


def build_graph(graph,node_embedding):
    G = from_networkx(graph)
    
    G.x = torch.Tensor(node_embedding)

    G = G.to(device)
    return G

# hparams = {
#     "input_size":cuts.shape[2],
#     "hsz":500,
#     "alpha": 0.2,
#     "num_heads":5,
#     "bsz":16,
#     "num_epochs":100,
#     "lr":1e-3,
#     "ngenes":10
# }

class GAT_Protein(nn.Module):
    def __init__(self,
                 input_size:int,
                 hid_dim:int,
                 alpha:float,
                 num_heads:int,
                 ngenes:int
                ):
        super(GAT_Protein, self).__init__()
        self.in_dim = input_size
        self.hsz = hid_dim
        self.num_heads = num_heads
        self.alpha = alpha
        
        self.input_layer = nn.Linear(self.in_dim,self.hsz)
        self.GATLayer1 = GATv2Conv(self.hsz,self.hsz)
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
        
    