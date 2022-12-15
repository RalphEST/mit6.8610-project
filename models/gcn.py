class GCN_Protein(nn.Module):
    def __init__(self,
                 in_dim:int,
                 embed_dims:int,
                 n_genes:int):
        
        super(GCN_Protein, self).__init__()
        
        self.in_dim = in_dim
        self.embed_dims = embed_dims
        
        self.n_genes = n_genes
        
        self.input_layer = nn.Linear(in_features=self.in_dim,
                                     out_features=self.embed_dims)
                            
        self.gcn_layer = GCNConv(in_channels=self.embed_dims,
                                     out_channels=self.embed_dims)
        
        self.output_layer1 = nn.Linear(self.embed_dims*ngenes,self.embed_dims)
        self.output_layer = nn.Linear(self.embed_dims,1)
        
    def forward(self, x, edge_index):
        # print(x)
        # print(type(x))
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.gcn_layer(x, edge_index))
        x = torch.flatten(x)
        x = torch.relu(self.output_layer1(x))
        x = self.output_layer(x)
        return x
    