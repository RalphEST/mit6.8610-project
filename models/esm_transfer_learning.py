import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import pretrained

# custom
PROJECT_PATH = '/n/groups/marks/databases/ukbiobank/users/ralphest/mit6.8610-project'
sys.path.insert(1, PROJECT_PATH)

from utils import (
    constants
)
from models import (
    mlp,
    gnn
)

class ESMTransferLearner(nn.Module):
    def __init__(self,
                 esm_model_name,
                 agg_emb_method, 
                 predict_method,
                 n_genes,
                 predictor_params,
                 add_residue_features={},
                 add_protein_features={},
                 edge_index=None,
                 first_finetune_layer=None,
                 remove_bos_eos=True):
        """
        Inputs:
        * `esm_model_name`:
        * `agg_emb_method`:
        * `predict_method`:
        * `n_genes`:
        * `predict_params` :
        * `add_residue_features`:
            Dictionary of the form `{feature_name: feature_dim}`. Can be obtained from the 
            `get_feature_dims` method in the `VariationDataset` class. Features will be added to
            the residue embeddings before aggregation. They must therefore have dimensions 
            (batch_size, gene_length, feature_dim). If the last dimension is missing (input is a 
            2-d matrix), an extra third dimension is added automatically.
        * `add_protein_features`:
            Dictionary of the form `{feature_name: feature_last_dim}`. Can be obtained from the 
            `get_feature_dims` method in the `VariationDataset` class. Features will be added to 
            the protein embeddings.
        * `edge_index`:
        * `first_finetune_layer`: 
            First layer to finetune. Indexing starts with 0. Like with a list, use `-X` 
            to finetune only the last `X` layers.
        """
        super(ESMTransferLearner, self).__init__() 
        
        self.agg_emb_method = agg_emb_method
        self.predict_method = predict_method
        self.predictor_params = predictor_params
        self.esm_model_name = esm_model_name
        self.n_genes = n_genes
        self.add_residue_features = add_residue_features
        self.add_protein_features = add_protein_features
        self.first_finetune_layer = first_finetune_layer

        
        if first_finetune_layer:
            self.finetune_esm = FinetuneESM(esm_model_name, first_finetune_layer)
                
        self.agg_embeddings = ProteinEmbeddingsGenerator(esm_model_name, 
                                                         agg_emb_method,
                                                         n_genes,
                                                         add_residue_features,
                                                         add_protein_features,
                                                         remove_bos_eos)
        
        protein_embedding_size = self.agg_embeddings.protein_embedding_size
        self.predictor = ProteinEmbeddingsPredictor(esm_model_name,
                                                    n_genes,
                                                    protein_embedding_size,
                                                    predict_method,
                                                    predictor_params,
                                                    edge_index)
    
    def forward(self, batch_dict):
        
        if self.first_finetune_layer:
            batch_dict['embeddings'] = self.finetune_esm(batch_dict)
        protein_embeddings = self.agg_embeddings(batch_dict)
        preds = self.predictor(protein_embeddings)

        return preds
    
    
class ProteinEmbeddingsGenerator(nn.Module):
    def __init__(self, 
                 esm_model_name, 
                 agg_emb_method,
                 n_genes,
                 add_residue_features,
                 add_protein_features, 
                 remove_bos_eos):
        '''
        Inputs:
        * `esm_model_name` (str):
            Name of ESM model used.
        * `agg_emb_method` (str):
            Aggregation method to get protein-wise embeddings. Choices include:
                - 'average':
                    Average across all protein positions.
                - 'weighted_average':
                    A simple attention-like mechanism in which one weight is computed per embedding using
                    a linear layer followed by a softmax.
        * `add_residue_features` (dict):
            Dictionary of the form `{feature_name: feature_last_dim}`. Can be obtained from the 
            `get_feature_dims` method in the `VariationDataset` class.
        
        '''
        super(ProteinEmbeddingsGenerator, self).__init__()
        
        self.esm_model_name = esm_model_name
        self.agg_emb_method = agg_emb_method
        self.n_genes = n_genes
        self.remove_bos_eos = remove_bos_eos
        
        # process feature dimensions
        self.add_residue_features = {k:v[-1] if len(v)==3 else 1 for k, v in add_residue_features.items()}
        self.add_protein_features = {k:v[-1] if len(v)==2 else 1 for k, v in add_protein_features.items()}   
        
        residue_embed_dim = constants.model_to_embed_dim[esm_model_name] + sum(self.add_residue_features.values())
        if agg_emb_method == "weighted_average":
            self.weight_logits = nn.ModuleList([nn.Linear(in_features = residue_embed_dim,
                                                          out_features = 1)
                                  for i in range(n_genes)])

            self.softmax = nn.Softmax(dim=1)
        self.protein_embedding_size = residue_embed_dim + sum(self.add_protein_features.values())
    
    def forward_single_gene(self, embeddings, gene_idx):
        '''
        Inputs: 
        * `embeddings` of shape (batch_size, protein_length, emb_size)
        Returns:
        * aggregated embeddings of shape (batch_size, emb_size)
        '''
        # embeddings.shape = (B, L, E)
        if self.agg_emb_method == "weighted_average":
            weights = self.weight_logits[gene_idx](embeddings)
            weights = self.softmax(weights) # weights.shape = (B, L, 1)
            return (embeddings * weights).sum(1)
        elif self.agg_emb_method == "average":
            return embeddings.mean(1)
            
    
    def forward(self, batch_dict):
        '''
        Inputs:
        * `batch_dict`: 
            Two-level dictionary with at least 'embeddings' and the 
            features in `add_residue_features` as first-level keys. 
            Second-level keys are the genes of interest.
        Returns:
        * A dictionary with genes as keys and protein-level embeddings as values. 
          The embeddings are tensors of shape (batch_size, emb_size)
        '''
        protein_embeddings = {}
        for gene_idx, (gene, embeddings) in enumerate(batch_dict['embeddings'].items()):
            if self.remove_bos_eos:
                embeddings = embeddings[:, 1:-1, :]
            for feature in self.add_residue_features.keys():
                feature_data = batch_dict[feature][gene]
                
                # make sure the feature tensor is 3-dimensional
                if len(feature_data.shape) == 2:
                    feature_data = feature_data.unsqueeze(-1)
                
                # embeddings.shape = (batch_size, gene_length, emb_size)
                # feature_data.shape = (batch_size, gene_length, feature_dim)
                
                embeddings = torch.cat([embeddings, feature_data], dim = -1)
                
            protein_emb = self.forward_single_gene(embeddings, gene_idx)
            
            for feature in self.add_protein_features.keys():
                feature_data = batch_dict[feature][gene]
                
                # make sure the feature tensor is 2-dimensional
                if len(feature_data.shape) == 1:
                    feature_data = feature_data.unsqueeze(-1)
                
                # protein_emb.shape = (batch_size, emb_size)
                # feature_data.shape = (batch_size, feature_dim)
                protein_emb = torch.cat([protein_emb, feature_data], dim = -1)
                
            protein_embeddings[gene] = protein_emb
            
        return protein_embeddings
    
class ProteinEmbeddingsPredictor(nn.Module):
    def __init__(self,
                 esm_model_name,
                 n_genes,
                 protein_embedding_size,
                 predict_method,
                 predictor_params,
                 edge_index):
        '''
        Inputs:
        * `esm_model_name` (str)
        * `predict_method` (str):
            Method to classify gene network based on protein embeddings. Choices include:
                - 'mlp': 
                    concatenate protein embeddings and classify using a multilayer perceptron
                - 'gat':
                    concatenate protein embeddings and classify using a graph-attention network
                - 'gcn':
                    concatenate protein embeddings and classify using a graph-convolutional network.
        * `edge_index` (pytorch tensor)
        '''
        super(ProteinEmbeddingsPredictor, self).__init__()
        
        self.esm_model_name = esm_model_name
        self.n_genes = n_genes
        self.protein_embedding_size = protein_embedding_size
        self.predict_method = predict_method
        self.predict_params = predictor_params
        self.edge_index = edge_index
        
        if predict_method == 'mlp':
            predictor_params['in_dim'] = n_genes * protein_embedding_size
            predictor_params['out_dim'] = 1
            self.predictor = mlp.MLP(**predictor_params)
        elif predict_method == 'gat':
            predictor_params['in_dim'] = protein_embedding_size
            predictor_params['n_nodes'] = n_genes
            self.predictor = gnn.GATv2(**predictor_params)
        elif predict_method == 'gcn':
            predictor_params['in_dim'] = protein_embedding_size
            predictor_params['n_nodes'] = n_genes
            self.predictor = gnn.GCN(**predictor_params)
        else:
            raise ValueError("predict_method can be either 'mlp', 'gat', or 'gcn'")
            
    def forward(self, protein_embeddings):
        if self.predict_method == 'mlp':
            x = torch.cat(list(protein_embeddings.values()), dim=1)
            return self.predictor(x)
        else:
            return self.predictor(protein_embeddings, self.edge_index)

    
class FinetuneESM(nn.Module):
    def __init__(self, 
                 esm_model_name, 
                 first_finetune_layer):
        super().__init__()
        
        # handle fine-tuning
        self.esm, self.alphabet = pretrained.load_model_and_alphabet_hub(esm_model_name)
        self.num_layers = self.esm.num_layers
        self.first_finetune_layer = first_finetune_layer
        
        self.finetune_layers = [str(layer) for layer in np.arange(self.num_layers)[first_finetune_layer:]]
        
        print("Preparing for fine-tuning:")
        for k, v in self.esm.named_parameters():
            if (k=='embed_tokens.weight') or (k.split('.')[1] not in self.finetune_layers):
                print(f"- Freezing layer {k}")
                v.requires_grad = False
            else:
                print(f"+ Not freezing layer {k}")

    def forward(self, batch_dict):
        '''
        Inputs:
        * `batch_dict`: of the form {'esm-tokens': {gene_name: tensor of shape (batch_size, gene_length + 2)}}
        Returns:
        * {gene_name:  tensor of shape (batch_size, gene_length + 2, vocab_size)}
        '''
        residue_embeddings = {}
        for gene, toks in batch_dict['esm-tokens'].items():
            results = self.esm_embed(toks, repr_layers=[self.num_layers])
            residue_embeddings[gene] = results["representations"][self.num_layers]
        return residue_embeddings 
 