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

import mlp
import gnn

class ESMTransferLearner(nn.Module):
    def __init__(self,
                 esm_model_name,
                 agg_emb_method, 
                 predict_method,
                 n_genes,
                 predictor_params,
                 concatenate_features=[],
                 edge_index = None,
                 first_finetune_layer=None):
        """
        Inputs:
        * `esm_model_name`:
        * `agg_emb_method`:
        * `predict_method`:
        * `predict_params` :
        * `concatenate_features`:
        * `first_finetune_layer`: 
            first layer to finetune. Indexing starts with 0. Use -1 to finetune only the last layer.
        """
        super(ESMTransferLearner, self).__init__() 
        
        self.agg_emb_method = agg_emb_method
        self.predict_method = predict_method
        self.predictor_params = predictor_params
        self.esm_model_name = esm_model_name
        self.n_genes = n_genes
        predictor_params['n_genes'] = n_genes
        self.first_finetune_layer = first_finetune_layer
        
        if first_finetune_layer:
            self.finetune_esm = FinetuneESM(esm_model_name, first_finetune_layer)
                
        self.agg_embeddings = ProteinEmbeddingsGenerator(esm_model_name, 
                                                         agg_emb_method,
                                                         n_genes,
                                                         concatenate_features)
        self.predictor = ProteinEmbeddingsPredictor(esm_model_name,
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
                 concatenate_features):
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
        * `concatenate_features` (list[str,...]):
            List of other data in the batch dictionary which can be concatenated to the embeddings before
            they are aggregated. Concatenation happens along the last dimension of the embeddings.
        
        '''
        super(ProteinEmbeddingsGenerator, self).__init__()
        
        self.esm_model_name = esm_model_name
        self.agg_emb_method = agg_emb_method
        self.concatenate_features = concatenate_features
        
        if agg_emb_method == "weighted_average":
            self.weight_logits = [nn.Linear(in_features = constants.model_to_embed_dim[esm_model_name],
                                          out_features = 1)
                                  for i in range(n_genes)]

            self.softmax = nn.Softmax(dim=1)
    
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
            features in `concatenate_features` as first-level keys. 
            Second-level keys are the genes of interest.
        Returns:
        * A dictionary with genes as keys and protein-level embeddings as values. 
          The embeddings are tensors of shape (batch_size, emb_size)
        '''
        protein_embeddings = {}
        for gene_idx, (gene, embeddings) in enumerate(batch_dict['embeddings'].items()):
            for feature in self.concatenate_features:
                feature_tensor = batch_dict[feature][gene]
                
                # make sure the feature tensor is 3-dimensional
                if len(feature_tensor.shape) == 2:
                    feature_tensor = feature_tensor.unsqueeze(-1)
                    
                embeddings = torch.cat([embeddings, feature_tensor], dim = -1)
            protein_embeddings[gene] = self.forward_single_gene(embeddings, gene_idx)
        return protein_embeddings
    
class ProteinEmbeddingsPredictor(nn.Module):
    def __init__(self,
                 esm_model_name,
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
                    NOT IMPLEMENTED YET.
        * `edge_index` (pytorch tensor)
        '''
        super(ProteinEmbeddingsPredictor, self).__init__()
        
        self.esm_model_name = esm_model_name
        self.predict_method = predict_method
        self.predict_params = predictor_params
        self.edge_index = edge_index
        
        if predict_method == 'mlp':
            self.predictor = mlp.MLP(predictor_params)
        elif predict_method == 'gat':
            self.predictor = mlp.GAT(predictor_params)
        elif predict_method == 'gcn':
            raise NotImplementedError # TODO
            
    def forward(self, protein_embeddings):
        
        
    

    
class FinetuneESM(nn.Module):
    def __init__(self, esm_model, first_finetune_layer):
        
         # handle fine-tuning
        self.esm, self.alphabet = pretrained.load_model_and_alphabet_hub(esm_model)
        
        if first_finetune_layer != -1 and first_finetune_layer != 0:
            finetune_layers = set(range(first_finetune_layer-1, len(self.esm_embed.layers)))
        elif first_finetune_layer == 0:
            finetune_layers = set(range(len(self.esm_embed.layers)))
        for k, v in self.esm_embed.named_parameters():
            if first_finetune_layer == -1:
                v.requires_grad = False
            elif k.split('.')[1] in finetune_layers:
                v.requires_grad = False
            elif first_finetune_layer != 0 and 'embed' in k:
                v.requires_grad = False
 