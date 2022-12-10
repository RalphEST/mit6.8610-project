import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinEmbeddingPredictor(Module):
    def __init__(self,collapse_method, predict_method, fine_tune=False, esm_model):
        
        
        if fine_tune:
            self.esm_embed = ESM()
        self.collapse_embeddings = ProteinEmbed(collapse_method, ...)
        self.predictor = EmbeddingPredict(predict_method, ...)
    
    def forward(self,batch_dict):
        # x (B, L, E) ---> (B, E)
        protein_embeddings = [self.collapse_embeddings(x) for x in batch_dict['embeddings'].values()]
        protein_embeddings = torch.cat(protein_embeddings, axis = -1)
        return EmbeddingPredict(protein_embeddings)
    
    
class ProteinEmbed(Module):
    def __init__(self,collapse_method...)
        if collapse_method == "mean":
            ...
        elif collapse_method == "weighted_avg":
            ...
        elif 
    
    
class EmbeddingPredict(Module):
    def __init__(self,predict_method...)
    

    
class ESM(Module):
    def __init__(self, esm_model, ...)
 