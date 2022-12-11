import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import pretrained

class ProteinEmbeddingPredictor(nn.Module):
    def __init__(self,collapse_method, predict_method, esm_model='esm2_t12_35M_UR50D', first_finetune_layer=-1):
        """
        Parameters:
            - first_fine: first layer to finetune (index starts from 1). -1 means no finetuning, 0 means finetune all layers including embedding layer.
        """
        self.esm_embed, self.alphabet = pretrained.load_model_and_alphabet_hub(esm_model)
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
        self.collapse_embeddings = ProteinEmbed(collapse_method, ...)
        self.predictor = EmbeddingPredict(predict_method, ...)
    
    def forward(self, batch_dict, ...):
        # x (B, L, E) ---> (B, E)
        out = self.esm_embed(tokens, repr_layers=[-1])
        token_embeddings = out['representations']
        protein_embeddings = self.collapse_embeddings(token_embeddings, lens)  # lens is a list of length of proteins in the batch.  Do all aggregations within protein_embeddings.
        
        # protein_embeddings = [self.collapse_embeddings(x) for x in batch_dict['embeddings'].values()]
        # protein_embeddings = torch.cat(protein_embeddings, axis = -1)
        
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
 