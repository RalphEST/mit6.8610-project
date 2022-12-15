import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import pretrained

class FinetuneESM(nn.Module):
    def __init__(self, esm_model_name, first_finetune_layer=None):
        super().__init__()
        
        # handle fine-tuning
        self.esm_embed, self.alphabet = pretrained.load_model_and_alphabet_hub(esm_model_name)
        self.num_layers = self.esm_embed.num_layers
        
        assert first_finetune_layer is not None, "must provide the first layer to finetune"
        self.first_finetune_layer = first_finetune_layer if first_finetune_layer!=-1 else self.num_layers-1
        
        if self.first_finetune_layer is not None:
            finetune_layers = set([str(layer) for layer in range(self.first_finetune_layer, self.num_layers)])
        
        for k, v in self.esm_embed.named_parameters():
            if k.split('.')[1] not in finetune_layers:
                v.requires_grad = False
            elif 'embed' in k:  # always freeze embedding layer
                v.requires_grad = False
            else:
                print(f"Not freeze {k}")

    def forward(self, batch_dict):
        residue_embeddings = {}
        
        for gene, toks in batch_dict['esm-tokens'].items():
            results = self.esm_embed(toks, repr_layers=[self.num_layers])  #  {gene_name:[batch, gene_len+2, vocab_size]}
            residue_embeddings[gene] = results["representations"][self.num_layers]
        
        return residue_embeddings  # {gene_name:[batch, gene_len+2, feat_dim]}
