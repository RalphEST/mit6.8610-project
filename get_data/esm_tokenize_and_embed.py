import sys, os
PROJECT_PATH = '/'.join(os.getcwd().split('/')[:-1])
sys.path.insert(1, PROJECT_PATH)

from utils import (
    pretrained_esm_utils
)

with open("../gene_list.txt", 'r') as file:
    gene_list = [x.strip() for x in file.readlines()]

paths_dict = {g:{'fasta': f'../data/data/{g}/hap_fasta.fasta',
                     'out_dir': "../data/data/"+g} 
                  for g in gene_list}

models = {
    "esm2_t6_8M_UR50D":{'prefix':'esm2s', 
                        'include_data': ['embeddings', 'logits', 'tokens']},
    "esm2_t12_35M_UR50D":{'prefix':'esm2m', 
                          'include_data': ['embeddings', 'logits']},
    "esm2_t33_650M_UR50D":{'prefix':'esm2l', 
                           'include_data': ['embeddings', 'logits']},
    "esm1v_t33_650M_UR90S_1":{'prefix':'esm1v', 
                              'include_data': ['embeddings', 'logits']}
}

for model, params in models.items():
    pretrained_esm_utils.tokenize_and_embed_many_genes(
        model_name=model,
        paths_dict=paths_dict,
        include_data=params['include_data'],
        output_files=[params['prefix']+'_'+d for d in params['include_data']],
        toks_per_batch=4096,
        truncation_seq_length=2500
    )
    

    