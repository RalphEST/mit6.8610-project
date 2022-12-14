# Some parts of this script are adapted from 
# https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
# from Facebook Research, but most of it has been modified for our purposes

import numpy as np
import os
import argparse
import pathlib
import torch
from pathlib import Path
from esm import (
    Alphabet, 
    FastaBatchedDataset, 
    ProteinBertModel, 
    pretrained, 
    MSATransformer
)

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=['esm2_t6_8M_UR50D', 
                 'esm2_t12_35M_UR50D', 
                 'esm2_t33_650M_UR50D', 
                 'esm1v_t33_650M_UR90S_1'],
        help= "Name of pretrained model to download",
    )
    parser.add_argument(
        "include_data",
        type=str,
        nargs="+",
        choices=["embeddings", "tokens", "logits"],
        help="data to be saved, including tokens, last-layer embeddings, and logits",
    )
    parser.add_argument(
        "output_files",
        type=str,
        nargs="+",
        help="files which each data included will be saved",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "--toks_per_batch", 
        type=int, 
        default=4096, 
        help="maximum batch size"
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=2500,
        help="truncate sequences longer than the given value",
    )
    return parser


def tokenize_and_embed(
    model_name,
    pretrained_model,
    fasta_file,
    include_data,
    output_files,
    toks_per_batch=4096,
    truncation_seq_length=4096):
    
    if isinstance(pretrained_model, str):
        model, alphabet = pretrained.load_model_and_alphabet(pretrained_model)
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
    else:
        model, alphabet = pretrained_model
    
    assert len(include_data) == len(output_files)
    output_data = {k:[] for k in include_data} 
    data_to_file = {d:f for d,f in zip(include_data, output_files)}
    
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=0)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(truncation_seq_length), 
        batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")
    
    # index of the last model layer
    repr_layer = (-1 + model.num_layers + 1) % (model.num_layers + 1)
    
    
    model_to_embed_dim = {
        'esm2_t6_8M_UR50D': 320, 
        'esm2_t12_35M_UR50D': 480, 
        'esm2_t33_650M_UR50D': 1280, 
        'esm1v_t33_650M_UR90S_1': 1280
    }
    
    N, L = len(dataset), len(dataset.sequence_strs[0])+2
    
    if 'tokens' in include_data:
        output_data['tokens'] = torch.empty((N, L), 
                                             dtype=torch.int8, 
                                             device = torch.device('cpu'))
    if 'embeddings' in include_data:
        output_data['embeddings'] = torch.empty((N, L, model_to_embed_dim[model_name]), 
                                                 device = torch.device('cpu'))
    if 'logits' in include_data:
        output_data['logits'] = torch.empty((N, L, len(alphabet.all_toks)), 
                                             device = torch.device('cpu'))
    
    i = 0
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            B = len(labels)
            j = i + B
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            
            if 'tokens' in include_data:
                output_data['tokens'][i:j] = toks
            
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            
            # esm-1v has a sequence length limit of 1024
            # if limit is exceeded, divide protein into 1024aa-long segments and concatenate them
            if (L > 1024) and ("esm1v" in model_name):
                out = {'representations' : {repr_layer:torch.empty(B, L, 1280)},
                       'logits' : torch.empty(B, L, len(alphabet.all_toks))}
                for start in np.arange(0, L, 1024):
                    end = min(start+1024,L)
                    trunc_out = model(toks[:, start:end], repr_layers=[repr_layer], return_contacts=False)
                    out['representations'][repr_layer][:, start:end, :] = trunc_out['representations'][repr_layer]
                    out['logits'][:, start:end, :] = trunc_out['logits']
            else:
                out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            
            if 'embeddings' in include_data:
                output_data['embeddings'][i:j] = out["representations"][repr_layer].to(device="cpu")
            if 'logits' in include_data: 
                output_data['logits'][i:j] = out["logits"].to(device="cpu")
            i = j
            
        assert j==len(dataset)
                
        for d in include_data:
            print(f"Saving {d} file {(data_to_file[d])}")
            torch.save(
                output_data[d],
                data_to_file[d]
            )

def tokenize_and_embed_many_genes(
    model_name,
    paths_dict,
    include_data,
    output_files,
    toks_per_batch=4096,
    truncation_seq_length=4096):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    for gene,paths in paths_dict.items():
        print(f"Processing gene {gene}")
        done_file = os.path.join(paths['out_dir'],model_name+'.done')
        if Path(done_file).exists():
            print(done_file)
            continue
        tokenize_and_embed(
            model_name,
            (model, alphabet),
            paths['fasta'],
            include_data,
            [os.path.join(paths['out_dir'],f) for f in output_files],
            toks_per_batch=toks_per_batch,
            truncation_seq_length=truncation_seq_length
        )
        Path(done_file).touch()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    tokenize_and_embed(
        model_name = args.model_name,
        fasta_file = args.fasta_file,
        include_data = args.include_data,
        output_files = args.output_files,
        toks_per_batch = args.toks_per_batch,
        truncation_seq_length = args.truncation_seq_length
    )