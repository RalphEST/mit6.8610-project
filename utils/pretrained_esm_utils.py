# This script is based on 
# https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
# from Facebook Research

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
    pretrained_model,
    fasta_file,
    include_data,
    output_files,
    toks_per_batch=4096,
    truncation_seq_length=2500):
    
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
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            
            buffer = {'tokens':None, 
                      'logits':None, 
                      'embeddings':None}
            buffer['tokens'] = toks
            
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            buffer['logits'] = out["logits"].to(device="cpu")
            buffer['embeddings'] = out["representations"][repr_layer].to(device="cpu")
            
            for d in include_data:
                output_data[d].append(buffer[d])
                
        for d in include_data:
            print(f"Saving {d} file {(data_to_file[d])}")
            torch.save(
                torch.cat(output_data[d], dim=0),
                data_to_file[d]
            )

def tokenize_and_embed_many_genes(
    model_name,
    paths_dict,
    include_data,
    output_files,
    toks_per_batch=4096,
    truncation_seq_length=2500):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    for gene,paths in paths_dict.items():
        done_file = os.path.join(paths['out_dir'],model_name+'.done')
        if Path(done_file).exists():
            continue
        print(f"Processing gene {gene}")
        tokenize_and_embed(
            (model, alphabet),
            paths['fasta'],
            include_data,
            [os.path.join(paths['out_dir'],f) for f in output_files],
            toks_per_batch=4096,
            truncation_seq_length=2500
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