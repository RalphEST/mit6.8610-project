import transformers

import torch
from torch import cuda
from torch.utils.data import TensorDataset
from transformers import EsmTokenizer, EsmModel
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

########GLOBAL Params#################
models={
	"esm2small":"facebook/esm2_t6_8M_UR50D",
	"esm2medium":"facebook/esm2_t12_35M_UR50D",
	"esm2large":"facebook/esm2_t33_650M_UR50D",
	"esm1v":"facebook/esm1v_t33_650M_UR90S_1"
}
#model_small = "facebook/esm2_t6_8M_UR50D"
#model_medium = "facebook/esm2_t12_35M_UR50D"
#model_large = "facebook/esm2_t33_650M_UR50D"
#model_variant = "facebook/esm1v_t33_650M_UR90S_1"
#model_pth = model_small
count = 0
verbose = 0
average = True

OUTPUT_DIR = "./"

#####################################


def tokenizer_function(input_data,tokenizer):
  input_ids = []
  attention_masks = []
  for seq in input_data:
    this_encoding = tokenizer.encode_plus(seq,return_attention_mask = True,return_tensors = 'pt')
    input_ids.append(this_encoding['input_ids'])
    attention_masks.append( this_encoding['attention_mask'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  # labels = torch.tensor(labels)
  tokenized_data = TensorDataset(input_ids, attention_masks)
  return tokenized_data

def parse_args():
    parser = argparse.ArgumentParser()
    
    # common args
    
    parser.add_argument("--pretrained_model", type=str, default='esm1v', help="pretrained model",choices=["esm2small","esm2medium","esm2large","esm1v"])
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output directory for results")
    parser.add_argument("--average",action='store_true', help="Average the embeddings")
    parser.add_argument("--input",type=str,required=True,help = "input file containing the sequences")

    
    args = parser.parse_args()
    
    return args

def main(args):
	#Check whether running in GPU
	if cuda.is_available():
	  device = 'cuda'
	else:
	  print('WARNING: you are running this code on a cpu!')
	  device = 'cpu'
	#Loading Model
	model_pth = models[args.pretrained_model]
	tokenizer = EsmTokenizer.from_pretrained(model_pth)
	model = EsmModel.from_pretrained(model_pth)
	model.to(device)
	

	input_data = np.load(args.input)
	#Tokenization	
	tokenized_data = tokenizer_function(input_data,tokenizer)

	batch_size = 1
	Inference_Loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)

	embeddings = []

	print("Starting inference")
	for batch in Inference_Loader:
	  #
	  # `batch` contains two pytorch tensors:
	  #   [0]: input ids 
	  #   [1]: attention masks

	  b_input_ids = batch[0].to(device)
	  b_input_mask = batch[1].to(device)
	       
	  outputs = model(b_input_ids, attention_mask=b_input_mask)
	  embedding = outputs.last_hidden_state
	  embedding = embedding.detach().cpu().numpy()
	  # print(embedding.shape)
	  
	  if(average == True):
	  	embedding = np.mean(embedding,axis=1)
	  	
	  embeddings.append(embedding)
	  global count
	  count+=1
	  
	  if(verbose):
	  	print("Done upto batch",count)

	embeddings = np.array(embeddings)
	print(embeddings.shape,embeddings[0].shape)

	np.save("{}/ESMEmbed.npy".format(args.output_dir),embeddings)

if __name__ == "__main__":
    # Pretrained model can be 4 in nature esm2small, esm2medium, esm2large or esm1-v
    # The input is parsed as a mutated sequence matrix preferably in numpy format (NOT ONE-HOT ENCODED)
  
    args = parse_args()
    # print(args)
    main(args)
