import transformers

import torch
from torch import cuda
from torch.utils.data import TensorDataset
from transformers import EsmTokenizer, EsmModel
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

def tokenizer_function(input_data):
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

#Check whether running in GPU
if cuda.is_available():
  device = 'cuda'
else:
  print('WARNING: you are running this code on a cpu!')
  device = 'cpu'


model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.to(device)
input_data = np.load("data/toy_data/mutseqs.npy")
tokenized_data = tokenizer_function(input_data)

batch_size = 1
Inference_Loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)

embeddings = []
count = 0
verbose = 0
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
  embeddings.append(embedding)

  count+=1
  # if(count == 5):
  # 	break
  if(verbose):
  	print("Done upto batch",count)

embeddings = np.array(embeddings)
print(embeddings.shape,embeddings[0].shape)

np.save("embeddings/toyDataEmbeddings/ESMEmbed.npy",embeddings)