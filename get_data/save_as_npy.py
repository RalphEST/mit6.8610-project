import glob
import torch
import numpy as np

with open("../gene_list.txt", 'r') as file:
    gene_list = [x.strip() for x in file.readlines()]
    
for gene in gene_list:
    file_list = glob.glob(f"../data/data/{gene}/esm*")
    for file in file_list:
        if (file[-5:] != '.done') and (file[-4:] != '.npy'):
            print(f'Saving {file} ...', end=" ")
            data = torch.load(file)
            np.save(file+'.npy', data.numpy())
            print("Done")