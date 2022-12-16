import logging
import os, sys
import json
import pandas as pd
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# custom
PROJECT_PATH = '/n/groups/marks/databases/ukbiobank/users/ralphest/mit6.8610-project'
sys.path.insert(1, PROJECT_PATH)

from utils import (
    data_utils, 
    eval_utils, 
    plotting_utils, 
    train_test_utils
)

from models import (
    esm_transfer_learning,
    gnn
)

# logging utils

def get_root_logger(fname=None, file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler(f"{fname}.txt", mode='w')
        handler.setFormatter(format)
        logger.addHandler(handler)

    logger.addHandler(logging.StreamHandler())
    
    return logger

# (supervised) training
def train_epoch(model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                device,
                log_every=10):
    model.train()
    total_loss = 0
    all_labels, all_preds, all_losses = [],[],[]
    
    for i, batch in enumerate(train_loader):
        # move batch dictionary to device
        data_utils.batch_dict_to_device(batch, device)
        labels = batch['labels']
        
        # compute prediction and loss
        preds = model(batch)
        loss = loss_fn(preds, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # tracking
        total_loss += loss.item()
        all_losses.append(loss.item())
        all_labels.append(labels.cpu())
        all_preds.append(preds.flatten().detach().cpu())
        
        # logging
        if (i % log_every == 0):
            print(f"\tBatch {i} | BCE Loss: {loss.item():.4f}")
    
    
    metrics = eval_utils.get_metrics(torch.cat(all_labels), 
                                     torch.cat(all_preds))
    metrics['loss'] = total_loss
    
    return metrics, all_losses

def test(model, 
         test_loader, 
         loss_fn,
         device):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [],[]
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # move batch dictionary to device
            data_utils.batch_dict_to_device(batch, device)
            labels = batch['labels']

            # compute prediction and loss
            preds = model(batch)
            loss = loss_fn(preds, labels)

            # tracking
            total_loss += loss.item()
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.flatten().detach().cpu())
    
    metrics = eval_utils.get_metrics(torch.cat(all_labels), 
                                     torch.cat(all_preds))
    metrics['loss'] = total_loss
    
    return metrics

def train(model, 
          train_loader,
          test_loader,
          save_model_to,
          save_metrics_to,
          log_every,
          lr=1e-3, 
          n_epochs=16):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    track_metrics = {'train':{str(i):None for i in range(n_epochs)}, 
                     'test': {str(i):None for i in range(n_epochs)}}
    all_losses = []
    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}:")
        train_metrics, epoch_losses = train_epoch(model, 
                                                train_loader, 
                                                optimizer, 
                                                loss_fn,
                                                device,
                                                log_every=log_every)
        print("Train metrics:")
        eval_utils.print_metrics(train_metrics)
        
        all_losses.append(epoch_losses)
        np.save(save_metrics_to + '_trainloss.npy', np.array(all_losses))
        torch.save(model.state_dict(), save_model_to + '_model.pt')
        
        test_metrics = test(model, 
                            test_loader, 
                            loss_fn,
                            device)
        print("Test metrics:")
        eval_utils.print_metrics(test_metrics)
        
        track_metrics['train'][str(epoch)] = train_metrics
        track_metrics['test'][str(epoch)] = test_metrics
        pd.DataFrame(track_metrics['train']).to_parquet(save_metrics_to + '_train.parquet')
        pd.DataFrame(track_metrics['test']).to_parquet(save_metrics_to + '_test.parquet')
    
    return track_metrics
