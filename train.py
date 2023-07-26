#!/usr/bin/env python

import torch
import gc

import pickle
import csv
import numpy as np
from matplotlib import pyplot as plt

import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import dgl
import copy
import dgl.function as fn
from rdkit import Chem

from deepchem.feat import MolGraphConvFeaturizer, DMPNNFeaturizer

from model import GraphAttentionNet

FILENAME = "data.pickle"
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-4
device = torch.device("cuda")

train_smiles, test_smiles, train_y, test_y = pickle.load(open(FILENAME, "rb" ))

class MolGraphDataset(torch.utils.data.Dataset):
  '''
  Construct train, test Datasets from smiles
  '''
  def __init__(self, smiles, y):
    super().__init__()
    self.smiles = smiles
    self.y = y

    self.graphs = [self.construct_graphs(s) for s in self.smiles]

  def construct_graphs(self, smiles):
    # you can choose different featurizers according to the model https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html
    #features = MolGraphConvFeaturizer(use_edges=True).featurize([smiles])[0] # 30, 11
    features = DMPNNFeaturizer().featurize([smiles])[0] # 133, 14
    graph = features.to_dgl_graph()
    return graph

  def __len__(self):
    return len(self.smiles)

  def __getitem__(self, idx):
    graph = self.graphs[idx]
    y = self.y[idx]
    return graph, y

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

#[construct train/validation/test datasets using your own chosen methods]

train_val_dataset = MolGraphDataset(train_smiles, train_y)
test_dataset = MolGraphDataset(test_smiles, test_y)

split_ratio = 0.9
n_train = int(len(train_val_dataset)*split_ratio)

train_dataset, val_dataset = random_split(train_val_dataset, [n_train, len(train_val_dataset)-n_train])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, drop_last=False, num_workers=2, collate_fn=collate, pin_memory=True)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=False, num_workers=2, collate_fn=collate, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate)  

model = GraphAttentionNet(input_node_dim=133,
                          input_edge_dim=14,
                         num_layers=8,
                         d_model=512,
                         num_heads=8,
                         d_ff=1024,
                         dropout=0.1,
                         d_mlp=512,
                         n_output=1,
                          use_edges=True,
                          use_mol_attention=True,
                          mol_layers=3
                         )

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, lr_scheduler):    
  train_loss = []
  val_loss = []
  accumulation_steps = 1

  best_val_loss = 1e9
  early_stopping_counter = 0
  patience = 5

  for i in range(EPOCHS):
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0
    
    n_iter = 0
    model.train()
    
    for x, y in train_dataloader:
      n_iter += 1
      optimizer.zero_grad()
    
      x = x.to(device)
      y = y.to(device)
      y_pred = model(x)
            
      loss = loss_fn(y_pred, y.view(-1, 1))
      
      #loss = torch.sqrt(loss)
        
      if torch.isnan(loss).any():
        print(f'Iteration {n_iter} gives NaN')
        return
              
      loss.backward()
      optimizer.step()
      #if n_iter % accumulation_steps == 0: 
      #  optimizer.step() 
      #  optimizer.zero_grad()
        
      train_loss_epoch += loss

    lr_scheduler.step()
    train_loss_epoch /= n_iter
    train_loss.append(train_loss_epoch)
    print(f"Epoch {i}, train loss {train_loss_epoch:.4f}")
    
    # validation
    n_iter = 0
    model.eval()
    for x, y in val_dataloader:
      n_iter += 1
    
      x = x.to(device)
      y = y.to(device)
      with torch.no_grad():
          y_pred = model(x)
          loss = loss_fn(y_pred, y.view(-1, 1))
             
      val_loss_epoch += loss
    
    val_loss_epoch /= n_iter
    val_loss.append(val_loss_epoch)
    print(f"Epoch {i}, val loss {val_loss_epoch:.4f}")
    
    '''
    if best_val_loss == None:
        best_val_loss = val_loss_epoch
    elif val_loss_epoch >= best_val_loss:
        early_stopping_counter += 1
        if early_stopping_counter == patience:
            break
    else:
        best_val_loss = val_loss_epoch
        early_stopping_counter = 0
    '''
    if val_loss_epoch < best_val_loss:
      best_val_loss = val_loss_epoch
        
      checkpoint = {
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss_epoch,
      }
    
  print(f"Best model epoch {checkpoint['epoch']}")
  torch.save(checkpoint['model_state_dict'], f"best_ckpt_epoch{checkpoint['epoch']}.pt")
    
  return checkpoint, train_loss, val_loss
    
def eval_model(model, test_smiles, test_y):
  model.eval()

  test_dataset = MolGraphDataset(test_smiles, test_y)
  test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2, collate_fn=collate)
  
  test_loss = 0.0
  n_iter = 0
    
  with torch.no_grad():
      for x, y in test_dataloader:
        n_iter += 1
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x) 
        loss = loss_fn(y_pred, y.view(-1, 1))
        test_loss += loss
        
  print(f"Test MSE {test_loss/n_iter:.4f}， RMSE {math.sqrt(test_loss/n_iter):.4f}")

checkpoint, train_loss, val_loss = train(model, train_dataloader, val_dataloader, loss_fn, optimizer, lr_scheduler)

model.load_state_dict(checkpoint['model_state_dict'])

eval_model(model, test_smiles, test_y)

def plot_lc(train_loss, val_loss):
  plt.plot([i.detach().cpu().numpy() for i in train_loss], label="Train")
  plt.plot([i.detach().cpu().numpy() for i in val_loss], label="Validation")

  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('lc.jpg')
 
plot_lc(train_loss, val_loss)
