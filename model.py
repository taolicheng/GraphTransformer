#!/usr/bin/env python

import torch
import gc
from GPUtil import showUtilization as gpu_usage

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def attention(g, q, k, v, e, d_k):
    '''
    update node representations according to self-attention, return updated node features
    '''
    with g.local_scope():
        g.ndata['Q'] = q
        g.ndata['K'] = k
        g.ndata['V'] = v
        
        g.apply_edges(lambda edges: {'m': torch.exp(edges.src['K'] * edges.dst['Q'] / math.sqrt(d_k))})
        
        g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'z'))
        
        if e is None:
          output_e = None
        if e is not None:
          g.edata['e'] = e
          # edge-weighted messages
          g.apply_edges(lambda edges:{'m': edges.data['m'] * edges.data['e']})
          output_e = g.edata['m']

        g.apply_edges(lambda edges: {'m': edges.data['m'] * edges.src['V'] / edges.dst['z']})

        g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'x'))
        output_x = g.ndata['x']
    
    return output_x, output_e 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, g, x):
                
        k = self.k_linear(x).view(-1, self.h, self.d_k) 
        q = self.q_linear(x).view(-1, self.h, self.d_k)
        v = self.v_linear(x).view(-1, self.h, self.d_k)
        
        output_x, _ = attention(g, q, k, v, None, self.d_k)
        
        concat = output_x.view(-1, self.d_model)
        
        output = self.out(concat)
    
        return output 

class EdgedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.e_linear = nn.Linear(d_model, d_model)
        
        self.out_x = nn.Linear(d_model, d_model)
        self.out_e = nn.Linear(d_model, d_model)
    
    def forward(self, g, x, e):
                
        k = self.k_linear(x).view(-1, self.h, self.d_k) 
        q = self.q_linear(x).view(-1, self.h, self.d_k)
        v = self.v_linear(x).view(-1, self.h, self.d_k)
        e = self.e_linear(e).view(-1, self.h, self.d_k)
        
        output_x, output_e  = attention(g, q, k, v, e, self.d_k)
        
        output_x = output_x.view(-1, self.d_model)
        output_x = self.out_x(output_x)
        
        output_e = output_e.view(-1, self.d_model)
        output_e = self.out_e(output_e)
    
        return output_x, output_e       
    
class Embedding(nn.Module):
  def __init__(self, input_dim, d_model):
    super().__init__()
    self.embed = nn.Linear(input_dim, d_model)

  def forward(self, x):
    return self.embed(x)

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()
    self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                            nn.ReLU(),
                            nn.Linear(d_ff, d_model))
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    return self.dropout(self.ff(x))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)    

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout, use_edges=False):
    super().__init__()
    
    self.use_edges = use_edges
    
    if use_edges:
        self.mha = EdgedMultiHeadAttention(d_model, num_heads)
    else:
        self.mha = MultiHeadAttention(d_model, num_heads)
    self.ff_x = FeedForward(d_model, d_ff, dropout)
    self.ff_e = FeedForward(d_model, d_ff, dropout)
    #self.dropout_1 = nn.Dropout(dropout)
    #self.dropout_2 = nn.Dropout(dropout)
    
    self.dropout_x = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
    self.dropout_e = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
    
    self.norm_x = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
    self.norm_e = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
    #self.norm_x = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(2)])
    #self.norm_e = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(2)]) 

  def forward(self, g, x, e):
    x0 = x
    if self.use_edges:
        e0 = e 
        
    if self.use_edges:
        x, e = self.mha(g, x, e)
    else:
        x = self.mha(g, x)
    
    x = x0 + self.dropout_x[0](x)
    x = self.norm_x[0](x)
    
    if self.use_edges:
        e = e0 + self.dropout_e[0](e)
        e = self.norm_e[0](e)

    x0 = x
    x = self.ff_x(x)
    x = x0 + self.dropout_x[1](x)
    x = self.norm_x[1](x)
    
    if self.use_edges:
        e0 = e
        e = self.ff_e(e)
        e = e0 + self.dropout_e[1](e)
        e = self.norm_e[1](e)
    
    return x, e

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, use_edges):
    super().__init__()
    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, use_edges=use_edges) for _ in range(num_layers)])

  def forward(self, g, x, e):
    for layer in self.encoder_layers:
      x, e = layer(g, x, e)
    return x

class GraphAttentionNet(nn.Module):
  def __init__(self, input_node_dim,
                     input_edge_dim,
                     num_layers=3,
                     d_model=256,
                     num_heads=8,
                     d_ff=256,
                     dropout=0.5,
                     d_mlp=256,
                     n_output=1,
                     use_edges=True,
                     use_mol_attention=False,
                     mol_layers=1
              ):
    super().__init__()
    
    self.use_edges = use_edges
    self.use_mol_attention = use_edges
    self.node_embedding = Embedding(input_node_dim, d_model)
    
    self.edge_embedding = Embedding(input_edge_dim, d_model)

    self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, use_edges=use_edges)

    if use_mol_attention:
        self.mol_attention = Encoder(mol_layers, d_model, num_heads, d_ff, dropout, use_edges=False)
        self.Norm = nn.LayerNorm(d_model)
    
    self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            Swish(),
            #nn.ReLU(),
            nn.Linear(d_mlp, d_mlp),
            Swish(),
            #nn.ReLU(),
            nn.Linear(d_mlp, n_output)
    )
    
  def rebatch(self, bg, bx):
    new_batch = []
    for g, x in zip(dgl.unbatch(bg), bx):
        g = dgl.add_nodes(g, 1, {'h': x.view(1,-1)})
        g = dgl.add_edges(g, g.nodes()[:-1], g.nodes()[-1])
        g = dgl.add_edges(g, g.nodes()[-1], g.nodes()[:-1])
        new_batch.append(g)
        
    return dgl.batch(new_batch)

  def readout(self, bg):
    readout_node = []
    for g in dgl.unbatch(bg):
        readout_node.append(g.ndata['h'][-1])
    #return torch.tensor(readout_node)
    return torch.stack(readout_node)

  def forward(self, g):
    
    h = self.node_embedding(g.ndata['x'])
    e = self.edge_embedding(g.edata['edge_attr'])

    x = self.encoder(g, h, e)
        
    g.ndata['h'] = x
    
    x = dgl.readout_nodes(g, 'h', op='sum')
    
    if self.use_mol_attention:
        g = self.rebatch(g, x)
        g.ndata['h'] = self.Norm(g.ndata['h'])
        x = self.mol_attention(g, g.ndata['h'], None)
        g.ndata['h'] = x
        #x = dgl.readout_nodes(g, 'h', op='max')
        x = self.readout(g)
        
    x = self.mlp(x)
    
    g.ndata.pop('h')
    return x

  def loss(self, yhat, y):
    loss = nn.MSELoss()(yhat, y)
    loss = torch.sqrt(loss)
    return loss

def explore_dataset(idx):
    print(train_val_dataset[idx][0])
    print(train_val_dataset[idx][1])
    
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-4
device = torch.device("cuda")

#split_ratio = 0.9
#n_train = int(len(train_val_dataset)*split_ratio)

#train_val_dataset = MolGraphDataset(train_smiles, train_logDs)

#test_dataset = MolGraphDataset(test_smiles, test_logDs)

#train_dataset, val_dataset = random_split(train_val_dataset, [n_train, len(train_val_dataset)-n_train])

#[construct train/validation/test datasets using your own chosen methods]

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, drop_last=False, num_workers=2, collate_fn=collate, pin_memory=True)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=False, num_workers=2, collate_fn=collate, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate)  

model = GraphAttentionNet(input_node_dim=30,
                          input_edge_dim=11,
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
    #if val_loss_epoch < best_val_loss:
      #best_val_loss = val_loss_epoch
  checkpoint = {
    'epoch': i,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss_epoch,
  }
    
  print(f"Best model epoch {checkpoint['epoch']}")
  torch.save(checkpoint['model_state_dict'], f"best_ckpt_epoch{checkpoint['epoch']}.pt")
    
  return checkpoint, train_loss, val_loss
    
def eval_model(model, test_smiles, test_logDs):
  model.eval()

  test_dataset = MolGraphDataset(test_smiles, test_logDs)
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

eval_model(model, test_smiles, test_logDs)

def plot_lc(train_loss, val_loss):
  plt.plot([i.detach().cpu().numpy() for i in train_loss], label="Train")
  plt.plot([i.detach().cpu().numpy() for i in val_loss], label="Validation")

  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('figs/Lipo_Pred_lc.jpg')
 
plot_lc(train_loss, val_loss)
