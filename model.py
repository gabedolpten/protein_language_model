import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.nn as nn 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import copy
import time
from torch.utils.data import Dataset, DataLoader
from dataloaders import *

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, seqlength=64):
        super().__init__()
        self.model_type = 'Transformer'
        print('d_model:', d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len = seqlength)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, 
                                          src_key_padding_mask=src_mask
                                         )
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Make a positional encoding
    """    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(torch.outer(position, div_term))
        pe[0, :, 1::2] = torch.cos(torch.outer(position, div_term))
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        pe_vec = self.pe[:x.size(0)]
        x = x + pe_vec
        return self.dropout(x)



from torch.utils.data import random_split
def make_train_test(masking_function, forkhead_dict, n_tokens, le, conservation_key, padding_index, masking_index, 
                    mut_prob, device, beta = 1, n_crop_per_protein=400, train_val_split = [.1, .9], **kwargs):
    """
    Make the training/test data sets
    """    
    random_mut_dataset = ProtSeqDataset(forkhead_dict,  n_tokens, le, conservation_key, padding_index, masking_index, 
                                        masking_function = masking_function, mut_prob = mut_prob, device=device,
                                        n_crop_per_protein = n_crop_per_protein, beta=beta,
                                        )
    (train_random_mut_dataset, test_random_mut_dataset) = random_split(random_mut_dataset, train_val_split, 
                                                                       generator=torch.Generator().manual_seed(42)
                                                                    )
    print(len(train_random_mut_dataset))
    train_dataloader = DataLoader(train_random_mut_dataset, batch_size=32, shuffle=True, num_workers=0,)
    val_dataloader = DataLoader(test_random_mut_dataset, batch_size=32, shuffle=True, num_workers=0,)
    return train_dataloader, val_dataloader

def make_model(d_hid, nlayers, nhead, dropout, emsize, device, n_tokens):
    """
    Make the model, optimizer, scheduler
    """
    model = TransformerModel(n_tokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # lr = 1e-4 # learning rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr = .5  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # lr = .5  # learning rate
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1)

    return model, optimizer, scheduler




def train_and_validate(model, train_dataloader, val_dataloader_dict, optimizer, scheduler, criterion, train_losses, val_loss_dict, epochs = 10):
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_dataloader, criterion, optimizer, scheduler, epoch)
        train_losses += train_loss

        for cond, val_dataloader in val_dataloader_dict.items():
            val_loss_dict.setdefault(cond, [])
            val_loss = evaluate(model, val_dataloader, criterion)
            val_loss_dict[cond].append(val_loss)
            
        if epoch % (epochs//2) == 0:
            print("Done with epoch", epoch)
        scheduler.step()    


import time
def train(model: nn.Module, dataloader, criterion, optimizer, scheduler, epoch) -> None:
    """
    Run one epoch
    """
    model.train()  # turn on train mode
    total_loss = 0.
    cur_loss = 0.
    if len(dataloader) > 30:
        log_interval = len(dataloader) // 30
    else:
        log_interval = min(len(dataloader)//2, 4)
    start_time = time.time()
    losses = []
    for batch, data in enumerate(dataloader):
        inputdata, target, masked = data['input'], data['output'], data['mask']
        output = model(inputdata, masked)
        loss = criterion(torch.swapaxes(output, 1, 2), torch.swapaxes(target, 1, 2))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        losses.append(loss.item())
        cur_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d} batches | '
                  f'lr {lr:02.2f} |'
                  f'loss {cur_loss:5.2f}')
            
            cur_loss = 0
            start_time = time.time()
    return losses

def evaluate(model: nn.Module, eval_data: DataLoader, criterion) -> float:
    """
    Evaluate a validation data set
    """
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for batch, data in enumerate(eval_data):
            inputdata, target, masked = data['input'], data['output'], data['mask']
            output = model(inputdata, masked)
            loss = criterion(torch.swapaxes(output, 1, 2), torch.swapaxes(target, 1, 2))
            total_loss += loss.item()
    return total_loss / (len(eval_data))
