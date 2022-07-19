import os
from re import A
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from AE import AE 


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, _ in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w.view(-1))
    return torch.stack(windows)  

def load_model(modelname, dims, config):
    model = AE(dims, config).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1
    accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def load_dataset(dataset):
    folder = os.path.join(f'data/processed/{dataset}/filtered')
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    _train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    _test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    _labels = loader[2]
    return _train_loader, _test_loader, _labels

def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction = 'none')
    n = epoch + 1
    l1s = []
    feats = dataO.shape[1]
    if training:
        for _, d in enumerate(data):
            x = model(d)
            loss = torch.mean(l(x, d))
            l1s.append(torch.mean(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        xs = []
        for d in data: 
            x = model(d)
            xs.append(x)
        xs = torch.stack(xs)
        y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
        loss = l(xs, data)
        loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
        return loss.detach().numpy(), y_pred.detach().numpy()
        
