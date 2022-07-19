import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from codes.parser import *


def load_dataset(dataset):
    if args.filter:
        folder = os.path.join(f'data/processed/{dataset}/filtered')
    else:
        folder = os.path.join('data/processed', dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if args.less:
        loader[0] = cut_array(0.2, loader[0])
    _train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    _test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    _labels = loader[2]
    return _train_loader, _test_loader, _labels


def cut_array(percentage, arr):
    print(f'Slicing dataset to {int(percentage * 100)}%')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]



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

