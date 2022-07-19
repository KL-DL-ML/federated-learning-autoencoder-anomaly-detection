import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from codes.constants import *


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    pass
    # folder = f'checkpoints/{args.model}_{args.dataset}/'
    # os.makedirs(folder, exist_ok=True)
    # file_path = f'{folder}/model.ckpt'
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict(),
    #     'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims, config):
    import codes.models
    model_class = getattr(codes.models, modelname)
    model = model_class(dims, config).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    # fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    fname = ''
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1
        l1s, l2s = [], []
        if training:
            for d in data:
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch + 1},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
        
    elif 'LSTM_AE' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1
        l1s = []
        if training:
            for _, d in enumerate(data):
                x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch + 1},\tMSE = {np.mean(l1s)}')
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
    elif 'AE' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1
        l1s = []
        if training:
            for _, d in enumerate(data):
                x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch + 1},\tMSE = {np.mean(l1s)}')
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
        
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()
        


  
##################
# Early Stopping #
##################
# source : https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta= -0.00001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # if the current score does not exceed the best scroe, run the codes following below
        else:  
            self.counter = 0
            self.best_score = score
            self.val_loss_min = val_loss