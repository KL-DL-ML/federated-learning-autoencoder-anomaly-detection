import torch
import torch.nn as nn
import pandas as pd
import argparse

from tqdm import tqdm
from time import time
from codes.evaluations.eval_utils import *
from codes.plotter import *
from codes.constants import *
from pprint import pprint

from torch.utils.data import DataLoader
# Import the model and data utility functions
from config import *
# End of imports


parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='ENERGY',
                    help="dataset from ENERGY")
parser.add_argument('--filter',
                    default=False,
                    action='store_true',
                    help="train with filter dataset")
args = parser.parse_args()


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


def load_dataset(dataset, filter=True):
    if filter:
        folder = os.path.join(f'data/processed/{dataset}/filtered')
    else:
        folder = os.path.join('data/processed', dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    _train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    _test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    _labels = loader[2]
    return _train_loader, _test_loader, _labels

def load_model(modelname, dims, config):
    try:
        import codes.models
        model_class = getattr(codes.models, modelname)
        model = model_class(dims, config).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
        return model, optimizer, scheduler, epoch, accuracy_list
    except:
        epoch = -1
        accuracy_list = []
        return model, optimizer, scheduler, epoch, accuracy_list
    
    
def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    feats = dataO.shape[1]
    l = nn.MSELoss(reduction = 'none')
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
    
    
def main():
    preds = []
    config = {
        "num_epochs": 15,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 10,
    }
    print(args)
    # Load Data
    train_loader, test_loader, labels = load_dataset(args.dataset, args.filter)
    model, optimizer, scheduler, epoch, accuracy_list = load_model('AE', labels.shape[1], config)
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['AE']:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    print(f'{color.HEADER}Training AE on {args.dataset}{color.ENDC}')
    num_epochs = config['num_epochs']
    e = epoch + 1
    start = time()
    for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
    
    ### Testing phase
    torch.zero_grad = True
    model.eval()
    with torch.no_grad():
        print(f'{color.HEADER}Testing AE on {args.dataset}{color.ENDC}')
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        ### Scores
        df = pd.DataFrame()
        lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
        # for i in range(loss.shape[1]):
        #     lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        #     result, pred = pot_eval(lt, l, ls)
        #     preds.append(pred)
        #     df = df.append(result, ignore_index=True)

        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        pprint(result)

if __name__ == '__main__':
    main()        