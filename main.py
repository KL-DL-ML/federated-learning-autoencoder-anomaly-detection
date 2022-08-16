import torch
import pandas as pd
import argparse

from tqdm import tqdm
from time import time
from codes.evaluations.eval_utils import *
from codes.plotter import *
from codes.constants import *
from pprint import pprint
# Import the model and data utility functions
from codes.model_utils import *
from codes.data_loader import *
from config import *
# End of imports

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='ENERGY',
                    help="dataset from ENERGY")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='AE',
                    help="model name")
parser.add_argument('--test',
                    action='store_true',
                    help="test the model")
parser.add_argument('--filter',
                    action='store_true',
                    help="train with filter dataset")
args = parser.parse_args()


def main():
    early_stopping = EarlyStopping(patience=5, verbose=False)
    # Load Config
    config = get_best_config(args.model)
    # Load Data
    train_loader, test_loader, labels = load_dataset(args.dataset, args.filter)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], config, args)
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['USAD', 'AE']:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = config['num_epochs']
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)            
            accuracy_list.append((lossT, lr))
            # Start Early Stopping
            # early_stopping(lossT)
            # if early_stopping.early_stop:
            #     print("Early stopping at epoch {}!!!".format(e))
            #     break
            # End Early Stopping
        training_time = time() - start
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list, args)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
        plot_losses(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    with torch.no_grad():
        print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
        loss, mae, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        ### Plot curves
        if not args.test:
            plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
        plot_actual_predicted(f'{args.model}_{args.dataset}', testO, y_pred)
        ### Scores
        lossT, _, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
            result, _ = pot_eval(lt, l, ls)
            anomalies(f'{args.model}_{args.dataset}', f'Dim {i}', l, result['threshold'])
            
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        pprint(result)
        
        print("MAE: ", np.mean(mae))
        print("MSE: ", np.mean(loss))
        # print('Traning Time: ', training_time)
        
        anomaly(f'{args.model}_{args.dataset}', loss, result['threshold'])
    
        cf_matrix = [[result['TP'], result['FP']], [result['FN'], result['TN']]]
        plot_confusion_matrix_fl(f'{args.model}_{args.dataset}', 'server', np.asarray(cf_matrix))

if __name__ == '__main__':
    main()        
