from unittest import result
from matplotlib.pyplot import plot
import torch
import pandas as pd
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


if __name__ == '__main__':
    early_stopping = EarlyStopping(patience=5, verbose=False)
    # Load Config
    config = get_best_config(args.model)
    # Load Data
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], config)
    print(model)
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
            early_stopping(lossT)
            if early_stopping.early_stop:
                print("Early stopping at epoch {}!!!".format(e))
                break
            # End Early Stopping
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    with torch.no_grad():
        print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        ### Plot curves
        if not args.test:
            plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

        # line_plot(f'{args.model}_{args.dataset}', testO, y_pred)
        ### Scores
        df = pd.DataFrame()
        lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
            result, pred = pot_eval(lt, l, ls)
            preds.append(pred)
            df = df.append(result, ignore_index=True)

        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        pprint(result)
    
        cf_matrix = [[result['TP'], result['FP']], [result['FN'], result['TN']]]
        plot_confusion_matrix(f'{args.model}_{args.dataset}', np.asarray(cf_matrix))
        
