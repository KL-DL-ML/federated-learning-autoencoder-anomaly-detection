import torch
import xlsxwriter
import pandas as pd
from torchinfo import summary
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
from sklearn.metrics import roc_curve, auc

datasets = ['ENERGY', 'SWaT']
models = ['AE', 'LSTM_AD', 'USAD']


def main(dataset):
    results = []
    fprs = []
    tprs = []
    for i, model_name in enumerate(models):
        ### Load data and prepare data
        config = get_best_config(model_name)
        train_loader, test_loader, labels = load_dataset(dataset)
        model, optimizer, scheduler, epoch, accuracy_list = load_model(model_name, labels.shape[1], config)
        trainD, testD = next(iter(train_loader)), next(iter(test_loader))
        trainO, testO = trainD, testD
        if model_name in ['USAD', 'AE']:
            trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
            
        ## Training phase
        print(f'{color.HEADER}Training {model_name} on {dataset}{color.ENDC}')
        num_epochs = config['num_epochs']
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        training_time = time() - start
        print(f'{color.HEADER}Training {model_name} on {dataset} finished in {training_time:.2f} seconds{color.ENDC}')
        
        # summary(model)

        ### Testing phase
        torch.zero_grad = True
        model.eval()
        with torch.no_grad():
            print(f'{color.HEADER}Testing {model_name} on {dataset}{color.ENDC}')
            df = pd.DataFrame()
            loss, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
            lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
            for i in range(loss.shape[1]):
                lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
                result, pred = pot_eval(lt, l, ls)
                preds.append(pred)
                df = df.append(result, ignore_index=True)
            lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
            labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
            result, pred = pot_eval(lossTfinal, lossFinal, labelsFinal)
            pprint(result)
            # Calcualte the ROC and AUC curve
            fpr, tpr, th = roc_curve(labelsFinal, pred)
            fprs.append(fpr)
            tprs.append(tpr)
            print(f'{color.HEADER}ROC AUC: {auc(fpr, tpr):.4f}{color.ENDC}')
            
        results.append({'model': model_name, 'dataset': dataset, 'result': result})
        model = None
    
    # row = 1
    # workbook = xlsxwriter.Workbook('results/output.xlsx')
    # worksheet = workbook.add_worksheet()
    # ### Add Columns
    # worksheet.write(0, 0, 'Model')
    # worksheet.write(0, 1, 'Dataset')
    # for i, key in enumerate(result):
    #     worksheet.write(0, i+2, key)
        
    # for result in results:
    #     worksheet.write(row, 0, result['model'])
    #     worksheet.write(row, 1, result['dataset'])
    #     for i, key in enumerate(result['result']):
    #         worksheet.write(row, i+2, float(result['result'][key]))
    #     row += 1
    # workbook.close()
        
    # Plot the ROC curve
    plot_roc_auc_curve(models, fprs, tprs)
    
    
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main('ENERGY/filtered')
    