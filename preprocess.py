import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as sg
from sklearn.preprocessing import MinMaxScaler

datasets = ['ENERGY', 'SWaT']

def filtering(dataset):
    try:
        for column in dataset.columns:
            try:
                dataset[column] = sg(dataset[column], window_length=9, polyorder=5)
            except Exception as e:
                print(e)
        return dataset
    except:
        print('Error in filtering dataframe.')

def normalize(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def energy_dataset(dataset, ls):
    scaler = MinMaxScaler()
    
    split_rate = float(args.trainsize / 100)
    train = dataset.loc[:dataset.shape[0] * split_rate - 1]
    test = dataset.loc[dataset.shape[0] * split_rate:]
    train, test = train.values[0:, 1:].astype(float), test.values[0:, 1:].astype(float)
    
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    ls = ls.values[:, 0].astype(int)
    labels = np.zeros_like(test)
    for i in range(-200, 200):
        labels[ls + i, :] = 1
    print(train.shape, test.shape, labels.shape)
    return train, test, labels

def swat_dataset(df_train, df_test, labels):
    train, min_a, max_a = normalize(df_train.values)
    test, _, _ = normalize(df_test.values, min_a, max_a)
    print(train.shape, test.shape, labels.shape)
    return train, test, labels


def load_data(dataset):
    folder = os.path.join('data/processed', dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'ENERGY':
        dataset_folder = 'data/raw/ENERGY'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        dataset = pd.read_csv(os.path.join(dataset_folder, 'energy_consumption_hourly.csv'))
        dataset = dataset[:3000]
        # Check if dataset needed to be filtered
        if args.filter:
            dataset = filtering(dataset)
            train, test, labels = energy_dataset(dataset, ls)
            os.makedirs(folder + '/filtered', exist_ok=True)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder + '/filtered', f'{file}.npy'), eval(file))
        #
        else:
            train, test, labels = energy_dataset(dataset, ls)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{file}.npy'), eval(file))
                
    # elif dataset == 'SKAB':
    #     split_rate = 0.60
    #     dataset_folder = 'data/raw/SKAB'
    #     # df = pd.read_csv(os.path.join(dataset_folder, 'alldata_skab.csv'))[20000:25000]
    #     df = pd.read_csv(os.path.join(dataset_folder, 'alldata_skab.csv'))[['datetime', 'anomaly', 'changepoint', 'Volume Flow RateRMS']][10000:20000]
    #     train, test = train_test_split(df, train_size=split_rate, random_state=42)
    
    #     lab = test['anomaly']
    #     train = train.drop(['datetime', 'anomaly', 'changepoint'], axis=1)
    #     test = test.drop(['datetime', 'anomaly', 'changepoint'], axis=1)
        
    #     train, min_a, max_a = normalize2(train)
    #     test, _, _ = normalize2(test, min_a, max_a)
        
    #     labels = np.zeros_like(test).astype(int)
    #     for i, data in enumerate(lab):
    #             labels[i, :] = data
    #     print(train.shape, test.shape, labels.shape)
    #     for file in ['train', 'test', 'labels']:
    #         np.save(os.path.join(folder, f'{file}.npy'), eval(file))
        

    elif dataset == 'SWaT':
        dataset_folder = 'data/raw/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        train = pd.read_json(file, lines=True)[['val']][3000:6000]
        test = pd.read_json(file, lines=True)[['val']][7000:12000]
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        # 
        if args.filter:
            train = filtering(train)
            test = filtering(test)
            train, test, labels = swat_dataset(train, test, labels)
            os.makedirs(folder + '/filtered', exist_ok=True)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder + "/filtered", f'{file}.npy'), eval(file))
        #
        else:
            train, test, labels = swat_dataset(train, test, labels)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{file}.npy'), eval(file))


    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='ENERGY',
                    help="dataset from ENERGY")
parser.add_argument('--trainsize',
                    metavar='-d',
                    type=int,
                    required=False,
                    default=80,
                    help="percentage of train size of the dataset.")
parser.add_argument('--filter',
                    action='store_true',
                    help="filter the dataset.")
args = parser.parse_args()

if __name__ == '__main__':
    print("Usage: python preprocess.py {}".format(args.dataset))
    load_data(args.dataset)