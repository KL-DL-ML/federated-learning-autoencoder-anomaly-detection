# Federated Deep Learning for Anomaly Detection in Multivariate Time Series Data
Follow the below steps to setup and use the application.

## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install -r requirements.txt
```

## Dataset Preprocessing
Preprocess all datasets using the command
```bash
python3 preprocess.py SWaT ENERGY
```
Distribution rights to some datasets may not be available. Check the readme files in the `./data/` folder for more details. If you want to ignore a dataset, remove it from the above command to ensure that the preprocessing does not fail.

## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can be either of 'USAD', 'LSTM', and dataset can be one of 'SWaT', and 'ENERGY (Energy Consumption Data)'. To train with 20% data, use the following command 
```bash
python3 main.py --model <model> --dataset <dataset> --retrain --less
```
All outputs can be run multiple times to ensure statistical significance. 

## Next Steps
- Implement the Flower framework to make uses of the federated learning.