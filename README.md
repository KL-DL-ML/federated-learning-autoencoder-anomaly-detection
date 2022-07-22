# Federated Deep Learning for Anomaly Detection in Multivariate Time Series Data
Follow the below steps to setup and use the application.

## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install -r requirements.txt
```

## Dataset Preprocessing
Preprocess any datasets using the command
```bash
python3 preprocess.py ENERGY
```

To use S-G filter on which dataset, using the command
```bash
python3 preprocess.py ENERGY --filter
```
This will create another folder inside your dataset name called filtered.

## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset>
```
where `<model>` can be either of 'AE', 'LSTM_AE', and dataset can be one of 'SWaT', and 'ENERGY (Energy Consumption Data)'. 

To run a model on a filtered dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset> --filter
```
It will produce error, if you don't have any filtered datasets.

## Next Steps
- Implement the Flower framework to make uses of the federated learning.

# Federated Learning on Embedded Devices with Flower

This code will show you how Flower makes it very easy to run Federated Learning workloads on edge devices.
We'll try to implement federated learning in our anomaly detection methods on edge devices to capture any abnormal data from our sensor devices.

## Getting things ready

This is a list of components that you'll need: 

* For server: A machine running Linux/macOS.
* For clients: either a Rapsberry Pi 3 B+ (RPi 4 would work too) or a Jetson Xavier-NX (or any other recent NVIDIA-Jetson device).