#!/bin/bash

python3 preprocess.py --dataset ENERGY --trainsize 75
# sleep 1
python3 preprocess.py --dataset ENERGY --filter --trainsize 75
# sleep 1
python3 eval_noise_removal.py
sleep 2
python3 eval_noise_removal.py --filter