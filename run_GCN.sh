#!/bin/bash

python train_GCN.py --lam 0.7 --dataset citeseer --hidden 64 --lr 0.01 --patience 200 --seed 100 python --layer 64 --alpha 0.2 --weight_decay 1e-4

