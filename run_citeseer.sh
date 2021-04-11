#!/bin/bash

mkdir citeseer

for ((i = 200 ; i <= 1600 ; i+=200)); do
   python train_pseudo.py --lam 0.7 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda --use_gibbs --train_size $i > "citeseer/pmle_$i.txt"
   python train_pseudo.py --lam 0.7 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda --train_size $i > "citeseer/pmle_no_gibbs_$i.txt"
   python train_no_graph.py --lam 0.7 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda  --train_size $i > "citeseer/logistic_$i.txt"
done
