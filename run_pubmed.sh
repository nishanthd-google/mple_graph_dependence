#!/bin/bash

mkdir pubmed

for ((i = 200 ; i <= 1600 ; i+=200)); do
   python train_pseudo.py --lam 1.0 --dataset pubmed --hidden 32 --lr 0.2 --patience 200 --seed 100 --no-cuda --use_gibbs --train_size $i > "pubmed/500_pmle_$i.txt"
   python train_pseudo.py --lam 1.0 --dataset pubmed --hidden 32 --lr 0.2 --patience 200 --seed 100 --no-cuda --train_size $i > "pubmed/500_pmle_no_gibbs_$i.txt"
   python train_no_graph.py --lam 1.0 --dataset pubmed --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda  --train_size $i > "pubmed/500_logistic_$i.txt"
done