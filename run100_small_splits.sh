#!/bin/bash

dataset='cora'
cora_sizes=(140 420 840 1120)
citeseer_sizes=(120 480 840 1200 1560)
pubmed_sizes=(60 300 1200 2400 3900 6000)

for train_size in ${cora_sizes[@]}; do
  for ((num = 100 ; num <= 109 ; num++)); do
    echo "$dataset/small-$train_size-$num.txt"
   	python train_pseudo.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num --num_gibbs_iters 20000 --small_splits --train_size $train_size > "$dataset/small=$train_size-$num.txt"
   	python train_no_graph.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num --small_splits --train_size $train_size > "$dataset/small=$train_size-$num-no-graph.txt"
  done
done