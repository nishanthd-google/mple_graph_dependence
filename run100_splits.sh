#!/bin/bash

dataset=$1
mkdir -p $dataset
for ((num = 100 ; num <= 107 ; num++)); do
  for ((i = 0 ; i < 10 ; i++)); do
	  echo "$dataset:seed-$num-split-$i"
	  python train_pseudo_splits.py --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num --split $i >"$dataset/splits-$i-$num-pmle-no-gibbs.txt"
	  python train_no_graph_splits.py --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num --split $i >"$dataset/splits-$i-$num-logistic.txt"
    # python train_pseudo.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed 100 --train_size $i > "$dataset/pmle_no_gibbs_$i.txt"
    # python train_no_graph.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed 100  --train_size $i > "$dataset/logistic_$i.txt"
  done
done

