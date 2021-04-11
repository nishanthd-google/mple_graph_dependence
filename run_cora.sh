#!/bin/bash

mkdir -p cora

for ((i = 0 ; i <= 10 ; i++)); do
	echo "$i"
	python train_pseudo_splits.py --dataset cora --hidden 32 --lr 0.01 --patience 200 --seed $i --no-cuda --split $i >"cora/splits-$i-pmle-no-gibbs.txt"
	python train_no_graph_splits.py --dataset cora --hidden 32 --lr 0.01 --patience 200 --seed $i --no-cuda --split $i >"cora/splits-$i-logistic.txt"
   # python train_pseudo.py --lam 1.0 --dataset cora --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda --train_size $i > "cora/pmle_no_gibbs_$i.txt"
   # python train_no_graph.py --lam 1.0 --dataset cora --hidden 32 --lr 0.01 --patience 200 --seed 100 --no-cuda  --train_size $i > "cora/logistic_$i.txt"
done


