#!bin/bash

dataset=$1
mkdir -p $dataset
for ((num = 100 ; num <= 109 ; num++)); do
		echo "$dataset/public-$num.txt"
   		python train_pseudo.py --lam 1.0 --dataset $datset --hidden 32 --lr 0.01 --patience 200 --seed $num --num_gibbs_iters 20000 > "$dataset/public=$num.txt"
   		python train_no_graph.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num > "$dataset/public=$num-no-graph.txt"
done
