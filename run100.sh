#!bin/bash

dataset=$1
mkdir -p $dataset
for ((num = 100 ; num <= 107 ; num++)); do
	for ((i = 500 ; i <= 1400 ; i+=300)); do
		echo "$dataset/randomize-$num-$i.txt"
   		python train_pseudo.py --lam 1.0 --dataset $datset --hidden 32 --lr 0.01 --patience 200 --seed $num --use_gibbs --train_size $i --num_gibbs_iters 20000 --randomize_train > "$dataset/randomize=$num-$i.txt"
   		python train_no_graph.py --lam 1.0 --dataset $dataset --hidden 32 --lr 0.01 --patience 200 --seed $num --train_size $i --randomize_train > "$dataset/randomize=$num-$i-no-graph.txt"
   	done
done
