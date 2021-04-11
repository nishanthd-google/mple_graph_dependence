mkdir -p citeseer
for ((num = 100 ; num <= 109 ; num++)); do
	for ((i = 500 ; i <= 1400 ; i+=300)); do
		echo "citeseer/randomize-$num-$i.txt"
   		python train_pseudo.py --lam 1.0 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed $num --use_gibbs --train_size $i --num_gibbs_iters 20000 --randomize_train --val_size 500 > "citeseer/randomize-500-$num-$i.txt"
   		python train_pseudo.py --lam 1.0 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed $num --train_size $i --randomize_train --val_size 500 > "citeseer/randomize-500-$num-$i-no-gibbs.txt"
   		python train_no_graph.py --lam 1.0 --dataset citeseer --hidden 32 --lr 0.01 --patience 200 --seed $num --train_size $i --randomize_train --val_size 500 > "citeseer/randomize-500-$num-$i-no-graph.txt"
   		# python train_grand.py --lam 0.7 --tem 0.3 --order 2 --sample 2 --dataset citeseer --input_droprate 0.0 --hidden_droprate 0.2 --hidden 32 --lr 0.01 --patience 200 --seed $num --dropnode_rate 0.5  --cuda_device 4 > citeseer/"$num".txt
   	done
done
