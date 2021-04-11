mkdir -p pubmed
for ((num = 100 ; num <= 107 ; num++)); do
	for ((i = 500 ; i <= 1400 ; i+=300)); do
		echo "pubmed/randomize-$num-$i.txt"
   		python train_pseudo.py --lam 1.0 --dataset pubmed --hidden 32 --lr 0.01 --patience 200 --seed $num --use_gibbs --train_size $i --num_gibbs_iters 20000 --randomize_train > "pubmed/randomize=$num-$i.txt"
   		python train_no_graph.py --lam 1.0 --dataset pubmed --hidden 32 --lr 0.01 --patience 200 --seed $num --train_size $i --randomize_train > "pubmed/randomize=$num-$i-no-graph.txt"
   	done
done