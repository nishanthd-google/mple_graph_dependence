import argparse
import os
import matplotlib.pyplot as plt


datasets = {'cora':{'pmle':[], 'pmle_no_gibbs':[], 'logistic':[], 'pmle_more_gibbs':[]}}
			# 'citeseer':{'pmle_gibbs':[], 'pmle_no_gibbs':[], 'logistic':[]}, 
			# 'pubmed':{'pmle_gibbs':[], 'pmle_no_gibbs':[], 'logistic':[]}
			# }

for dataset in datasets:
	for key in datasets[dataset]:
		for i in range(200, 1700, 200):
			with open(dataset + '/' + key + '_' + str(i) + '.txt', 'r') as reader:
				text = reader.readlines()
				test_accuracy = float(text[-1].split()[-1])
				datasets[dataset][key].append(test_accuracy)
		

for dataset in datasets:
	x = range(200, 1700, 200)
	# plt.plot(x, datasets[dataset]['pmle_gibbs'], label='pmle_gibbs')
	for key in datasets[dataset]:
		plt.plot(x, datasets[dataset][key], label=key)
	
	plt.legend()
	plt.show()
