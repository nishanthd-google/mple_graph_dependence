import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt

dataset = sys.argv[1]

res = {}
res_no = {}

train_sizes = {}
train_sizes['cora'] = [140, 420, 840, 1120]
train_sizes['citeseer'] = [120, 480, 840, 1200, 1560, 1920]
train_sizes['pubmed'] = [60, 300, 1200, 2400, 3900, 6000]

y_mean = []
y_std = []
y_min = []
y_max = []

y_min_no = []
y_max_no = []
y_mean_no = []
y_std_no = []

for num in train_sizes[dataset]:
    res[num] = []
    res_no[num] = []
    for i in range(100, 110, 1):
        ofile = 'small=' + str(num) + '-' + str(i) + '.txt'
        with open(os.path.join(dataset, ofile), 'r') as f:
            line = f.readlines()[-1]
            line = line.strip().split()[-1]
            res[num].append(float(line))
        ofile = 'small=' + str(num) + '-' + str(i) + '-no-graph.txt'
        with open(os.path.join(dataset, ofile), 'r') as f:
            line = f.readlines()[-1]
            line = line.strip().split()[-1]
            res_no[num].append(float(line))
        # ofile = 'randomize-500-' + str(i) + '-' + str(num) + '-no-gibbs.txt'
        # with open(os.path.join(dataset, ofile), 'r') as f:
        #     line = f.readlines()[-1]
        #     line = line.strip().split()[-1]
        #     res_no_gibbs[num].append(float(line))
    
    y_mean.append(np.mean(res[num]))
    y_std.append(np.std(res[num]))
    y_min.append(np.min(res[num]))
    y_max.append(np.max(res[num]))

    y_mean_no.append(np.mean(res_no[num]))
    y_std_no.append(np.std(res_no[num]))
    y_min_no.append(np.min(res_no[num]))
    y_max_no.append(np.max(res_no[num]))

print('MPLE', y_mean, y_std)
print('Logistic', y_mean_no, y_std_no)
#print('No Gibbs', y_mean_no_gibbs, y_std_no_gibbs)
x_value = train_sizes[dataset]
plt.fill_between(x_value, y_max, y_min, alpha=.5)
plt.plot(x_value, y_mean, label=r'MPLE-$\beta$')

plt.fill_between(x_value, y_max_no, y_min_no, alpha=.5)
plt.plot(x_value, y_mean_no, label=r'MPLE-$0$')

# plt.fill_between(x_value, y_max_no_gibbs, y_min_no_gibbs, alpha=.5)
# plt.plot(x_value,y_mean_no, label='no gibbs')
plt.xlabel('Train data size')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig(dataset+'-small.jpg')
plt.show()

