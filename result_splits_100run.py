import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt

dataset = sys.argv[1]

res = {}
res_no = {}
res_no_gibbs = {}

splits = range(10)
y_min = []
y_max = []
y_mean = []
y_std = []

y_min_no = []
y_max_no = []
y_mean_no = []
y_std_no = []


y_min_no_gibbs = []
y_max_no_gibbs = []
y_mean_no_gibbs = []
y_std_no_gibbs = []

for split in splits:
    res[split] = []
    res_no[split] = []
    res_no_gibbs[split] = []
    for i in range(100, 107, 1):
        ofile = 'splits-' + str(split) + '-' + str(i) + '-pmle-no-gibbs.txt'
        with open(os.path.join(dataset, ofile), 'r') as f:
            line = f.readlines()[-1]
            line = line.strip().split()[-1]
            res_no_gibbs[split].append(float(line))
        ofile = 'splits-' + str(split) + '-' + str(i) + '-logistic.txt'
        with open(os.path.join(dataset, ofile), 'r') as f:
            line = f.readlines()[-1]
            line = line.strip().split()[-1]
            res_no[split].append(float(line))
    
    # y_mean.append(np.mean(res[num]))
    # y_min.append(np.min(res[num]))
    # y_max.append(np.max(res[num]))
    # y_std.append(np.std(res[num]))      
    y_mean_no.append(np.mean(res_no[num]))
    y_min_no.append(np.min(res_no[num]))
    y_max_no.append(np.max(res_no[num]))
    y_std_no.append(np.std(res_no[num]))
    y_mean_no_gibbs.append(np.mean(res_no_gibbs[num]))
    y_min_no_gibbs.append(np.min(res_no_gibbs[num]))
    y_max_no_gibbs.append(np.max(res_no_gibbs[num]))
    y_std_no_gibbs.append(np.std(res_no_gibbs[num]))

# print(res_no)
# print('Gibbs', y_mean)
print('Logistic', y_mean_no, y_std_no)
print('No Gibbs', y_mean_no_gibbs, y_std_no_gibbs)
# plt.fill_between(x_value, y_max, y_min, alpha=.5)
# plt.plot(x_value,y_mean, label='pmle')

plt.fill_between(splits, y_max_no, y_min_no, alpha=.5)
plt.plot(splits, y_mean_no, label='logistic')

plt.fill_between(splits, y_max_no_gibbs, y_min_no_gibbs, alpha=.5)
plt.plot(splits, y_mean_no, label='no gibbs')

plt.legend()
plt.savefig(dataset+'-splits-pmle-no-gibbs-vs-logistic.jpg')
plt.show()


