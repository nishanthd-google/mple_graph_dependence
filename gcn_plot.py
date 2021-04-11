import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt

dataset = sys.argv[1]

res = {}
res_no = {}
res_no_gibbs = {}

x_value = range(10)
y_min = []
y_max = []
y_mean = []
y_std = []

for num in x_value:
    res[num] = []
    for i in range(100, 106, 1):
        ofile = 'gcn-' + str(num) + '-' + str(i) + '.txt'
        with open(os.path.join(dataset, ofile), 'r') as f:
            lines = f.readlines()
            # print(lines)
            if len(lines) == 0:
                continue
            line = lines[-1]
            line = line.strip().split()[-1]
            res[num].append(float(line))
    
    y_mean.append(np.mean(res[num]))
    y_min.append(np.min(res[num]))
    y_max.append(np.max(res[num]))
    y_std.append(np.std(res[num]))      

# print(res_no)
print('GCN', np.mean(y_mean))
# print('Logistic', y_mean_no, y_std_no)
# print('No Gibbs', y_mean_no_gibbs, y_std_no_gibbs)
plt.fill_between(x_value, y_max, y_min, alpha=.5)
plt.plot(x_value,y_mean, label='GCN')

# plt.fill_between(x_value, y_max_no, y_min_no, alpha=.5)
# plt.plot(x_value,y_mean_no, label='logistic')

# plt.fill_between(x_value, y_max_no_gibbs, y_min_no_gibbs, alpha=.5)
# plt.plot(x_value,y_mean_no, label='no gibbs')

# plt.legend()
# plt.savefig(dataset+'.jpg')
plt.show()

