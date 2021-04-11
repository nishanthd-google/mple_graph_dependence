from __future__ import division
from __future__ import print_function

import time
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import MLP
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--train_size', type=int, default=500, help='Training Set Size')
parser.add_argument('--val_size', type=int, default=100, help='Validation Set Size')
parser.add_argument('--randomize_train', action='store_true', default=False, help='Randomize Training Data')
parser.add_argument('--small_splits', action='store_true', default=False, help='Use small class-sensitive splits')
parser.add_argument('--one_class', action='store_true', default=False, help='Binary one vs all')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Cuda:", args.cuda)
# torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 
print("Device is", device) 

def convert_labels_to_onehot(labels):
  reshaped_labels = torch.reshape(labels, (-1, 1))
  labels_onehot = torch.zeros((len(labels), labels.max().item() + 1), device=device)
  labels_onehot.zero_()
  labels_onehot.scatter_(1, reshaped_labels, 1)
  return labels_onehot

# Load data
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = torch.arange(idx_train.shape[0], labels.shape[0]-1, dtype=int)
if args.cuda:
  labels = labels.cuda()
if args.one_class:
  labels = (labels > 0).int()
num_labels = labels.max().item() + 1
labels_onehot = convert_labels_to_onehot(labels)
idx_labels = {}
for i in range(labels_onehot.shape[1]):
  idx_labels[i] = torch.nonzero(labels_onehot[:,i])
len_train = args.train_size
len_val = args.val_size
#len_train = len(idx_train)
#len_val = len(idx_val)
len_test = len(idx_test)
# idx_val = torch.arange(len_train, len_train + len_val, dtype=int)
#idx_val = torch.arange(idx_test[0] - len_val, idx_test[0], dtype=int)
#idx_train = torch.arange(0, len_train, dtype=int)

if args.randomize_train:
    perm = torch.randperm(idx_test[0] - len_val)
    idx_train = perm[:len_train]

if args.small_splits:
  idx_train = torch.zeros(len_train, dtype=torch.long)
  size_per_label = len_train // (num_labels)
  for i in range(num_labels):
    for j in range(size_per_label):
      idx_train[i*size_per_label + j] = idx_labels[i][j]
    


# print('Test data size', idx_test)
# print('Train data size', idx_train)
# print('Val data size', idx_val)

# print('Adjancency', A.to_dense()[idx_train][idx_train])

# Model and optimizer
model = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            use_bn = args.use_bn)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()

def train(epoch):
    t = time.time()
    
    X = features
    
    model.train()
    optimizer.zero_grad()


    output = torch.log_softmax(model(X), dim=-1)

    
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(X)
        output = torch.log_softmax(output, dim=-1)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(), acc_val.item()

def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    # create a checkpoints directory if it does not exist
    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints')
    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        # print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), 'checkpoints/' + dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('checkpoints/' + dataset + '.pkl'))



def test():
    model.eval()
    X = features
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
Train()
test()
