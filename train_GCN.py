from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, load_data_split
from pygcn.models_gcn import GCNII
from sklearn.preprocessing import StandardScaler
# from pygcn.process import full_load_data


scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--split', type=int, default=0, help='Split Number')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')

#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data

splitstr = 'splits/'+args.dataset+'_split_0.6_0.2_'+str(args.split)+'.npz'
A, features, labels, idx_train, idx_val, idx_test = load_data_split(args.dataset, splitstr)
len_test = len(idx_test)
len_train = len(idx_train)
len_val = len(idx_val)
num_features = len(features[0])
num_labels = labels.max().item() + 1
# print(labels)
# print(idx_val)
# print(idx_test)
# print(features.shape)
# print(labels.shape)
# print('Labels', labels)
print('Test data size', len_test)
print('Train data size', len_train)
print('Val data size', len_val)
# A = A.to_dense()
# print('Adjancency', A.shape)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Model and optimizer

model_GCN = GCNII(nfeat=num_features,
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer = optim.Adam(model_GCN.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)

if args.cuda:
    model_GCN.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train_GCN(epoch):
    t = time.time()
    
    # print(features[idx_train].shape, interaction_train.unsqueeze(dim=1).shape)
    # X = torch.cat((features[idx_train], interaction_train), 1)
    X = features
    X[idx_test] = 0
    model_GCN.train()
    optimizer.zero_grad()
    output = model_GCN(X, A)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()

    optimizer.step()
    
    model_GCN.eval()
    output = torch.log_softmax(model_GCN(features, A), dim=-1) 
    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if epoch % 100 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(), acc_val.item()

def Train_GCN():
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

        l, a = train_GCN(epoch)
        loss_values.append(l)
        acc_values.append(a)

        # print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model_GCN.state_dict(), 'checkpoints/' + dataset + '_GCN.pkl')

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
    model_GCN.load_state_dict(torch.load('checkpoints/' + dataset +'_GCN.pkl'))

# use_gibbs = True

def test_GCN():
    model_GCN.eval()
    output = model_GCN(features, A)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


Train_GCN()
test_GCN()
