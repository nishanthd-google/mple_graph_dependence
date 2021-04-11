from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import MLP, PMLE
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
parser.add_argument('--num_gibbs_iters', type=int, default=10000, help='Numner of iterations for the Gibbs Sampler')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--use_gibbs', action='store_true', default=False, help='Use Gibbs Sampler')
parser.add_argument('--train_size', type=int, default=500, help='Training Set Size')
parser.add_argument('--val_size', type=int, default=100, help='Validation Set Size')
parser.add_argument('--randomize_train', action='store_true', default=False, help='Randomize Training Data')
parser.add_argument('--small_splits', action='store_true', default=False, help='Use small class-sensitive splits')
parser.add_argument('--one_class', action='store_true', default=False, help='Binary one vs all')
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

# idx_val = torch.arange(len_train, len_train + len_val, dtype=int)

# print(features.shape)
# print(labels.shape)
# print('Labels', labels)
# print('Test data size', idx_test)
# print('Train data size', idx_train)
# print('Val data size', idx_val)
A = A.to_dense()
# print('Adjancency', A.to_dense()[idx_train][idx_train]) 

def compute_interaction_term_train():
    interaction = torch.zeros((len(idx_train), labels.max().item() + 1), device=device)
    label_tensor = torch.zeros((len(idx_train), labels.max().item() + 1), device=device)
    label_tensor.zero_()
    label_tensor.scatter_(1, torch.reshape(labels[idx_train], (-1,1)), 1)
    interaction = torch.matmul(A[idx_train][:, idx_train], label_tensor)
    return interaction

def compute_interaction_term_val():
    interaction = torch.zeros((len(idx_val), labels.max().item() + 1), device=device)
    label_tensor = torch.zeros((len(idx_train)+len(idx_val), labels.max().item() + 1), device=device)
    label_tensor.zero_()
    label_tensor.scatter_(1, torch.reshape(labels[torch.cat((idx_train, idx_val))], (-1,1)), 1)
    interaction = torch.matmul(A[idx_val][:, torch.cat((idx_train, idx_val))], label_tensor)
    return interaction

def compute_interaction_term_test():
    interaction = torch.zeros((len(idx_test), labels.max().item() + 1))
    label_tensor = torch.zeros((len(idx_train)+len(idx_val), labels.max().item() + 1), device=device)
    label_tensor.zero_()
    label_tensor.scatter_(1, torch.reshape(labels[torch.cat((idx_train, idx_val))], (-1,1)), 1)
    interaction = torch.matmul(A[idx_test][:, torch.cat((idx_train, idx_val))], label_tensor)
    return interaction

def compute_no_val_interaction_term_test():
    interaction = torch.zeros((len(idx_test), labels.max().item() + 1))
    label_tensor = torch.zeros((len(idx_train), labels.max().item() + 1), device=device)
    label_tensor.zero_()
    label_tensor.scatter_(1, torch.reshape(labels[idx_train], (-1,1)), 1)
    interaction = torch.matmul(A[idx_test][:, idx_train], label_tensor)
    return interaction


# print(interaction_train)

# print(interaction_train.shape)
# Model and optimizer
model = PMLE(nfeat=features.shape[1],
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

# make A zero-diagonal
A = A - torch.diag_embed(torch.diagonal(A))
interaction_train = compute_interaction_term_train()
interaction_val = compute_interaction_term_val()
interaction_test = compute_interaction_term_test()

if args.cuda:
    interaction_train = interaction_train.cuda()
    interaction_val = interaction_val.cuda()
    interaction_test = interaction_test.cuda()

def train(epoch):
    t = time.time()
    
    # print(features[idx_train].shape, interaction_train.unsqueeze(dim=1).shape)
    # X = torch.cat((features[idx_train], interaction_train), 1)
    X = features[idx_train]
    model.train()
    optimizer.zero_grad()


    output = torch.log_softmax(model(X, interaction_train), dim=-1)

    
    loss_train = F.nll_loss(output, labels[idx_train])

    acc_train = accuracy(output, labels[idx_train])

    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        model.eval()
        output_val = model(X, interaction_train)
        output = torch.log_softmax(output, dim=-1)
    
    output_val = torch.log_softmax(model(features[idx_val], interaction_val), dim=-1) 
    loss_val = F.nll_loss(output_val, labels[idx_val]) 
    acc_val = accuracy(output_val, labels[idx_val])

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
    model.load_state_dict(torch.load('checkpoints/' + dataset +'.pkl'))


# use_gibbs = True

num_iter = args.num_gibbs_iters

def test():
    print(model.beta.weight.data)
## Gibbs sampler
  # A is whole adjacency graph now
    num_classes = labels.max().item() + 1
    model.eval()
    X = features[idx_test]
    output = model(X, interaction_test)
    # output_probs = torch.log_softmax(output, dim=-1)
    # test_labels = output_probs.max(1)[1].type_as(labels)
    if args.use_gibbs:
        # init randomly
        test_labels = torch.LongTensor(len_test,1).random_(0, num_classes)
        test_labels_onehot = torch.FloatTensor(len_test, num_classes)
        test_labels_onehot.zero_()
        test_labels_onehot.scatter_(1, test_labels, 1)
        test_labels_onehot = test_labels_onehot.cuda()
        for iter in range(num_iter):
          # choose a random node
          i = np.random.randint(0, len_test)
          # re-sample from conditional distribution
          # 1. calculate logits for each class
          logits = model.beta.weight.data * torch.matmul(A[idx_test[i]][idx_test], test_labels_onehot) + output[i]
          # 2. apply softmax
          probs = F.softmax(logits, dim=0)
          test_labels_onehot[i] = torch.distributions.OneHotCategorical(probs).sample()
        output = test_labels_onehot
    else:
        output = torch.log_softmax(output, dim=-1)
    
    loss_test = F.nll_loss(output, labels[idx_test])
    acc_test = accuracy(output, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

Train()
test()




# Old versions
# def compute_interaction_term_train():
#     interaction = torch.zeros((len(idx_train), labels.max().item() + 1))
#     for i in range(len_train):
#         for j in range(len_train):
#             if i != j:
#                 interaction[i][labels[idx_train[j]]] += A[idx_train[i]][idx_train[j]]
#     return interaction

# def compute_interaction_term_val():
#     interaction = torch.zeros((len(idx_val), labels.max().item() + 1))
#     for i in range(len_val):
#         for j in range(len_train):
#             interaction[i][labels[idx_train[j]]] += A[idx_val[i]][idx_train[j]]
#         for j in range(len_val):
#             if i != j:
#                 interaction[i][labels[idx_val[j]]] += A[idx_val[i]][idx_val[j]]
#     return interaction

# def compute_interaction_term_test():
#     interaction = torch.zeros((len(idx_test), labels.max().item() + 1))
#     for i in range(len_test):
#         for j in range(len_train):
#             interaction[i][labels[idx_train[j]]] += A[idx_test[i]][idx_train[j]]
#         for j in range(len_val):
#             interaction[i][labels[idx_val[j]]] += A[idx_test[i]][idx_val[j]]
#     return interaction