from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, load_data_split
from pygcn.models import MLP, PMLE
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
parser.add_argument('--epochs', type=int, default=5000,
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
parser.add_argument('--num_gibbs_iters', type=int, default=10000, help='Numner of iterations for the Gibbs Sampler')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--use_gibbs', action='store_true', default=False, help='Use Gibbs Sampler')

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

# print(labels)
# print(idx_val)
# print(idx_test)
# print(features.shape)
# print(labels.shape)
# print('Labels', labels)
print('Test data size', len_test)
print('Train data size', len_train)
print('Val data size', len_val)
A = A.to_dense()
A = A - torch.diag_embed(torch.diagonal(A))

# print('Adjancency', A.shape)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

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



# Model and optimizer

model = GCNII(nfeat=num_features,
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer_pmle = optim.Adam(model_pmle.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)

model_pmle = PMLE_Linear(nhid=args.hidden, 
                            nclass=labels.max().item() + 1,
                            dropout = args.dropout)

    
model.load_state_dict(torch.load(dataset +'_GCN.pkl'))
model.eval()
new_features = model.forward_last(features, A)

model_pmle.linear = model.fcs[-1]
model_pmle.beta.weight.data.fill_(0.0)

# new_features = rand_prop(features, training=False)
# new_features = model.forward_last(features).data
# print('New Features', new_features)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

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


A = A.to_dense()

interaction_train = compute_interaction_term_train()
interaction_val = compute_interaction_term_val()
interaction_test = compute_interaction_term_test()
if args.cuda:
    interaction_train = interaction_train.cuda()
    interaction_val = interaction_val.cuda()
    interaction_test = interaction_test.cuda()


def train_pmle(epoch):
    t = time.time()
    
    # print(features[idx_train].shape, interaction_train.unsqueeze(dim=1).shape)
    # X = torch.cat((features[idx_train], interaction_train), 1)
    X = new_features
    model_pmle.train()
    optimizer_pmle.zero_grad()
    output = model_pmle(new_features[idx_train], interaction_train)

    
    loss_train = F.nll_loss(output, labels[idx_train])


    acc_train = accuracy(output, labels[idx_train])

    loss_train.backward()

    optimizer_pmle.step()

    # if not args.fastmode:
        # X = rand_prop(X,training=False)
        # output = model_pmle(X[idx_train, interaction_train)
        # output = torch.log_softmax(output, dim=-1)
    
    model_pmle.eval()
    output_val = model_pmle(new_features[idx_val], interaction_val)
    loss_val = F.nll_loss(output_val, labels[idx_val]) 
    acc_val = accuracy(output_val, labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(), acc_val.item()

def Train_pmle():
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

        l, a = train_pmle(epoch)
        loss_values.append(l)
        acc_values.append(a)

        # print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model_pmle.state_dict(), 'checkpoints/' + dataset +'_pmle_on_grand.pkl')

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
    model_pmle.load_state_dict(torch.load('checkpoints/' + dataset +'_pmle_on_grand.pkl'))


# use_gibbs = True


def test_pmle():
    print(model_pmle.beta.weight.data)
    model_pmle.eval()
    output = model_pmle(new_features[idx_test], interaction_test)
    loss_test = F.nll_loss(output, labels[idx_test])
    acc_test = accuracy(output, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# model_pmle.load_state_dict(torch.load(dataset +'_pmle_on_grand.pkl'))
# test_pmle()

Train_pmle()
test_pmle()
