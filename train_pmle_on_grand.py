from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP, MLP_Grand, PMLE_Linear
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
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--use_gibbs', action='store_true', default=False, help='Use Gibbs Sampler')
parser.add_argument('--num_gibbs_iters', type=int, default=10000, help='Number of iterations for the Gibbs Sampler')
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
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)


len_train = len(idx_train)
len_val = len(idx_val)
len_test = len(idx_test)

def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()

def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)

        features = masks * features
            
    else:
            
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)    
    return features

# Model and optimizer
model = MLP_Grand(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn = args.use_bn)

model_pmle = PMLE_Linear(nhid=args.hidden, nclass=labels.max().item() + 1, hidden_droprate=args.hidden_droprate)


optimizer_pmle = optim.Adam(model_pmle.parameters(),
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
    model_pmle.cuda()
    features = features.cuda()

    
model.load_state_dict(torch.load(dataset +'_grand.pkl'))

model_pmle.layer = model.layer2
model_pmle.beta.weight.data.fill_(0.0)

def test():
    model.eval()
    X = features
    X = rand_prop(X, training=False)
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

test()

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

def compute_no_val_interaction_term_test():
    interaction = torch.zeros((len(idx_test), labels.max().item() + 1))
    label_tensor = torch.zeros((len(idx_train), labels.max().item() + 1), device=device)
    label_tensor.zero_()
    label_tensor.scatter_(1, torch.reshape(labels[idx_train], (-1,1)), 1)
    interaction = torch.matmul(A[idx_test][:, idx_train], label_tensor)
    return interaction


A = A.to_dense()
# make A zero-diagonal
A = A - torch.diag_embed(torch.diagonal(A))
interaction_train = compute_interaction_term_train()
interaction_val = compute_interaction_term_val()
interaction_test = compute_interaction_term_test()
if args.cuda:
    interaction_train = interaction_train.cuda()
    interaction_val = interaction_val.cuda()
    interaction_test = interaction_test.cuda()

new_features = rand_prop(features,training=False)
new_features = model.forward_last(new_features).data

def train_pmle(epoch):
    t = time.time()
    
    # print(features[idx_train].shape, interaction_train.unsqueeze(dim=1).shape)
    # X = torch.cat((features[idx_train], interaction_train), 1)
    X = features
    model_pmle.train()
    optimizer_pmle.zero_grad()
    X_list = []
    K = args.sample
    for k in range(K):
        X_list.append(rand_prop(X, training=True))

    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(model_pmle(model.forward_last(X_list[k][idx_train]).data, interaction_train), dim=-1))

    
    loss_train = 0.
    for k in range(K):
        loss_train += F.nll_loss(output_list[k], labels[idx_train])
     
        
    loss_train = loss_train/K

    # output = torch.log_softmax(model_pmle(X, interaction_train), dim=-1)

    
    # loss_train = F.nll_loss(output, labels[idx_train])

    acc_train = accuracy(output_list[0], labels[idx_train])

    loss_train.backward()

    optimizer_pmle.step()

    # if not args.fastmode:
        # X = rand_prop(X,training=False)
        # output = model_pmle(X[idx_train, interaction_train)
        # output = torch.log_softmax(output, dim=-1)
    
    model_pmle.eval()
    output_val = torch.log_softmax(model_pmle(new_features[idx_val], interaction_val), dim=-1) 
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

num_iter = args.num_gibbs_iters

def test_pmle():
    print(model_pmle.beta.weight.data)

## Gibbs sampler
  # A is whole adjacency graph now
    num_classes = labels.max().item() + 1
    model_pmle.eval()
    output = model_pmle(new_features[idx_test], interaction_test)
    # output_probs = torch.log_softmax(output, dim=-1)
    # test_labels = output_probs.max(1)[1].type_as(labels)
    if args.use_gibbs:
        # init randomly
        test_labels = torch.LongTensor(len_test,1).random_(0, num_classes)
        test_labels_onehot = torch.FloatTensor(len_test, num_classes)
        test_labels_onehot.zero_()
        test_labels_onehot.scatter_(1, test_labels, 1)
        for iter in range(num_iter):
          # choose a random node
          i = np.random.randint(0, len_test)
          # re-sample from conditional distribution
          # 1. calculate logits for each class
          logits = model_pmle.beta.weight.data * torch.matmul(A[idx_test[i]][idx_test], test_labels_onehot) + output[i]
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


model_pmle.load_state_dict(torch.load(dataset +'_pmle_on_grand.pkl'))
test_pmle()

Train_pmle()
test_pmle()
