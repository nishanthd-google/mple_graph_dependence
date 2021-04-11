import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from pygcn.layers import GraphConvolution, MLPLayer, InteractionUnit

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self, x, adj):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        
        x = F.relu(self.gc1(x, adj))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.gc2(x, adj)

        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn =False):
    # def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        # self.input_droprate = input_droprate
        # self.hidden_droprate = hidden_droprate
        # self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        # x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        # x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x

class PMLE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn =False):
        super(PMLE, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        # self.input_droprate = input_droprate
        # self.hidden_droprate = hidden_droprate
        # self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.beta = InteractionUnit()
        self.use_bn = use_bn
        
    def forward(self, x, y):
         
        if self.use_bn: 
            x = self.bn1(x)
        # x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        # x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x) + self.beta(y)
        return x


class PMLE_Linear(nn.Module):
    def __init__(self, nhid, nclass, hidden_droprate, use_bn =False):
        super(PMLE_Linear, self).__init__()

        self.layer = MLPLayer(nhid, nclass)

        # self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        # self.is_cuda = is_cuda
        # self.bn1 = nn.BatchNorm1d(nfeat)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.beta = InteractionUnit()
        # self.use_bn = use_bn
        
    def forward(self, x, y):
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer(x) + self.beta(y)
        return x

class MLP_Grand(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP_Grand, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x

    def forward_last(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        # x = F.dropout(x, self.hidden_droprate, training=self.training)
        # x = self.layer2(x)

        return x