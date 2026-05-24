import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, layers, h1=None, h2=None, name="", enable_bn=False):
        super(GCN, self).__init__()
        self.name = name
        self.nhid = nhid
        if enable_bn:
            self.body = GCN_Body_bn(nfeat, nhid, dropout, layers)
        else:
            self.body = GCN_Body(nfeat, nhid, dropout, layers)
        self.fc = nn.Linear(nhid, nclass)
        self.h1 = h1
        self.h2 = h2
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.body(x, edge_index)

        if self.h1 is not None:
            h = h - self.h1

        if self.h2 is not None:
            h = h - self.h2

        x = self.fc(h)
        return x, h

    def get_embeddings(self, x, edge_index):
        h = self.body(x, edge_index)

        if self.h1 is not None:
            h = h - self.h1

        if self.h2 is not None:
            h = h - self.h2

        return h


class GCN_Body_bn(nn.Module):
    def __init__(self, nfeat, nhid, dropout, layers):
        super(GCN_Body_bn, self).__init__()
        self.layers = layers
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)

    def forward(self, x, edge_index):
        if self.layers == 1:
            x = self.gc1(x, edge_index)
            x = self.bn1(x)
        elif self.layers == 2:
            x = self.gc1(x, edge_index)
            x = self.bn1(x)
            x = self.gc2(x, edge_index)
            x = self.bn2(x)
        elif self.layers == 3:
            x = self.gc1(x, edge_index)
            x = self.bn1(x)
            x = self.gc2(x, edge_index)
            x = self.bn2(x)
            x = self.gc3(x, edge_index)
            x = self.bn3(x)
        else:
            raise ValueError("Invalid number of layers")
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout, layers):
        super(GCN_Body, self).__init__()
        self.layers = layers
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nhid)

    def forward(self, x, edge_index):
        if self.layers == 1:
            x = self.gc1(x, edge_index)
        elif self.layers == 2:
            x = self.gc1(x, edge_index)
            x = self.gc2(x, edge_index)
        elif self.layers == 3:
            x = self.gc1(x, edge_index)
            x = self.gc2(x, edge_index)
            x = self.gc3(x, edge_index)
        else:
            raise ValueError("Invalid number of layers")

        return x
