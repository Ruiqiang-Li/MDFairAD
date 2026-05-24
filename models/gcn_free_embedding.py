import torch
import torch.nn as nn
from models import GCN_Body
from models.gcn import GCN_Body_bn


class GCN_free_embedding(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nsamples, layers, name="", enable_bn=False):
        super(GCN_free_embedding, self).__init__()
        self.name = name
        self.nhid = nhid
        if enable_bn:
            self.body = GCN_Body_bn(nfeat, nhid, dropout, layers)
        else:
            self.body = GCN_Body(nfeat, nhid, dropout, layers)
        self.fc = nn.Linear(nhid, nclass)

        self.free_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nsamples, nfeat)))
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.body(self.free_embedding, edge_index)
        x = self.fc(h)

        return x, h

    def get_embeddings(self, x, edge_index):
        h = self.body(self.free_embedding, edge_index)
        return h
