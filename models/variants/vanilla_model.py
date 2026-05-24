from constants import VANILLA
from models.variants import SingleBaseModel
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from models.helpers import WandbSingleton
from models.variants.anomaly_head_model import CoLAAnomalyHead, CONADAnomalyHead, DominantAnomalyHead, VGODAnomalyHead

class VanillaModel(SingleBaseModel):
    def __init__(self, args, data):
        super(VanillaModel, self).__init__(args, data)

        model = GCN(nfeat=data.x.shape[1],
                    nhid=args.hidden,
                    nclass=args.num_classes,
                    dropout=args.dropout)

        self.model = model
        self.edge_index = data.edge_index
        optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_model = optimizer_model

    def _get_best_condition(self, loss_val, tradeoff_val):
        return loss_val < self.best_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.name = VANILLA
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nhid, nclass),
        )
    def forward(self, x, edge_index):
        h = self.body(x, edge_index)
        x = self.fc(h)
        return x, h


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        return x
