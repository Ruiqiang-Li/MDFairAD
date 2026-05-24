"""CONAD anomaly detection head: contrastive head over shuffled attribute and graph views."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CONADSharedEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.graph_encoder_1 = GCNConv(in_dim, hidden_dim)
        self.graph_encoder_2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.graph_encoder_1(x, edge_index))
        h = F.relu(self.graph_encoder_2(h, edge_index))
        return h


class CONADAttributeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.mlp(x)


class CONADDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.disc = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.disc.weight.data)
        if self.disc.bias is not None:
            self.disc.bias.data.fill_(0.0)

    def forward(self, h_attr, h_attr_shfs, h_graph, h_graph_shfs):
        logit1 = self.disc(h_attr, h_graph)
        logit2 = self.disc(h_attr_shfs, h_graph)
        logit3 = self.disc(h_attr, h_graph_shfs)
        logit4 = self.disc(h_attr_shfs, h_graph_shfs)
        return torch.cat([logit1, logit2, logit3, logit4], dim=1)


class CONADModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.attr_encoder = CONADAttributeEncoder(in_dim, hidden_dim)
        self.graph_encoder = CONADSharedEncoder(in_dim, hidden_dim)
        self.discriminator = CONADDiscriminator(hidden_dim)

    def forward(self, edge_index, graph_feat, graph_feat_shfs, attr_feat, attr_feat_shfs):
        h_attr = self.attr_encoder(attr_feat)
        h_attr_shfs = self.attr_encoder(attr_feat_shfs)
        h_graph = self.graph_encoder(graph_feat, edge_index)
        h_graph_shfs = self.graph_encoder(graph_feat_shfs, edge_index)
        logits = self.discriminator(h_attr, h_attr_shfs, h_graph, h_graph_shfs)
        return logits, h_attr, h_graph

    def score(self, edge_index, graph_feat, attr_feat):
        h_attr = self.attr_encoder(attr_feat)
        h_graph = self.graph_encoder(graph_feat, edge_index)
        return self.discriminator.disc(h_attr, h_graph).view(-1)


class CONADAnomalyHead(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()
        self.args = args
        self.hidden_dim = int(getattr(args, 'e2e_conad_hidden_dim', 64))
        self.model = CONADModel(in_dim=in_dim, hidden_dim=self.hidden_dim)

    @staticmethod
    def _shuffle_rows(x):
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm], perm

    def compute_loss(self, embedding, data, epoch=None):
        graph_feat_shfs, graph_perm = self._shuffle_rows(embedding)
        attr_feat_shfs, attr_perm = self._shuffle_rows(embedding)
        logits, h_attr, h_graph = self.model(
            data.edge_index,
            embedding,
            graph_feat_shfs,
            embedding,
            attr_feat_shfs,
        )
        train_idx = data.idx_train
        logits_train = logits[train_idx]
        labels_train = torch.cat(
            [
                torch.ones((train_idx.numel(), 1), device=embedding.device),
                torch.zeros((train_idx.numel(), 3), device=embedding.device),
            ],
            dim=1,
        )
        loss_mat = F.binary_cross_entropy_with_logits(logits_train, labels_train, reduction='none')
        loss_head = loss_mat.mean()
        score_vec = self.model.score(data.edge_index, embedding, embedding)
        return {
            'logits': logits,
            'labels_train': labels_train,
            'graph_perm': graph_perm,
            'attr_perm': attr_perm,
            'h_attr': h_attr,
            'h_graph': h_graph,
            'score_vec': score_vec,
            'loss_head': loss_head,
            'loss_attr': loss_head,
            'loss_struct': torch.zeros_like(loss_head),
            'loss_contrast': loss_head,
        }

    def score(self, head_outputs, data):
        return -head_outputs['score_vec']
