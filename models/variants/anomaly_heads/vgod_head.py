"""VGOD anomaly detection head combining attribute reconstruction with neighborhood variance."""

import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor

from models.gcn import GCN_Body, GCN_Body_bn


class MeanConv(MessagePassing):
    def __init__(self, aggr: str = 'mean', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index, edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        return x_j


class CovConv(MessagePassing):
    def __init__(self, aggr: str = 'mean', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index, ner_mean, edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, ner_mean=ner_mean[edge_index[1]], edge_weight=edge_weight, size=size)
        return torch.sum(out, dim=-1)

    def message(self, x_j: Tensor, ner_mean) -> Tensor:
        return (x_j - ner_mean) ** 2


class VGODReconDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, layers, dropout, enable_bn=False):
        super().__init__()
        body_layers = max(1, min(layers, 3))
        if enable_bn:
            self.body = GCN_Body_bn(in_dim, in_dim, dropout, body_layers)
        else:
            self.body = GCN_Body(in_dim, in_dim, dropout, body_layers)
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight.data)
        if self.proj.bias is not None:
            self.proj.bias.data.fill_(0.0)

    def forward(self, z, edge_index):
        h = self.body(z, edge_index)
        return self.proj(h)


class VGODVarianceDecoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb_dim)
        self.mean = MeanConv()
        self.cov = CovConv()

    def forward(self, z, edge_index):
        h = self.lin(z)
        h = h / torch.norm(h, dim=-1, keepdim=True).clamp_min(1e-6)
        mean = self.mean(h, edge_index)
        return self.cov(h, edge_index, mean)


class VGODAnomalyHead(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()
        self.args = args
        var_dim = int(getattr(args, 'e2e_vgod_emb_dim', 0))
        if var_dim <= 0:
            var_dim = in_dim
        self.recon_decoder = VGODReconDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            layers=max(1, args.layers - 1),
            dropout=args.dropout,
            enable_bn=getattr(args, 'enable_bn', False),
        )
        self.var_decoder = VGODVarianceDecoder(in_dim=in_dim, emb_dim=var_dim)

    def compute_loss(self, embedding, data, epoch=None):
        x_hat = self.recon_decoder(embedding, data.edge_index)
        score_recon = torch.sum(torch.square(data.x - x_hat), dim=-1)
        neg_edge_index = pyg_utils.negative_sampling(
            data.edge_index,
            num_nodes=embedding.size(0),
            num_neg_samples=data.edge_index.size(1),
        )
        pos_var = self.var_decoder(embedding, data.edge_index)
        neg_var = self.var_decoder(embedding, neg_edge_index)
        idx = data.idx_train
        loss_attr = score_recon[idx].mean()
        loss_struct = pos_var.mean() - neg_var.mean()
        str_epoch = int(getattr(self.args, 'e2e_vgod_str_epoch', 10))
        if epoch is not None and int(epoch) >= str_epoch:
            loss_struct = loss_struct * 0.0
        alpha = getattr(self.args, 'e2e_rec_alpha', 0.5)
        loss_head = alpha * loss_attr + (1.0 - alpha) * loss_struct
        return {
            'x_hat': x_hat,
            'score_recon': score_recon,
            'score_var': pos_var,
            'pos_var': pos_var,
            'neg_var': neg_var,
            'neg_edge_index': neg_edge_index,
            'loss_head': loss_head,
            'loss_attr': loss_attr,
            'loss_struct': loss_struct,
        }

    def score(self, head_outputs, data):
        score_recon = head_outputs['score_recon']
        score_var = head_outputs['score_var']

        def std_scale(x):
            mean = torch.mean(x)
            std = torch.std(x).clamp_min(1e-6)
            return (x - mean) / std

        alpha = float(getattr(self.args, 'e2e_vgod_var_weight', 1.0))
        return std_scale(score_recon) + alpha * std_scale(score_var)
