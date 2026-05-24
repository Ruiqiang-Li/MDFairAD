"""DOMINANT-style anomaly detection head with attribute and structure GCN decoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcn import GCN_Body, GCN_Body_bn


class DominantAttributeDecoder(nn.Module):
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


class DominantStructureDecoder(nn.Module):
    def __init__(self, in_dim, layers, dropout, enable_bn=False):
        super().__init__()
        body_layers = max(1, min(layers, 3))
        if enable_bn:
            self.body = GCN_Body_bn(in_dim, in_dim, dropout, body_layers)
        else:
            self.body = GCN_Body(in_dim, in_dim, dropout, body_layers)

    def edge_logits(self, z, edge_index):
        h = self.body(z, edge_index)
        src, dst = edge_index
        return (h[src] * h[dst]).sum(dim=1)


class DominantAnomalyHead(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()
        self.args = args
        self.attr_decoder = DominantAttributeDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            layers=max(1, args.layers - 1),
            dropout=args.dropout,
            enable_bn=getattr(args, 'enable_bn', False),
        )
        self.struct_decoder = DominantStructureDecoder(
            in_dim=in_dim,
            layers=max(1, args.layers - 1),
            dropout=args.dropout,
            enable_bn=getattr(args, 'enable_bn', False),
        )

    @staticmethod
    def sample_negative_edges(edge_index, num_nodes, num_neg=None):
        num_pos = edge_index.shape[1]
        if num_neg is None:
            num_neg = num_pos
        device = edge_index.device
        src = torch.randint(0, num_nodes, (num_neg,), device=device)
        dst = torch.randint(0, num_nodes, (num_neg,), device=device)
        return torch.stack([src, dst], dim=0)

    def forward(self, embedding, edge_index):
        x_hat = self.attr_decoder(embedding, edge_index)
        pos_logits = self.struct_decoder.edge_logits(embedding, edge_index)
        neg_edge_index = self.sample_negative_edges(edge_index, embedding.shape[0])
        neg_logits = self.struct_decoder.edge_logits(embedding, neg_edge_index)
        return x_hat, pos_logits, neg_logits, neg_edge_index

    def compute_loss(self, embedding, data, epoch=None):
        x_hat, pos_logits, neg_logits, neg_edge_index = self.forward(embedding, data.edge_index)
        idx = data.idx_train

        loss_attr = F.mse_loss(x_hat[idx], data.x[idx])
        pos_label = torch.ones_like(pos_logits)
        neg_label = torch.zeros_like(neg_logits)
        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_label)
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_label)
        loss_struct = 0.5 * (loss_pos + loss_neg)

        alpha = getattr(self.args, 'e2e_rec_alpha', 0.5)
        loss_head = alpha * loss_attr + (1.0 - alpha) * loss_struct
        return {
            'x_hat': x_hat,
            'pos_logits': pos_logits,
            'neg_logits': neg_logits,
            'neg_edge_index': neg_edge_index,
            'loss_head': loss_head,
            'loss_attr': loss_attr,
            'loss_struct': loss_struct,
        }

    def score(self, x_hat, pos_logits=None, neg_logits=None, neg_edge_index=None, data=None):
        if isinstance(x_hat, dict):
            head_outputs = x_hat
            data = pos_logits
            x_hat = head_outputs['x_hat']
            pos_logits = head_outputs['pos_logits']
            neg_logits = head_outputs['neg_logits']
            neg_edge_index = head_outputs['neg_edge_index']
        score_attr = torch.mean((x_hat - data.x) ** 2, dim=1)

        pos_prob = torch.sigmoid(pos_logits)
        pos_err = 1.0 - pos_prob
        pos_src, pos_dst = data.edge_index
        pos_score = torch.zeros(data.x.shape[0], device=data.x.device)
        pos_degree = torch.zeros(data.x.shape[0], device=data.x.device)
        pos_score.index_add_(0, pos_src, pos_err)
        pos_score.index_add_(0, pos_dst, pos_err)
        pos_degree.index_add_(0, pos_src, torch.ones_like(pos_err))
        pos_degree.index_add_(0, pos_dst, torch.ones_like(pos_err))
        pos_degree = pos_degree.clamp_min(1.0)
        pos_score = pos_score / pos_degree

        neg_prob = torch.sigmoid(neg_logits)
        neg_err = neg_prob
        neg_src, neg_dst = neg_edge_index
        neg_score = torch.zeros(data.x.shape[0], device=data.x.device)
        neg_degree = torch.zeros(data.x.shape[0], device=data.x.device)
        neg_score.index_add_(0, neg_src, neg_err)
        neg_score.index_add_(0, neg_dst, neg_err)
        neg_degree.index_add_(0, neg_src, torch.ones_like(neg_err))
        neg_degree.index_add_(0, neg_dst, torch.ones_like(neg_err))
        neg_degree = neg_degree.clamp_min(1.0)
        neg_score = neg_score / neg_degree

        observed_degree = torch.bincount(pos_src, minlength=data.x.shape[0]).float()
        observed_degree = observed_degree + torch.bincount(pos_dst, minlength=data.x.shape[0]).float()
        degree_center = torch.median(observed_degree)
        degree_scale = torch.mean(torch.abs(observed_degree - degree_center)).clamp_min(1.0)
        degree_dev = torch.abs(observed_degree - degree_center) / degree_scale

        score_struct = 0.5 * pos_score + 0.35 * neg_score + 0.15 * degree_dev

        alpha = getattr(self.args, 'e2e_rec_alpha', 0.5)
        attr_weight = alpha
        struct_weight = 1.0 - alpha
        return attr_weight * score_attr + struct_weight * score_struct
