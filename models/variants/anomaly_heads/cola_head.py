"""CoLA anomaly detection head: subgraph contrastive scoring with mini-batch training."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoLAGCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, dim=1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, dim=1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, dim=1).values


class WSReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, self.hidden_dim)
        out = torch.mul(seq, sim)
        return torch.sum(out, dim=1)


class CoLADiscriminator(nn.Module):
    def __init__(self, hidden_dim, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.negsamp_round = negsamp_round
        nn.init.xavier_uniform_(self.f_k.weight.data)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scores = [self.f_k(h_pl, c)]
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), dim=0)
            scores.append(self.f_k(h_pl, c_mi))
        return torch.cat(tuple(scores), dim=0)


class CoLAModel(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super().__init__()
        self.read_mode = readout
        self.gcn = CoLAGCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout(n_h)
        else:
            raise ValueError(f'Unsupported CoLA readout: {readout}')
        self.disc = CoLADiscriminator(n_h, negsamp_round)

    def forward(self, seq1, adj):
        h_1 = self.gcn(seq1, adj)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, :-1, :])
            h_mv = h_1[:, -1, :]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
        return self.disc(c, h_mv)


class CoLASubgraphBuilder:
    def __init__(self, subgraph_size, restart_prob=0.9, walk_multiplier=3):
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.walk_multiplier = walk_multiplier

    @staticmethod
    def _build_neighbors(edge_index, num_nodes):
        edge_index_cpu = edge_index.detach().cpu()
        neighbors = [[] for _ in range(num_nodes)]
        for src, dst in edge_index_cpu.t().tolist():
            neighbors[src].append(dst)
            if src != dst:
                neighbors[dst].append(src)
        for node in range(num_nodes):
            if len(neighbors[node]) == 0:
                neighbors[node].append(node)
        return neighbors

    def _sample_single(self, center, neighbors):
        reduced_size = self.subgraph_size - 1
        visited = []
        current = center
        max_steps = max(self.subgraph_size * self.walk_multiplier, self.subgraph_size + 2)
        for _ in range(max_steps):
            if random.random() < self.restart_prob:
                current = center
            current = random.choice(neighbors[current])
            if current != center and current not in visited:
                visited.append(current)
            if len(visited) >= reduced_size:
                break
        retry = 0
        while len(visited) < reduced_size and retry < 10:
            current = center
            for _ in range(max_steps + self.subgraph_size):
                if random.random() < self.restart_prob:
                    current = center
                current = random.choice(neighbors[current])
                if current != center and current not in visited:
                    visited.append(current)
                if len(visited) >= reduced_size:
                    break
            retry += 1
        if len(visited) == 0:
            visited = [center]
        while len(visited) < reduced_size:
            visited.append(visited[len(visited) % len(visited)])
        visited = visited[:reduced_size]
        visited.append(center)
        return visited

    def generate(self, edge_index, num_nodes):
        neighbors = self._build_neighbors(edge_index, num_nodes)
        return [self._sample_single(i, neighbors) for i in range(num_nodes)]


class CoLAAnomalyHead(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()
        self.args = args
        self.embedding_dim = int(getattr(args, 'e2e_cola_embedding_dim', 64))
        self.negsamp_ratio = int(getattr(args, 'e2e_cola_negsamp_ratio', 1))
        self.subgraph_size = int(getattr(args, 'e2e_cola_subgraph_size', 4))
        self.batch_size = int(getattr(args, 'e2e_cola_batch_size', 300))
        self.readout = str(getattr(args, 'e2e_cola_readout', 'avg'))
        self.model = CoLAModel(in_dim, self.embedding_dim, 'prelu', self.negsamp_ratio, self.readout)
        self.subgraph_builder = CoLASubgraphBuilder(self.subgraph_size)

    @staticmethod
    def _normalize_adj(adj):
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        return adj * d_inv_sqrt.unsqueeze(1) * d_inv_sqrt.unsqueeze(0)

    def _build_batch_tensors(self, embedding, subgraphs, idx, edge_index):
        device = embedding.device
        hidden_dim = embedding.size(1)
        base_size = self.subgraph_size
        src_all = edge_index[0].detach().cpu().tolist()
        dst_all = edge_index[1].detach().cpu().tolist()
        adj_batch = []
        feat_batch = []
        for node_idx in idx:
            nodes = subgraphs[node_idx]
            sub_emb = embedding[nodes]
            sub_adj = torch.zeros((base_size, base_size), device=device)
            base_nodes = nodes
            position = {node: pos for pos, node in enumerate(base_nodes)}
            for s, d in zip(src_all, dst_all):
                if s in position and d in position:
                    sub_adj[position[s], position[d]] = 1.0
            sub_adj = self._normalize_adj(sub_adj + torch.eye(base_size, device=device))
            sub_adj = torch.cat((sub_adj, torch.zeros((1, base_size), device=device)), dim=0)
            extra_col = torch.zeros((base_size + 1, 1), device=device)
            extra_col[-1, 0] = 1.0
            sub_adj = torch.cat((sub_adj, extra_col), dim=1)
            sub_feat = torch.cat((sub_emb[:-1], torch.zeros((1, hidden_dim), device=device), sub_emb[-1:].clone()), dim=0)
            adj_batch.append(sub_adj.unsqueeze(0))
            feat_batch.append(sub_feat.unsqueeze(0))
        return torch.cat(feat_batch, dim=0), torch.cat(adj_batch, dim=0)

    def _run_model_batched(self, embedding, data, node_indices, subgraphs):
        logits_chunks = []
        for start in range(0, len(node_indices), self.batch_size):
            batch_idx = node_indices[start:start + self.batch_size]
            if len(batch_idx) == 0:
                continue
            batch_feat, batch_adj = self._build_batch_tensors(embedding, subgraphs, batch_idx, data.edge_index)
            batch_logits = self.model(batch_feat, batch_adj).view(-1)
            cur_batch_size = len(batch_idx)
            pos_logits = batch_logits[:cur_batch_size]
            neg_logits = batch_logits[cur_batch_size:].view(self.negsamp_ratio, cur_batch_size).mean(dim=0)
            logits_chunks.append((batch_idx, pos_logits, neg_logits))
        return logits_chunks

    def compute_loss(self, embedding, data, epoch=None):
        num_nodes = embedding.size(0)
        subgraphs = self.subgraph_builder.generate(data.edge_index, num_nodes)
        train_idx = data.idx_train.detach().cpu().tolist()
        pos_logits = torch.zeros(num_nodes, device=embedding.device)
        neg_logits = torch.zeros(num_nodes, device=embedding.device)
        loss_sum = torch.zeros((), device=embedding.device)
        loss_count = 0
        pos_weight = torch.tensor([self.negsamp_ratio], device=embedding.device, dtype=embedding.dtype)

        for batch_idx, batch_pos_logits, batch_neg_logits in self._run_model_batched(embedding, data, train_idx, subgraphs):
            batch_idx_tensor = torch.tensor(batch_idx, device=embedding.device, dtype=torch.long)
            pos_logits[batch_idx_tensor] = batch_pos_logits
            neg_logits[batch_idx_tensor] = batch_neg_logits
            logits_train = torch.cat((batch_pos_logits.unsqueeze(1), batch_neg_logits.unsqueeze(1)), dim=0)
            labels_train = torch.cat(
                (
                    torch.ones_like(batch_pos_logits).unsqueeze(1),
                    torch.zeros_like(batch_neg_logits).unsqueeze(1),
                ),
                dim=0,
            )
            batch_loss = F.binary_cross_entropy_with_logits(
                logits_train,
                labels_train,
                reduction='none',
                pos_weight=pos_weight,
            ).mean()
            loss_sum = loss_sum + batch_loss * len(batch_idx)
            loss_count += len(batch_idx)

        loss_head = loss_sum / max(loss_count, 1)

        full_idx = list(range(num_nodes))
        for batch_idx, batch_pos_logits, batch_neg_logits in self._run_model_batched(
            embedding.detach(), data, full_idx, subgraphs
        ):
            batch_idx_tensor = torch.tensor(batch_idx, device=embedding.device, dtype=torch.long)
            pos_logits[batch_idx_tensor] = batch_pos_logits.detach()
            neg_logits[batch_idx_tensor] = batch_neg_logits.detach()

        return {
            'pos_logits': pos_logits,
            'neg_logits': neg_logits,
            'subgraphs': subgraphs,
            'loss_head': loss_head,
            'loss_attr': loss_head,
            'loss_struct': torch.zeros_like(loss_head),
        }

    def score(self, head_outputs, data):
        pos_prob = torch.sigmoid(head_outputs['pos_logits'])
        neg_prob = torch.sigmoid(head_outputs['neg_logits'])
        return -(pos_prob - neg_prob)
