# =============================================================================
# Attribute branch of MDFP.
#
# Disentangled multi-channel encoder (DisGCN) followed by a Gumbel channel
# masker that suppresses sensitive-correlated channels, supervised by a
# branch-level BCE on the labelled training split and a feature-covariance
# fairness regulariser, corresponding to Sec. 3 of the paper.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.modules.loss import _Loss
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
from copy import deepcopy

from constants import ATTRIBUTE
import metrics as metrics_utils
from models.helpers import WandbSingleton
from models.variants.single_basemodel import SingleBaseModel


class _DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(_DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = out_dim // channels
        self.reduce = reduce

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(
                    nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(
                    Linear(in_channels=self.per_channel_dim, out_channels=self.per_channel_dim,
                           bias=False, weight_initializer='glorot'))
            else:
                self.conv_layers.append(
                    Linear(in_channels=self.in_dim, out_channels=self.per_channel_dim,
                           bias=False, weight_initializer='glorot'))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float),
                         requires_grad=True)
            for _ in range(self.channels))

    def get_reddim_k(self, x):
        return [self.lin_layers[k](x) for k in range(self.channels)]

    def get_k_feature(self, x):
        return [x for _ in range(self.channels)]

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[1]
        z_feats = self.get_reddim_k(x) if self.reduce else self.get_k_feature(x)
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(
                edge_index_copy.storage.value() * edge_weight[:, k])
            out = self.propagate(edge_index_copy, x=c_temp)
            out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
        return torch.cat(c_feats, dim=1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class _NeiborAssigner(nn.Module):
    def __init__(self, nfeats, channels):
        super(_NeiborAssigner, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        return torch.softmax(self.layers(features_pair), dim=1)


class _DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, chan_num, layer_num, dropout=0.5):
        super(_DisGCN, self).__init__()
        self.chan_num = chan_num
        self.assigner = _NeiborAssigner(nfeat, chan_num)
        self.disenlayers = nn.ModuleList()
        for i in range(layer_num - 1):
            in_dim = nfeat if i == 0 else nhid
            self.disenlayers.append(_DisenLayer(in_dim, nhid, chan_num))
        self.dropout = nn.Dropout(dropout)

    def init_parameters(self):
        for item in self.parameters():
            torch.nn.init.normal_(item, mean=0, std=1)

    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected SparseTensor edge_index"
        feats_pair = torch.cat(
            [x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        edge_weight = self.assigner(feats_pair.detach())
        for layer in self.disenlayers:
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        return x


class _ChannelMasker(nn.Module):
    def __init__(self, hid_num):
        super(_ChannelMasker, self).__init__()
        self.weights = nn.Parameter(
            torch.distributions.Uniform(0, 1).sample((hid_num, 2)))

    def forward(self, x):
        mask = F.gumbel_softmax(self.weights, tau=1, hard=False)[:, 0]
        return x * mask


class _FeatCov(_Loss):
    def forward(self, features, sens):
        cov = 0
        for k in range(features.shape[1]):
            cov += torch.abs(torch.mean(
                (sens - torch.mean(sens)) * (features[:, k] - torch.mean(features[:, k]))))
        return cov


class _DistCor(_Loss):
    def _distance_correlation(self, c1, c2):
        assert c1.shape[1] == c2.shape[1]
        corr = 0
        for i in range(c1.shape[1]):
            a = c1[:, i].unsqueeze(1)
            b = c2[:, i].unsqueeze(1)
            ma = torch.sqrt(torch.sum((a.unsqueeze(0) - a.unsqueeze(1)) ** 2, dim=-1) + 1e-12)
            mb = torch.sqrt(torch.sum((b.unsqueeze(0) - b.unsqueeze(1)) ** 2, dim=-1) + 1e-12)
            Ma = ma - ma.mean(0, keepdim=True) - ma.mean(1, keepdim=True) + ma.mean()
            Mb = mb - mb.mean(0, keepdim=True) - mb.mean(1, keepdim=True) + mb.mean()
            n2 = Ma.shape[0] * Ma.shape[1]
            gxy = (Ma * Mb).sum() / n2
            gxx = (Ma * Ma).sum() / n2
            gyy = (Mb * Mb).sum() / n2
            corr += gxy / torch.sqrt(gxx * gyy + 1e-9)
        return corr

    def forward(self, c1, c2):
        return self._distance_correlation(c1, c2)


class _Wrapper(nn.Module):
    """Packs all attribute-branch sub-modules into a single nn.Module for save/load."""

    def __init__(self, name, nfeat, nhid, nclass, channels, dropout):
        super(_Wrapper, self).__init__()
        self.name = name
        self.nhid = nhid
        self.channels = channels
        per_channel_dim = nhid // channels

        self.encoder = _DisGCN(nfeat=nfeat, nhid=nhid,
                               chan_num=channels, layer_num=2, dropout=dropout)
        self.masker = _ChannelMasker(nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.channel_cls = nn.Linear(per_channel_dim, channels)

        self.encoder.init_parameters()
        self.encoder.init_edge_weight()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h_raw = self.encoder(x, edge_index)
        h_masked = self.masker(h_raw)
        output = self.fc(h_masked)
        return output, h_masked, h_raw


class AttributeModel(SingleBaseModel):
    """"""

    def __init__(self, args, data):
        super(AttributeModel, self).__init__(args, data)

        channels = getattr(args, 'channels', 4)
        assert args.hidden % channels == 0, (
            f"args.hidden ({args.hidden}) must be divisible by channels ({channels})"
        )

        model = _Wrapper(
            name=ATTRIBUTE,
            nfeat=data.x.shape[1],
            nhid=args.hidden,
            nclass=args.num_classes,
            channels=channels,
            dropout=args.dropout,
        )

        n = data.x.shape[0]
        row, col = data.knn_edge_index
        sparse_knn = SparseTensor(row=row, col=col, sparse_sizes=(n, n))

        self.model = model
        self.edge_index = sparse_knn
        self.channels = channels
        self.per_channel_dim = args.hidden // channels

        self.criterion_dc = _DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        self.criterion_mask = _FeatCov()

        self.optimizer_model = optim.Adam(
            list(model.encoder.parameters()) +
            list(model.masker.parameters()) +
            list(model.fc.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay)
        self.optimizer_channel_cls = optim.Adam(
            model.channel_cls.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)

    def to(self, device):
        self.model.to(device)
        if isinstance(self.edge_index, SparseTensor):
            self.edge_index = self.edge_index.to(device)
        self.data.to(device)
        if self.args.wd_loss:
            self.wd_approximator.to(device)

    def _build_downstream_embedding(self, h_masked, h_raw):
        downstream_mix = 0.7
        downstream_embedding = downstream_mix * h_raw + (1.0 - downstream_mix) * h_masked
        downstream_embedding = F.normalize(downstream_embedding, p=2, dim=1)
        return downstream_embedding

    def forward(self):
        data = self.data
        args = self.args
        fs_alpha = getattr(args, 'fs_alpha', 0.25)
        fs_beta = getattr(args, 'fs_beta', 0.25)

        self.model.train()

        output, h_masked, h_raw = self.model(data.x, self.edge_index)

        bce_loss = F.binary_cross_entropy_with_logits(
            output[data.idx_train],
            data.y[data.idx_train].unsqueeze(1).float().to(args.device))

        auc_roc_train = roc_auc_score(
            data.y[data.idx_train].cpu().numpy(),
            output[data.idx_train].detach().cpu().numpy())

        loss_chan = 0
        for i in range(self.channels):
            s, e = i * self.per_channel_dim, (i + 1) * self.per_channel_dim
            chan_out = self.model.channel_cls(h_masked[:, s:e])
            chan_tar = (torch.ones(chan_out.shape[0], dtype=torch.long) * i).to(args.device)
            loss_chan += self.criterion_mul_cls(chan_out, chan_tar)

        loss_disen = 0
        for i in range(self.channels):
            for j in range(i + 1, self.channels):
                loss_disen += self.criterion_dc(
                    h_masked[data.idx_train, i * self.per_channel_dim:(i + 1) * self.per_channel_dim],
                    h_masked[data.idx_train, j * self.per_channel_dim:(j + 1) * self.per_channel_dim])

        loss_mask = self.criterion_mask(
            h_masked[data.idx_train],
            data.sens[data.idx_train].float().to(args.device))

        loss_train = bce_loss + fs_alpha * (loss_chan + loss_disen) + fs_beta * loss_mask

        model_name = self.model.name
        self.log_dict.update({
            f"{model_name}_bce_loss_train": bce_loss,
            f"{model_name}_loss_train": loss_train,
        })

        downstream_embedding = self._build_downstream_embedding(h_masked, h_raw)
        return loss_train, downstream_embedding, auc_roc_train

    def eval(self, epoch):
        args = self.args
        data = self.data

        self.model.eval()
        output, h_masked, h_raw = self.model(data.x, self.edge_index)

        if args.dataset == 'twitter':
            best_threshold, parity_val, equality_val, f1_val, accuracy_val = metrics_utils.select_best_threshold(
                output=output, labels=data.y, sens=data.sens, idx=data.idx_val)
            bce_loss_val, auc_roc_val, parity_val, equality_val, f1_val, accuracy_val = metrics_utils.eval_metric_with_threshold(
                output=output, labels=data.y, sens=data.sens, idx=data.idx_val, args=args, threshold=best_threshold)
            _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = metrics_utils.eval_metric_with_threshold(
                output=output, labels=data.y, sens=data.sens, idx=data.idx_test, args=args, threshold=best_threshold)
            self._log_preds_ratio(output=output, idx=data.idx_val, annotation='val', threshold=best_threshold)
            self._log_preds_ratio(output=output, idx=data.idx_test, annotation='test', threshold=best_threshold)
        else:
            bce_loss_val, auc_roc_val, parity_val, equality_val, f1_val, accuracy_val = metrics_utils.eval_metric(
                output=output, labels=data.y, sens=data.sens, idx=data.idx_val, args=args)
            _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = metrics_utils.eval_metric(
                output=output, labels=data.y, sens=data.sens, idx=data.idx_test, args=args)
            self._log_preds_ratio(output=output, idx=data.idx_val, annotation='val')
            self._log_preds_ratio(output=output, idx=data.idx_test, annotation='test')

        tradeoff_val = self._get_tradeoff(accuracy_val, auc_roc_val, f1_val, parity_val, equality_val)
        loss_val = bce_loss_val

        model_name = self.model.name
        self.log_dict.update({
            f"{model_name}_logits": wandb.Histogram((output.squeeze() > 0).type_as(data.y).cpu()),
            f"{model_name}_bce_loss_val": bce_loss_val,
            f"{model_name}_loss_val": loss_val,
            f"{model_name}_tradeoff_val": tradeoff_val,
            f"{model_name}_f1_val": f1_val,
            f"{model_name}_auc_val": auc_roc_val,
            f"{model_name}_acc_val": accuracy_val,
            f"{model_name}_sp_val": parity_val,
            f"{model_name}_eo_val": equality_val,
            f"{model_name}_f1_test": f1_test,
            f"{model_name}_auc_test": auc_roc_test,
            f"{model_name}_acc_test": accuracy_test,
            f"{model_name}_sp_test": parity_test,
            f"{model_name}_eo_test": equality_test,
        })

        if self._get_best_condition(loss_val, tradeoff_val):
            self.best_loss = loss_val.item()
            self.best_tradeoff_val = tradeoff_val
            self.best_epoch = epoch
            self.best_embedding = self._build_downstream_embedding(h_masked, h_raw)

            WandbSingleton().wandb_log_without_step_inc(
                {f"{model_name}_best_tradeoff_val": tradeoff_val})
            if args.save:
                self.best_state_dict = deepcopy(self.model.state_dict())
                self.save(self.best_state_dict)

            self.summary_dict.update({
                f'{model_name}_best_bce_loss_val': bce_loss_val,
                f'{model_name}_best_acc_val': accuracy_val,
                f'{model_name}_best_f1_val': f1_val,
                f'{model_name}_best_auc_val': auc_roc_val,
                f'{model_name}_best_sp_val': parity_val,
                f'{model_name}_best_eo_val': equality_val,
                f'{model_name}_best_tradeoff_val': tradeoff_val,
                f'{model_name}_best_acc_test': accuracy_test,
                f'{model_name}_best_f1_test': f1_test,
                f'{model_name}_best_auc_test': auc_roc_test,
                f'{model_name}_best_sp_test': parity_test,
                f'{model_name}_best_eo_test': equality_test,
                f'{model_name}_best_loss_val': loss_val,
            })

        return loss_val, self._build_downstream_embedding(h_masked, h_raw), auc_roc_val

    def get_embeddings(self):
        """Return the masked attribute embedding for downstream MDFP fusion."""
        self.model.eval()
        _, h_masked, h_raw = self.model(self.data.x, self.edge_index)
        return self._build_downstream_embedding(h_masked, h_raw)

    def optimize_channel_classifier(self):
        if self.optimizer_channel_cls is not None:
            self.optimizer_channel_cls.step()
            self.optimizer_channel_cls.zero_grad()

    def optimize_wd_approximator(self):
        pass
