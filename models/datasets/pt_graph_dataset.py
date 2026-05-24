""""""
import os
import random
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from models.datasets.base_dataset import BaseDataset


class PTGraphDataset(BaseDataset):
    """"""

    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        pt_path = os.path.join(path, f"{dataset}.pt")
        print(f'Loading {dataset} from {pt_path}')
        pt = torch.load(pt_path, weights_only=False)

        features   = pt.x.float()                                # [N, F]
        labels     = pt.y.long()                                  # [N]
        sens       = pt.sensitive.float()                         # [N]
        edge_index = pt.edge_index.long()                         # [2, E]

        raw_c = pt.contamination
        self.contamination = float(raw_c.item() if torch.is_tensor(raw_c) else raw_c)

        N = features.shape[0]
        print(f'  nodes={N}, features={features.shape[1]}, '
              f'anomalies={int(labels.sum())}, contamination={self.contamination:.4f}')

        rows = edge_index[0].numpy()
        cols = edge_index[1].numpy()
        adj  = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(N, N), dtype=np.float32)
        adj  = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj  = adj + sp.eye(N)

        edge_index_sym, _ = from_scipy_sparse_matrix(adj)

        random.seed(20)
        label_idx_0 = np.where(labels.numpy() == 0)[0]
        label_idx_1 = np.where(labels.numpy() == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        print('label_number:', label_number)

        idx_train = np.append(
            label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)):],
            label_idx_1[int(0.75 * len(label_idx_1)):])

        idx_train = torch.LongTensor(idx_train)
        idx_val   = torch.LongTensor(idx_val)
        idx_test  = torch.LongTensor(idx_test)

        return features, adj, edge_index_sym, labels, idx_train, idx_val, idx_test, sens

    def _on_init_feature_normalization(self, feature, sens_idx):
        """"""
        return self._feature_norm(feature)
