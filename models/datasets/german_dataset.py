from models.datasets.base_dataset import BaseDataset
import pandas as pd
import random
import scipy.sparse as sp
import torch
import numpy as np
import os
from torch_geometric.utils import from_scipy_sparse_matrix


class GermanDataset(BaseDataset):
    def _on_init_get_data_specification(self):
        sens_attr    = "Gender"
        predict_attr = "GoodCustomer"
        path         = "dataset/german/"
        label_number = 100
        sens_idx     = 0
        return sens_attr, predict_attr, path, label_number, sens_idx

    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')

        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt', usecols=(0, 1)).astype('int')
        else:
            edges_unordered = self._build_relationship(idx_features_labels[header], thresh=0.8)
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels   = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0

        N = features.shape[0]
        mask = (edges_unordered[:, 0] < N) & (edges_unordered[:, 1] < N)
        edges_unordered = edges_unordered[mask]

        adj = sp.coo_matrix(
            (np.ones(edges_unordered.shape[0]),
             (edges_unordered[:, 0], edges_unordered[:, 1])),
            shape=(N, N), dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        edge_index, _ = from_scipy_sparse_matrix(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels   = torch.LongTensor(labels)

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
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

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens      = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val   = torch.LongTensor(idx_val)
        idx_test  = torch.LongTensor(idx_test)

        return features, adj, edge_index, labels, idx_train, idx_val, idx_test, sens
