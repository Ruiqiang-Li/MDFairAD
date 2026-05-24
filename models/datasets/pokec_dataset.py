from models.datasets.base_dataset import BaseDataset
import pandas as pd
import scipy.sparse as sp
import torch
import numpy as np
import os
import random
from torch_geometric.utils import from_scipy_sparse_matrix
from models.helpers.parser_singleton import ParserSingleton


class _PokecDataset(BaseDataset):
    dataset_name = None
    label_number_default = None

    def _on_init_get_data_specification(self):
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        path = "dataset/pokec/"
        label_number = self.label_number_default
        sens_idx = 3
        return sens_attr, predict_attr, path, label_number, sens_idx

    def _on_init_feature_normalization(self, feature, sens_idx):
        return feature

    def _on_init_set_num_features_and_classes(self, data):
        ParserSingleton().args.num_features = data.x.shape[1]
        ParserSingleton().args.num_classes = 1

    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        idx_features_labels = pd.read_csv(os.path.join(path, f"{self.dataset_name}.csv"))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(path, f"{self.dataset_name}_relationship.txt"), dtype=int
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        edge_index, _ = from_scipy_sparse_matrix(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        labels[labels > 1] = 1

        seed = 20
        sens_number = 200
        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        split_ratio = [0.5, 0.25, 0.25]
        idx_train = label_idx[:min(int(split_ratio[0] * len(label_idx)), label_number)]
        idx_val = label_idx[
            int(split_ratio[0] * len(label_idx)):int((split_ratio[0] + split_ratio[1]) * len(label_idx))
        ]
        idx_test = label_idx[int((split_ratio[0] + split_ratio[1]) * len(label_idx)):]

        sens = idx_features_labels[sens_attr].values
        sens_idx_set = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx_set & set(idx_test)))
        sens = torch.FloatTensor(sens)

        idx_sens_train = list(sens_idx_set - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = idx_sens_train[:sens_number]
        _ = torch.LongTensor(idx_sens_train)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return features, adj, edge_index, labels, idx_train, idx_val, idx_test, sens


class PokecZDataset(_PokecDataset):
    dataset_name = "region_job"
    label_number_default = 4000


class PokecNDataset(_PokecDataset):
    dataset_name = "region_job_2"
    label_number_default = 3500
