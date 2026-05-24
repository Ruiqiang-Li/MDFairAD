import pandas as pd
import random
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import numpy as np
from abc import ABCMeta, abstractmethod
from torch_geometric.utils import from_scipy_sparse_matrix
from constants import VANILLA
from models.helpers.parser_singleton import ParserSingleton
from torch_geometric.data import Data


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, dataname):
        sens_attr, predict_attr, path, label_number, sens_idx = self._on_init_get_data_specification()
        feature, adj, edge_index, labels, idx_train, idx_val, idx_test, sens = self.load_data(dataset=dataname,
                                                                                              sens_attr=sens_attr,
                                                                                              predict_attr=predict_attr,
                                                                                              path=path,
                                                                                              label_number=label_number)
        self.feature = self._on_init_feature_normalization(feature, sens_idx)
        self.adj = adj
        self.edge_index = edge_index
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx

        self.data = Data(x=self.feature,
                         edge_index=self.edge_index,
                         y=self.labels.float(),
                         idx_train=self.idx_train,
                         idx_val=self.idx_val,
                         idx_test=self.idx_test,
                         sens=self.sens)

        if ParserSingleton().args.model != VANILLA:
            self.data.knn_edge_index = self.__on_init_make_knn_edge_index(feature)
        else:
            self.data.knn_edge_index = None

        self._on_init_set_num_features_and_classes(self.data)

    @abstractmethod
    def _on_init_get_data_specification(self):
        raise NotImplementedError

    @abstractmethod
    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        raise NotImplementedError

    def _on_init_feature_normalization(self, feature, sens_idx):
        norm_features = self._feature_norm(feature)
        norm_features[:, sens_idx] = feature[:, sens_idx]
        return norm_features

    def _on_init_set_num_features_and_classes(self, data):
        ParserSingleton().args.num_features = data.x.shape[1]
        ParserSingleton().args.num_classes = len(data.y.unique()) - 1

    def __on_init_make_knn_edge_index(self, feature):
        k = ParserSingleton().args.k

        nearest_neighbors = NearestNeighbors(n_neighbors=k)

        nearest_neighbors.fit(feature)

        adjacency_matrix_sparse = nearest_neighbors.kneighbors_graph(feature, mode='connectivity')

        adjacency_matrix_transposed = adjacency_matrix_sparse.transpose()
        adjacency_matrix_symmetric = adjacency_matrix_sparse.maximum(adjacency_matrix_transposed)

        knn_adj = sp.coo_matrix(adjacency_matrix_symmetric, dtype=np.float32)

        knn_edge_index, _ = from_scipy_sparse_matrix(knn_adj)

        return knn_edge_index

    @staticmethod
    def _build_relationship(x, thresh=0.25):
        df_euclid = pd.DataFrame(
            1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
            random.seed(912)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        idx_map = np.array(idx_map)

        return idx_map

    @staticmethod
    def _feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1