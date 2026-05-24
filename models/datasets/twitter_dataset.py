from models.datasets.pt_graph_dataset import PTGraphDataset


class TwitterDataset(PTGraphDataset):
    """"""

    def _on_init_get_data_specification(self):
        sens_attr    = "sensitive"
        predict_attr = "y"
        path         = "dataset/data/"
        label_number = 10000
        sens_idx     = 0
        return sens_attr, predict_attr, path, label_number, sens_idx
