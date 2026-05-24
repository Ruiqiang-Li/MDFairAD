import torch.nn as nn


class WDapproximator(nn.Module):
    def __init__(self, nfeat, name=""):
        super(WDapproximator, self).__init__()
        self.name = name
        self.lin = nn.Linear(nfeat, 1)

    def forward(self, x):
        h = self.lin(x)
        return h
