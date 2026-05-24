import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, nhid, nclass, name=""):
        super(Classifier, self).__init__()
        self.name = name
        # self.lin1 = nn.Linear(nhid, nhid // 2)
        # self.lin2 = nn.Linear(nhid // 2, nhid // 4)
        # self.lin3 = nn.Linear(nhid // 4, nclass)
        self.lin1 = nn.Linear(nhid, nhid // 2)
        self.bn1 = nn.BatchNorm1d(nhid // 2)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(nhid // 2, nhid // 4)
        self.bn2 = nn.BatchNorm1d(nhid // 4)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.2)
        self.lin3 = nn.Linear(nhid // 4, nclass)

    def forward(self, h, edge_index=None):
        # h = self.lin1(h)
        # h = self.lin2(h)
        # h = self.lin3(h)
        h = self.lin1(h)
        h = self.drop1(self.relu1(self.bn1(h)))
        h = self.lin2(h)
        h = self.drop2(self.relu2(self.bn2(h)))
        h = self.lin3(h)

        return h