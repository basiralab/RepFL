import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

class Generator1(nn.Module):
    """
        Generator that predicts a target brain graph of resolution 160 from a source brain graph of resolution 35
        Input: Data object from torch_geometric.data describing a graph with resolution = 35
        Output: Node feature matrix x (which in this case is equal to the edge_attr parameter from Data object) representing the adjacency matrix with resolution = 160
    """

    def __init__(self):
        super(Generator1, self).__init__()

        nn = Sequential(Linear(1, 1225),ReLU())
        self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 5600), ReLU())
        self.conv2 = NNConv(35, 160, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(160, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3  = torch.matmul(x2.t(), x2)

        return x3

class Generator2(nn.Module):
    """
        Generator that predicts a target brain graph of resolution 268 from a source brain graph of resolution 35
        Input: Data object from torch_geometric.data describing a graph with resolution = 35
        Output: Node feature matrix x (which in this case is equal to the edge_attr parameter from Data object) representing the adjacency matrix with resolution = 268
    """

    def __init__(self):
        super(Generator2, self).__init__()

        nn = Sequential(Linear(1, 1225),ReLU())
        self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 5600), ReLU())
        self.conv2 = NNConv(35, 160, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(160, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 42880), ReLU())
        self.conv3 = NNConv(160, 268, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(268, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr)))
        x3 = F.dropout(x3, training=self.training)

        x4  = torch.matmul(x3.t(), x3)

        return x4
