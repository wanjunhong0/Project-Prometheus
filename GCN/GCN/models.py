import torch
import torch.nn.functional as F
from GCN.layers import GraphConvolution


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_hiddens, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_features, n_hiddens)
        self.gc2 = GraphConvolution(n_hiddens, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
