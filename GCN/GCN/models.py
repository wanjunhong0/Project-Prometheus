import torch
import torch.nn.functional as F
from GCN.layers import GraphConvolution


class GCN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_feature, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Args:
            x (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): output layer of GCN
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
