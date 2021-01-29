import torch
import torch.nn.functional as F
from FastGCN.layers import GraphConvolution


class FastGCN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(FastGCN, self).__init__()

        self.dropout = dropout
        self.gc1 = GraphConvolution(n_feature, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.relu(self.gc1(feature, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
