import torch
import torch.nn.functional as F
from layers import ChebConvolution


class ChebNet(torch.nn.Module):
    def __init__(self, k, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            k (int): k-th Propagation
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(ChebNet, self).__init__()

        self.dropout = dropout
        self.lin = torch.nn.Linear(n_feature, n_hidden)
        self.gc = ChebConvolution(n_hidden, n_hidden, k)
        self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(feature, self.dropout, training=self.training)
        x = F.relu(self.lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc(x, adj))
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
