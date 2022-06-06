import torch
import torch.nn.functional as F
from layers import Propagation


class GPRGNN(torch.nn.Module):
    def __init__(self, k, n_feature, n_hidden, n_class, dropout, alpha):
        """
        Args:
            k (int): k-th Propagation
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            alpha (float): hyperparameter
        """
        super(GPRGNN, self).__init__()

        self.k = k
        self.dropout = dropout
        self.gamma = alpha * (1 - alpha) ** torch.arange(k + 1)
        self.gamma[-1] = (1 - alpha) ** k
        self.lin1 = torch.nn.Linear(n_feature, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, n_class)
        self.props = torch.nn.ModuleList()
        for i in range(k):
            self.props.append(Propagation())

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(feature, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        out = x * self.gamma[0]
        for i in range(self.k):
            x = self.props[i](x, adj)
            out = x * self.gamma[i+1] + out

        return F.log_softmax(out, dim=1)
