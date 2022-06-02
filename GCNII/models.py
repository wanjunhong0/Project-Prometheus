import torch
import torch.nn.functional as F
from layers import GraphConvolutionII


class GCNII(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout, alpha, theta):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            alpha (float): hyperparamter
            theta (float): hyperparamter
        """
        super(GCNII, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.lin = torch.nn.Linear(n_feature, n_hidden)
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            self.gcs.append(GraphConvolutionII(n_hidden, n_hidden, alpha, theta))
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
        h = h0 = F.relu(self.lin(x))

        for i in range(self.n_layer):
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(self.gcs[i](h, adj, i + 1, h0))
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fc(h)

        return F.log_softmax(out, dim=1)
