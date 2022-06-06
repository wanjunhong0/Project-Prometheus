import torch
import torch.nn.functional as F
from layers import GraphConvolution


class GIN(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GIN, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            dim_in = n_feature if i == 0 else n_hidden
            self.gcs.append(GraphConvolution(dim_in, n_hidden))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(n_hidden, n_class), torch.nn.ReLU())

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = feature
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(self.n_layer):
            x = self.gcs[i](x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)
