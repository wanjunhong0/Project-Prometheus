import torch
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(GCN, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            dim_in = n_feature if i == 0 else n_hidden
            dim_out = n_class if i == n_layer - 1 else n_hidden
            self.gcs.append(GraphConvolution(dim_in, dim_out))

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = feature
        for i in range(self.n_layer):
            x = self.gcs[i](x, adj)
            if i < self.n_layer - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)
