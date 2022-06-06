import torch
import torch.nn.functional as F
from layers import GraphConvolution


class PairNorm(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout, scale, mode):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            scale (float): Row-normalization scale
            mode (str): Mode for PairNorm
        """
        super(PairNorm, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.scale = scale
        self.mode = mode
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            dim_in = n_feature if i == 0 else n_hidden
            dim_out = n_class if i == n_layer - 1 else n_hidden
            self.gcs.append(GraphConvolution(dim_in, dim_out))

    def pairnorm(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x

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
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gcs[i](x, adj)
            if i < self.n_layer - 1:
                x = self.pairnorm(x)
                x = F.relu(x)

        return F.log_softmax(x, dim=1)
