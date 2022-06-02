import torch
import torch.nn.functional as F
from layers import GraphConvolution


class JKnet(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout, mode):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            mode (str): cat or max
        """
        super(JKnet, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            dim_in = n_feature if i == 0 else n_hidden
            self.gcs.append(GraphConvolution(dim_in, n_hidden))
        self.mode = mode
        if mode == 'cat':
            self.fc = torch.nn.Linear(n_layer * n_hidden, n_class)
        elif mode == 'max':
            self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = feature
        xs = []
        for i in range(self.n_layer):
            x = self.gcs[i](x, adj)
            if i < self.n_layer - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)

        if self.mode == 'cat':
            out = self.fc(torch.cat(xs, dim=-1))
        elif self.mode == 'max':
            out = self.fc(torch.stack(xs, dim=-1).max(dim=-1)[0])

        return F.log_softmax(out, dim=1)
