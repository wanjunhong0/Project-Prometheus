import torch
import torch.nn.functional as F
from layers import FALayer


class FAGCN(torch.nn.Module):
    def __init__(self, n_layer, n_feature, n_hidden, n_class, dropout, eps):
        """
        Args:
            n_layer (int): the number of layer
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            eps (float): hyperparamter
        """
        super(FAGCN, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.eps = eps
        self.lin = torch.nn.Linear(n_feature, n_hidden)
        self.gcs = torch.nn.ModuleList()
        for i in range(n_layer):
            self.gcs.append(FALayer(n_hidden))
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
        x = h0 = F.dropout(x, self.dropout, training=self.training)

        for i in range(self.n_layer):
            x = F.relu(self.gcs[i](x, adj)) + self.eps * h0
        out = self.fc(x)

        return F.log_softmax(out, dim=1)
