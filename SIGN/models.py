import torch
import torch.nn.functional as F


class SIGN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, n_layer, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            n_layer (int): the number of layers
            dropout (float): dropout rate
        """
        super(SIGN, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.lins = torch.nn.ModuleList()
        for _ in range(self.n_layer + 1):
            self.lins.append(torch.nn.Linear(n_feature, n_hidden))
        self.fc = torch.nn.Linear((n_layer + 1) * n_hidden, n_class)

    def forward(self, feature):
        """
        Args:
            feature (torch Tensor): feature input

        Returns:
            (torch Tensor): log probability for each class in label
        """
        xs = []
        for i in range(self.n_layer + 1):
            x = F.relu(self.lins[i](feature[i]))
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)
        xs = torch.cat(xs, dim=1)
        return F.log_softmax(x, dim=1)
