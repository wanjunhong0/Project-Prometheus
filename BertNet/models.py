import torch
import torch.nn.functional as F
from layers import Bert_Propagation


class BertNet(torch.nn.Module):
    def __init__(self, k, n_feature, n_hidden, n_class, dropout):
        """
        Args:
            k (int): k-th Propagation
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(BertNet, self).__init__()

        self.dropout = dropout
        self.lin1 = torch.nn.Linear(n_feature, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, n_class)
        self.prop = Bert_Propagation(k)

    def forward(self, feature, adj, adj_):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix
            adj_ (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(feature, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.prop(x, adj, adj_)

        return F.log_softmax(x, dim=1)
