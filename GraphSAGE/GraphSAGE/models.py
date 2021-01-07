import torch
import torch.nn.functional as F
from GraphSAGE.layers import Aggregator


class SupervisedGraphSAGE(torch.nn.Module):
    """
    Simple supervised GraphSAGE model
    """
    def __init__(self, n_feature, n_hidden, n_class, agg_type, dropout):
        """Encodes a node's using 'convolutional' GraphSage approach

        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            agg_type (str): the type of Aggregator
            dropout (int): dropout rate
        """
        super(SupervisedGraphSAGE, self).__init__()

        self.dropout = dropout
        self.agg1 = Aggregator(n_feature, n_hidden, agg_type)
        self.agg2 = Aggregator(n_hidden, n_hidden, agg_type)

        self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): normalized adjacency matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(feature, self.dropout, training=self.training)
        x = self.agg1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.agg2(x, adj)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)



