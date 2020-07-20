import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GraphSAGE.aggregators import MeanAggregator


class SupervisedGraphSAGE(torch.nn.Module):
    """
    Simple supervised GraphSAGE model
    """
    def __init__(self, n_feature, n_hidden, n_class, n_sample, dropout):
        """Encodes a node's using 'convolutional' GraphSage approach

        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            n_sample (int): the number of neighbor that sampled for each node. No sampling if None.
            dropout (int): dropout rate
        """
        super(SupervisedGraphSAGE, self).__init__()
        self.agg1 = MeanAggregator(n_sample)
        self.gc1 = torch.nn.Linear(2 * n_feature, n_hidden, bias=False)
        self.agg2 = MeanAggregator(n_sample)
        self.gc2 = torch.nn.Linear(2 * n_hidden, n_hidden, bias=False)
        self.fc = torch.nn.Linear(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, nodes, feature, neighbor_list):
        """
        Args:
            nodes (torch Tensor): node id
            feature (torch Tensor): feature input
            neighbor_list (numpy array): neighbor id list for each node

        Returns:
            (torch Tensor): output layer of GraphSAGE
        """
        x = self.agg1(feature, neighbor_list)
        x = torch.cat([feature, x], dim=1)
        emb1 = F.relu(self.gc1(x))
        emb1 = F.dropout(emb1, self.dropout, training=self.training)
        # in order to fully propagation
        # only the last layer could computer target nodes for reducing time
        x = self.agg2(emb1, neighbor_list[nodes])
        x = torch.cat([emb1[nodes], x], dim=1)
        emb2 = F.relu(self.gc2(x))
        emb2 = F.dropout(emb2, self.dropout, training=self.training)
        scores = self.fc(emb2)
        return F.log_softmax(scores, dim=1)



