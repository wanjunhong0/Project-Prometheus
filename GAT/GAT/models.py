import torch
import torch.nn.functional as F
from GAT.layers import GraphAttentionLayer


class GAT(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout, n_head):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            dropout (float): dropout rate
            n_head (int): the number of attention head
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        # multi-head graph attention
        self.attentions = [GraphAttentionLayer(n_feature, n_hidden) for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_attention = GraphAttentionLayer(n_hidden * n_head, n_class)

    def forward(self, x, edge_list):
        """
        Args:
            x (torch Tensor): feature input
            edge_list (torch Tensor): node index for every edge in graph

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_attention(x, edge_list)
        
        return F.log_softmax(x, dim=1)
