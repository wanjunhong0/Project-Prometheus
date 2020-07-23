import torch
import torch.nn.functional as F
from GAT.layers import GraphAttentionLayer


class GAT(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout, n_head):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # multi-head graph attention
        self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout=dropout) for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hidden * n_head, n_class, dropout=dropout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


