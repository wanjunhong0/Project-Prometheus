import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim

        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.a = Parameter(torch.FloatTensor(out_dim*2, 1))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, edge_list):
        """
        Args:
            input (torch Tensor): H hiddens 
            edge_list (torch Tensor): node index for every edge in graph

        Returns:
            (torch Tensor): H' hiddens after propagation using attention matrix
        """
        h = torch.mm(input, self.W)
        N = h.shape[0]

        # matrix form to calculate every W * h_i || W * h_j in edge_list
        a_input = torch.cat([h[edge_list[0], :], h[edge_list[1], :]], dim=1)
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=0.2).view(-1)
        e = torch.sparse.FloatTensor(edge_list, e, torch.Size([N, N]))
        attention = torch.sparse.softmax(e, dim=1)
        h_prime = torch.sparse.mm(attention, h)

        return F.elu(h_prime)
