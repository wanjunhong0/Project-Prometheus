import math
import torch
from torch.nn.parameter import Parameter


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.in_dims = in_dim
        self.out_dims = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias
