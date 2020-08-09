import math
import torch
from torch.nn.parameter import Parameter


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(GraphConvolution, self).__init__()
        self.in_dims = in_dim
        self.out_dims = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        Args:
            input (torch Tensor): H Hiddens
            adj (torch Tensor): L Laplacian matrix

        Returns:
            (torch Tensor): W * L * H + b
        """
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias
