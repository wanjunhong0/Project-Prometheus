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

        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.b = Parameter(torch.FloatTensor(out_dim))
        
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.zeros_(self.b)

    def forward(self, input, adj):
        """
        Args:
            input (torch Tensor): H Hiddens
            adj (torch Tensor): L Laplacian matrix

        Returns:
            (torch Tensor): W * L * H + b
        """
        support = torch.mm(input, self.W)
        output = torch.sparse.mm(adj, support)
        return output + self.b
