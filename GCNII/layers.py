import torch
import math
from torch.nn.parameter import Parameter


class GraphConvolutionII(torch.nn.Module):
    """GCNII layer
    """
    def __init__(self, in_dim, out_dim, alpha, theta):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(GraphConvolutionII, self).__init__()
        self.alpha = alpha
        self.theta = theta
        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))

        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, input, adj, lth, h0):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix
            lth (torch tensor): l-th layer

        Returns:
            (torch tensor): ((1 - alpha) * A * H + alpha * H0)((1 - beta) * I + beta * W)
        """
        beta = math.log(self.theta / lth + 1)
        h = torch.sparse.mm(adj, input)
        support = (1 - self.alpha) * h + self.alpha * h0
        output = beta * torch.mm(support, self.W) + (1 - beta) * support

        return output
