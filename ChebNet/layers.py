import torch
from torch.nn.parameter import Parameter


class ChebConvolution(torch.nn.Module):
    def __init__(self, in_dim, out_dim, k):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(ChebConvolution, self).__init__()

        self.k = k
        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))

        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, input, adj):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix
        """
        xs = []
        for i in range(self.k):
            if i == 0:
                xs.append(torch.mm(input, self.W))
            elif i == 1:
                support = torch.sparse.mm(adj, input)
                xs.append(torch.mm(input, self.W))
            else:
                support = torch.sparse.mm(2 * adj, xs[i-1]) - xs[i-2]
                xs.append(torch.mm(support, self.W))

        return xs[-1]
