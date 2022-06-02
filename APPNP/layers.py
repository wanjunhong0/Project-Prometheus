import torch
from torch.nn.parameter import Parameter


class Propagation(torch.nn.Module):
    def __init__(self, in_dim, out_dim, alpha):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            alpha (float): hyperparameter
        """
        super(Propagation, self).__init__()

        self.alpha = alpha
        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))

        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, input, adj, h):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix
            h (torch tensor): H0

        Returns:
            (torch tensor): (1 - alpha) * A * H + alpha * H0
        """

        output = (1 - self.alpha) * torch.sparse.mm(adj, input) + self.alpha * h

        return output
