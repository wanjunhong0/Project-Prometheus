import torch


class Propagation(torch.nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()

    def forward(self, input, adj):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix

        Returns:
            (torch tensor): A * H
        """

        output = torch.sparse.mm(adj, input)

        return output
