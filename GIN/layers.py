import torch


class GraphConvolution(torch.nn.Module):
    """GIN layer
    """
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(GraphConvolution, self).__init__()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(out_dim, out_dim), torch.nn.ReLU())

    def forward(self, input, adj):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix

        Returns:
            (torch tensor): MLP((A + (1 + eps) * I) * X)
        """
        output = torch.sparse.mm(adj, input)
        output = self.mlp(output)

        return output
