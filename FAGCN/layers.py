import torch


class FALayer(torch.nn.Module):
    """FAGCN layer
    """
    def __init__(self, in_dim):
        """
        Args:
            in_dim (int): input dimension
        """
        super(FALayer, self).__init__()
        self.gate = torch.nn.Linear(2 * in_dim, 1)

    def forward(self, input, adj):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix

        """
        edge_list = adj._indices()
        a_input = torch.cat([input[edge_list[0]], input[edge_list[1]]], dim=1)
        a = torch.tanh(self.gate(a_input)).squeeze()
        adj = torch.sparse_coo_tensor(adj._indices(), adj._values() * a, adj.size())

        output = torch.sparse.mm(adj, input)

        return output
